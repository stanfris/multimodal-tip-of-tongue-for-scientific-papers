"""Resumable MinerU API extraction for large PDF corpora."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
import zipfile
from io import BytesIO
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from PIL import Image


DEFAULT_API_URL = "http://127.0.0.1:8002"
DEFAULT_INPUT_DIR = Path("data") / "acl_subset" / "pdfs"
DEFAULT_OUTPUT_DIR = Path("data") / "processed" / "mineru_pdf_extraction"
FIGURE_TYPES = {"image", "chart"}
TERMINAL_SUCCESS = {"completed", "complete", "success", "succeeded", "done"}
TERMINAL_FAILURE = {"failed", "failure", "error", "cancelled", "canceled"}


@dataclass(frozen=True)
class MinerUOptions:
    api_url: str = DEFAULT_API_URL
    backend: str = "hybrid-engine"
    effort: str = "medium"
    parse_method: str = "auto"
    lang: str = "ch"
    formula: bool = True
    table: bool = True
    image_analysis: bool | None = None
    max_in_flight: int = 4
    poll_interval: float = 2.0
    request_timeout: float = 120.0
    result_timeout: float = 3600.0
    retries: int = 2
    min_markdown_chars: int = 200
    allow_tiny_markdown: bool = False


@dataclass
class ExtractionStats:
    total: int = 0
    already_complete: int = 0
    submitted: int = 0
    in_flight: int = 0
    completed: int = 0
    failed: int = 0
    pages: int = 0
    figures: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        processed = self.completed + self.failed
        return {
            "total": self.total,
            "already_complete": self.already_complete,
            "submitted": self.submitted,
            "in_flight": self.in_flight,
            "completed": self.completed,
            "failed": self.failed,
            "remaining": max(self.total - self.already_complete - processed, 0),
            "pages": self.pages,
            "figures": self.figures,
            "elapsed_seconds": round(elapsed, 3),
            "papers_per_second": round(self.completed / elapsed, 4),
            "pages_per_second": round(self.pages / elapsed, 4),
        }


class ExtractionError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.attempts = 1


def build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PDFs through a persistent MinerU API/router.")
    add_common_args(parser, include_max_in_flight=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of PDFs to process.")
    parser.add_argument("--start-index", type=int, default=0, help="First discovered PDF index to process.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive discovered PDF index to process.")
    return parser


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MinerU medium-effort extraction on a representative PDF subset."
    )
    add_common_args(parser, include_max_in_flight=False)
    parser.set_defaults(effort="medium")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of PDFs per benchmark run.")
    parser.add_argument(
        "--max-in-flight-values",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Client-side in-flight task counts to benchmark.",
    )
    return parser


def add_common_args(parser: argparse.ArgumentParser, *, include_max_in_flight: bool) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing PDFs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Extraction output directory.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="MinerU API or router base URL.")
    parser.add_argument("--backend", default="hybrid-engine", help="MinerU backend.")
    parser.add_argument("--effort", default="medium", choices=["medium"], help="Hybrid parsing effort.")
    parser.add_argument("--parse-method", default="auto", choices=["auto", "txt", "ocr"], help="MinerU parse method.")
    parser.add_argument("--lang", default="ch", help="OCR language hint for pipeline/hybrid backends.")
    if include_max_in_flight:
        parser.add_argument("--max-in-flight", type=int, default=4, help="Maximum submitted MinerU tasks in flight.")
    else:
        parser.set_defaults(max_in_flight=4)
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between task status polls.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Submit/status HTTP timeout.")
    parser.add_argument("--result-timeout", type=float, default=3600.0, help="Per-task terminal-state timeout.")
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first failed attempt.")
    parser.add_argument("--min-markdown-chars", type=int, default=200, help="Minimum successful Markdown length.")
    parser.add_argument("--allow-tiny-markdown", action="store_true", help="Warn instead of failing on tiny Markdown.")
    parser.add_argument("--no-formula", action="store_true", help="Disable formula parsing.")
    parser.add_argument("--no-table", action="store_true", help="Disable table parsing.")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--image-analysis", dest="image_analysis", action="store_true")
    image_group.add_argument("--no-image-analysis", dest="image_analysis", action="store_false")
    parser.set_defaults(image_analysis=None)


def options_from_args(args: argparse.Namespace) -> MinerUOptions:
    return MinerUOptions(
        api_url=args.api_url.rstrip("/"),
        backend=args.backend,
        effort=args.effort,
        parse_method=args.parse_method,
        lang=args.lang,
        formula=not args.no_formula,
        table=not args.no_table,
        image_analysis=args.image_analysis,
        max_in_flight=max(args.max_in_flight, 1),
        poll_interval=args.poll_interval,
        request_timeout=args.request_timeout,
        result_timeout=args.result_timeout,
        retries=max(args.retries, 0),
        min_markdown_chars=max(args.min_markdown_chars, 0),
        allow_tiny_markdown=args.allow_tiny_markdown,
    )


def run_extract(args: argparse.Namespace) -> Path:
    pdfs = select_pdfs(
        discover_pdfs(args.input_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        limit=args.limit,
    )
    stats = asyncio.run(extract_many(pdfs, args.output_dir, options_from_args(args)))
    print(json.dumps({"output": str(args.output_dir), "stats": stats.snapshot()}, indent=2, sort_keys=True))
    return args.output_dir


def run_benchmark(args: argparse.Namespace) -> Path:
    pdfs = select_pdfs(discover_pdfs(args.input_dir), limit=args.sample_size)
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {args.input_dir}")

    output_dir = args.output_dir / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for max_in_flight in args.max_in_flight_values:
        run_args = argparse.Namespace(**vars(args))
        run_args.effort = "medium"
        run_args.max_in_flight = max(max_in_flight, 1)
        run_output = output_dir / f"{args.backend}_medium_inflight_{run_args.max_in_flight}"
        started = time.monotonic()
        stats = asyncio.run(extract_many(pdfs, run_output, options_from_args(run_args)))
        elapsed = time.monotonic() - started
        results.append(
            {
                "backend": args.backend,
                "effort": "medium",
                "max_in_flight": run_args.max_in_flight,
                "pdfs": len(pdfs),
                "wall_time_seconds": round(elapsed, 3),
                **stats.snapshot(),
            }
        )
    best = recommend_benchmark_result(results)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "backend": args.backend,
        "effort": "medium",
        "sample_pdfs": [str(path) for path in pdfs],
        "environment": probe_environment(),
        "recommendation": best,
        "results": results,
    }
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return output_dir


def recommend_benchmark_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [row for row in results if row.get("completed", 0) > 0 and row.get("failed", 0) == 0]
    candidates = successful or [row for row in results if row.get("completed", 0) > 0]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            float(row.get("papers_per_second", 0.0)),
            float(row.get("pages_per_second", 0.0)),
        ),
    )
    return {
        "backend": best.get("backend"),
        "effort": best.get("effort"),
        "max_in_flight": best.get("max_in_flight"),
        "papers_per_second": best.get("papers_per_second"),
        "pages_per_second": best.get("pages_per_second"),
        "note": "Use this as the starting production setting, then confirm on a larger sample.",
    }


def probe_environment() -> dict[str, Any]:
    packages = {}
    for name in ("mineru", "magic-pdf", "vllm", "torch", "httpx", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": _run_probe(["uname", "-m"]),
        "commands": {
            command: shutil.which(command)
            for command in ("mineru", "mineru-api", "mineru-router", "mineru-openai-server", "nvidia-smi")
        },
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _run_probe(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "torch": _probe_torch(),
    }
    return env


async def extract_many(pdfs: list[Path], output_dir: Path, options: MinerUOptions) -> ExtractionStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "papers").mkdir(exist_ok=True)
    (output_dir / "_tmp").mkdir(exist_ok=True)
    stats = ExtractionStats(total=len(pdfs))
    pending = [pdf for pdf in pdfs if not is_complete(output_dir, paper_id_for_pdf(pdf))]
    stats.already_complete = len(pdfs) - len(pending)
    if not pending:
        return stats

    timeout = httpx.Timeout(options.request_timeout, read=options.request_timeout)
    async with httpx.AsyncClient(base_url=options.api_url, timeout=timeout, follow_redirects=True) as client:
        health = await get_health(client)
        write_json(output_dir / "mineru_health.json", health)
        queue: asyncio.Queue[Path | None] = asyncio.Queue(maxsize=options.max_in_flight * 2)

        async def worker() -> None:
            while True:
                pdf = await queue.get()
                if pdf is None:
                    queue.task_done()
                    return
                stats.submitted += 1
                stats.in_flight += 1
                try:
                    record = await extract_one_with_retries(client, pdf, output_dir, options)
                except ExtractionError as exc:
                    stats.failed += 1
                    append_failure(output_dir, pdf, exc.error_type, str(exc), exc.attempts)
                else:
                    stats.completed += 1
                    stats.pages += int(record.get("num_pages") or 0)
                    stats.figures += len(record.get("figures", []))
                finally:
                    stats.in_flight -= 1
                    print_progress(stats)
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(options.max_in_flight)]
        for pdf in pending:
            await queue.put(pdf)
        for _ in workers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*workers)
    return stats


async def extract_one_with_retries(
    client: httpx.AsyncClient,
    pdf: Path,
    output_dir: Path,
    options: MinerUOptions,
) -> dict[str, Any]:
    attempts = options.retries + 1
    last: ExtractionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await extract_one(client, pdf, output_dir, options, attempt=attempt)
        except ExtractionError as exc:
            last = exc
            last.attempts = attempt
            if exc.error_type in {"invalid_pdf", "empty_output", "missing_images", "malformed_structured_output"}:
                break
            if attempt < attempts:
                await asyncio.sleep(min(2 ** attempt, 30))
        except Exception as exc:
            last = ExtractionError("unknown", str(exc))
            last.attempts = attempt
            if attempt < attempts:
                await asyncio.sleep(min(2 ** attempt, 30))
    assert last is not None
    raise last


async def extract_one(
    client: httpx.AsyncClient,
    pdf: Path,
    output_dir: Path,
    options: MinerUOptions,
    *,
    attempt: int,
) -> dict[str, Any]:
    if not has_pdf_header(pdf):
        raise ExtractionError("invalid_pdf", f"File does not look like a PDF: {pdf}")

    paper_id = paper_id_for_pdf(pdf)
    final_dir = output_dir / "papers" / paper_id
    tmp_dir = output_dir / "_tmp" / f"{paper_id}.{os.getpid()}.{attempt}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        task_id = await submit_task(client, pdf, options)
        await wait_for_task(client, task_id, options)
        result_bytes = await fetch_result(client, task_id, options)
        raw_dir = tmp_dir / "mineru"
        extract_result_zip(result_bytes, raw_dir)
        record = validate_and_normalize(raw_dir, pdf, tmp_dir, final_dir, options)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.rename(final_dir)
        success_path = final_dir / "_SUCCESS"
        success_path.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        return json.loads((final_dir / "paper.json").read_text(encoding="utf-8"))
    except httpx.HTTPError as exc:
        raise ExtractionError("submission_failure", str(exc)) from exc
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


async def get_health(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/health")
    response.raise_for_status()
    try:
        health = response.json()
    except json.JSONDecodeError as exc:
        raise ExtractionError("submission_failure", f"Malformed /health response: {exc}") from exc
    print(json.dumps({"mineru_health": health}, indent=2, sort_keys=True))
    return health


async def submit_task(client: httpx.AsyncClient, pdf: Path, options: MinerUOptions) -> str:
    data: dict[str, Any] = {
        "backend": options.backend,
        "effort": options.effort,
        "parse_method": options.parse_method,
        "lang_list": options.lang,
        "formula_enable": str(options.formula).lower(),
        "table_enable": str(options.table).lower(),
        "return_md": "true",
        "return_middle_json": "false",
        "return_model_output": "false",
        "return_content_list": "true",
        "return_images": "true",
        "response_format_zip": "true",
        "return_original_file": "false",
        "client_side_output_generation": "false",
        "start_page_id": "0",
        "end_page_id": "99999",
    }
    if options.image_analysis is not None:
        data["image_analysis"] = str(options.image_analysis).lower()
    with pdf.open("rb") as handle:
        response = await client.post("/tasks", data=data, files={"files": (pdf.name, handle, "application/pdf")})
    response.raise_for_status()
    payload = response.json()
    task_id = payload.get("task_id") or payload.get("id")
    if not task_id:
        raise ExtractionError("submission_failure", f"MinerU did not return task_id: {payload}")
    return str(task_id)


async def wait_for_task(client: httpx.AsyncClient, task_id: str, options: MinerUOptions) -> None:
    deadline = time.monotonic() + options.result_timeout
    while time.monotonic() < deadline:
        response = await client.get(f"/tasks/{task_id}")
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status") or payload.get("state") or "").lower()
        if status in TERMINAL_SUCCESS:
            return
        if status in TERMINAL_FAILURE:
            message = payload.get("error") or payload.get("message") or payload
            raise ExtractionError("mineru_failure", f"Task {task_id} failed: {message}")
        await asyncio.sleep(options.poll_interval)
    raise ExtractionError("timeout", f"Timed out waiting for MinerU task {task_id}")


async def fetch_result(client: httpx.AsyncClient, task_id: str, options: MinerUOptions) -> bytes:
    timeout = httpx.Timeout(options.request_timeout, read=options.request_timeout)
    response = await client.get(f"/tasks/{task_id}/result", timeout=timeout)
    response.raise_for_status()
    content = response.content
    if not zipfile.is_zipfile(BytesIO(content)):
        raise ExtractionError("malformed_structured_output", "MinerU result endpoint did not return a ZIP payload")
    return content


def validate_and_normalize(
    raw_dir: Path,
    pdf: Path,
    work_dir: Path,
    final_dir: Path,
    options: MinerUOptions,
) -> dict[str, Any]:
    markdown_path = choose_markdown(raw_dir)
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace").strip()
    if len(markdown) < options.min_markdown_chars and not options.allow_tiny_markdown:
        raise ExtractionError("empty_output", f"Markdown too small for {pdf}: {len(markdown)} characters")

    content_paths = find_content_lists(raw_dir)
    if not content_paths:
        raise ExtractionError("malformed_structured_output", f"No content_list JSON found under {raw_dir}")
    try:
        content_data = {path.name: read_json(path) for path in content_paths}
    except json.JSONDecodeError as exc:
        raise ExtractionError("malformed_structured_output", f"Could not parse MinerU content list JSON: {exc}") from exc

    paper_id = paper_id_for_pdf(pdf)
    markdown_relpath = Path("markdown.md")
    stable_markdown_path = work_dir / markdown_relpath
    stable_markdown_path.write_text(markdown + "\n", encoding="utf-8")

    preferred_content = choose_structured_content(content_data)
    figures = normalize_figures(preferred_content, content_paths[0].parent, paper_id, pdf, work_dir, final_dir)
    num_pages = estimate_page_count(preferred_content)
    record = {
        "paper_id": paper_id,
        "source_pdf": str(pdf.resolve()),
        "markdown_relpath": markdown_relpath.as_posix(),
        "markdown_sha256": sha256_bytes((markdown + "\n").encode("utf-8")),
        "mineru_version": package_version("mineru") or package_version("magic-pdf"),
        "backend": options.backend,
        "effort": options.effort,
        "parse_method": options.parse_method,
        "num_pages": num_pages,
        "figures": figures,
        "structured_outputs": {
            path.name: str(path.relative_to(work_dir).as_posix()) for path in content_paths if path.is_relative_to(work_dir)
        },
        "mineru_raw_relpath": "mineru",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(work_dir / "paper.json", record)
    write_json(work_dir / "figures.json", figures)
    return record


def normalize_figures(
    data: Any,
    base_dir: Path,
    paper_id: str,
    pdf: Path,
    work_dir: Path,
    final_dir: Path,
) -> list[dict[str, Any]]:
    figures = []
    for index, block in enumerate(iter_content_blocks(data)):
        block_type = str(block.get("type") or "").lower()
        sub_type = block.get("sub_type")
        if block_type not in FIGURE_TYPES:
            continue
        img_path = image_path_from_block(block)
        if not img_path:
            continue
        resolved_image = resolve_image_path(img_path, base_dir)
        if not resolved_image.exists():
            raise ExtractionError("missing_images", f"Referenced image does not exist for {pdf}: {img_path}")
        width, height = image_dimensions(resolved_image)
        figure_id = stable_figure_id(paper_id, block.get("page_idx"), index, img_path)
        image_relpath = (
            resolved_image.relative_to(work_dir).as_posix()
            if resolved_image.is_relative_to(work_dir)
            else str(resolved_image)
        )
        image_path = final_dir / image_relpath if not Path(image_relpath).is_absolute() else Path(image_relpath)
        figures.append(
            {
                "paper_id": paper_id,
                "source_pdf": str(pdf.resolve()),
                "figure_id": figure_id,
                "type": block_type,
                "sub_type": sub_type,
                "page_idx": block.get("page_idx"),
                "bbox": block.get("bbox"),
                "caption": caption_from_block(block),
                "footnote": footnote_from_block(block),
                "image_relpath": image_relpath,
                "image_path": str(image_path.resolve()),
                "image_width": width,
                "image_height": height,
            }
        )
    return figures


def iter_content_blocks(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, list) and data and all(isinstance(page, list) for page in data):
        for page_idx, page in enumerate(data):
            for block in page:
                if isinstance(block, dict):
                    normalized = flatten_v2_block(block)
                    normalized.setdefault("page_idx", page_idx)
                    yield normalized
        return
    if isinstance(data, list):
        for block in data:
            if isinstance(block, dict):
                yield flatten_v2_block(block)


def flatten_v2_block(block: dict[str, Any]) -> dict[str, Any]:
    result = dict(block)
    content = result.get("content")
    if isinstance(content, dict):
        for key, value in content.items():
            result.setdefault(key, value)
    return result


def image_path_from_block(block: dict[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def caption_from_block(block: dict[str, Any]) -> list[str]:
    block_type = str(block.get("type") or "image").lower()
    return text_list(block.get(f"{block_type}_caption") or block.get("image_caption") or block.get("caption"))


def footnote_from_block(block: dict[str, Any]) -> list[str]:
    block_type = str(block.get("type") or "image").lower()
    return text_list(block.get(f"{block_type}_footnote") or block.get("image_footnote") or block.get("footnote"))


def text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        text = span_text(value)
        return [text] if text else []
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            text = span_text(value)
            return [text] if text else []
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def span_text(value: Any) -> str:
    if isinstance(value, dict):
        if "content" in value and not isinstance(value["content"], (list, dict)):
            return str(value["content"]).strip()
        if "children" in value:
            return span_text(value["children"])
        return " ".join(filter(None, (span_text(item) for item in value.values()))).strip()
    if isinstance(value, list):
        return " ".join(filter(None, (span_text(item) for item in value))).strip()
    return str(value).strip() if value is not None else ""


def choose_structured_content(content_data: dict[str, Any]) -> Any:
    for name in sorted(content_data):
        if "content_list_v2" in name:
            return content_data[name]
    for name in sorted(content_data):
        if "content_list" in name:
            return content_data[name]
    return next(iter(content_data.values()))


def estimate_page_count(data: Any) -> int | None:
    if isinstance(data, list) and data and all(isinstance(page, list) for page in data):
        return len(data)
    pages = [block.get("page_idx") for block in iter_content_blocks(data) if isinstance(block.get("page_idx"), int)]
    return max(pages) + 1 if pages else None


def discover_pdfs(input_dir: Path) -> list[Path]:
    if input_dir.is_file() and input_dir.suffix.lower() == ".pdf":
        return [input_dir]
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def select_pdfs(pdfs: list[Path], *, start_index: int = 0, end_index: int | None = None, limit: int | None = None) -> list[Path]:
    selected = pdfs[start_index:end_index]
    if limit is not None:
        selected = selected[:limit]
    return selected


def paper_id_for_pdf(pdf: Path) -> str:
    return pdf.stem


def is_complete(output_dir: Path, paper_id: str) -> bool:
    paper_dir = output_dir / "papers" / paper_id
    if not (paper_dir / "_SUCCESS").exists() or not (paper_dir / "paper.json").exists():
        return False
    try:
        record = read_json(paper_dir / "paper.json")
    except (OSError, json.JSONDecodeError):
        return False
    markdown_relpath = record.get("markdown_relpath")
    if not isinstance(markdown_relpath, str) or not (paper_dir / markdown_relpath).exists():
        return False
    for figure in record.get("figures", []):
        if not isinstance(figure, dict):
            return False
        image_relpath = figure.get("image_relpath")
        if not isinstance(image_relpath, str):
            return False
        image_path = Path(image_relpath)
        if not image_path.is_absolute():
            image_path = paper_dir / image_path
        if not image_path.exists():
            return False
    return True


def choose_markdown(raw_dir: Path) -> Path:
    candidates = [path for path in raw_dir.rglob("*.md") if path.is_file()]
    non_empty = [path for path in candidates if path.stat().st_size > 0]
    if not non_empty:
        raise ExtractionError("empty_output", f"No non-empty Markdown file found under {raw_dir}")
    return max(non_empty, key=lambda path: path.stat().st_size)


def find_content_lists(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*content_list*.json") if path.is_file())


def resolve_image_path(img_path: str, base_dir: Path) -> Path:
    candidate = Path(img_path)
    if candidate.is_absolute():
        return candidate
    local = base_dir / candidate
    if local.exists():
        return local
    for root in (base_dir, base_dir.parent, base_dir.parent.parent):
        found = root / candidate
        if found.exists():
            return found
    return local


def extract_result_zip(content: bytes, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "result.zip"
        zip_path.write_bytes(content)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    destination = output_dir / member.filename
                    if not destination.resolve().is_relative_to(output_dir.resolve()):
                        raise ExtractionError("malformed_structured_output", f"Unsafe ZIP member: {member.filename}")
                    archive.extract(member, output_dir)
        except zipfile.BadZipFile as exc:
            raise ExtractionError("malformed_structured_output", f"MinerU result ZIP is invalid: {exc}") from exc


def append_failure(output_dir: Path, pdf: Path, error_type: str, message: str, attempts: int) -> None:
    failure = {
        "source": str(pdf.resolve()),
        "paper_id": paper_id_for_pdf(pdf),
        "error_type": error_type,
        "message": message,
        "attempts": attempts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


def print_progress(stats: ExtractionStats) -> None:
    snapshot = stats.snapshot()
    print(
        "progress "
        f"completed={snapshot['completed']} failed={snapshot['failed']} "
        f"in_flight={snapshot['in_flight']} remaining={snapshot['remaining']} "
        f"papers/s={snapshot['papers_per_second']} "
        f"pages/s={snapshot['pages_per_second']}",
        flush=True,
    )


def has_pdf_header(path: Path) -> bool:
    try:
        return path.read_bytes()[:5] == b"%PDF-"
    except OSError:
        return False


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def stable_figure_id(paper_id: str, page_idx: Any, block_index: int, img_path: str) -> str:
    key = f"{paper_id}|{page_idx}|{block_index}|{img_path}"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{paper_id}.fig.{block_index:04d}.{suffix}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _run_probe(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _probe_torch() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": props.name, "memory_total": props.total_memory})
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "gpus": gpus,
    }
