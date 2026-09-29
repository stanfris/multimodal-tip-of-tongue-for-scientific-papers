#!/usr/bin/env python3
"""Analyze preprocessed MinerU outputs without truncating document content."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from preprocessing.preprocessed import select_preprocessed_paper_dirs


EQUATION_TYPES = {"equation", "equation_interline", "equation_inline"}
IMAGE_LIKE_TYPES = {"image", "chart", "table"}


def main() -> None:
    args = parse_args()
    root = args.preprocessed_dir
    if not root.exists():
        raise SystemExit(f"Preprocessed directory does not exist: {root}")

    aggregate_block_types: Counter[str] = Counter()
    aggregate_image_types: Counter[str] = Counter()
    aggregate_image_extensions: Counter[str] = Counter()
    aggregate_equation_types: Counter[str] = Counter()
    equation_count = 0
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []

    print(
        json.dumps(
            {
                "stage": "reduce",
                "status": "selecting",
                "source": str(args.split_index) if args.split_index else "recursive_scan",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selection = select_preprocessed_paper_dirs(
        root,
        split_index=args.split_index,
        split=args.split,
    )
    paper_dirs = selection.paper_dirs
    total_papers = len(paper_dirs)
    print(
        json.dumps(
            {
                "stage": "reduce",
                "status": "started",
                "total": total_papers,
                "expected": selection.expected_count,
                "not_extracted": selection.missing_count,
                "selection_source": selection.source,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(reduce_paper_dir, paper_dir, args.dry_run) for paper_dir in paper_dirs]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                paper_summary = future.result()
            except Exception as exc:
                failures.append({"paper_dir": "<unknown>", "error": str(exc)})
                if args.fail_fast:
                    raise
                continue

            if paper_summary["status"] == "skipped":
                skipped += 1
                continue
            if paper_summary["status"] == "failed":
                failures.append({"paper_dir": paper_summary["paper_dir"], "error": paper_summary["error"]})
                if args.fail_fast:
                    raise RuntimeError(f"{paper_summary['paper_dir']}: {paper_summary['error']}")
                continue

            processed += 1
            aggregate_block_types.update(paper_summary["block_type_counts"])
            aggregate_image_types.update(paper_summary["image_like_type_counts"])
            aggregate_image_extensions.update(paper_summary["image_extensions"])
            aggregate_equation_types.update(paper_summary["equation_type_counts"])
            equation_count += paper_summary["equation_count"]
            if should_report_progress(index, total_papers, args.progress_every):
                print(
                    json.dumps(
                        {
                            "stage": "reduce",
                            "status": "running" if index < total_papers else "complete",
                            "seen": index,
                            "total": total_papers,
                            "percent": round((index / total_papers) * 100, 1) if total_papers else 100.0,
                            "processed": processed,
                            "skipped": skipped,
                            "failures": len(failures),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    report = {
        "preprocessed_dir": str(root),
        "dry_run": args.dry_run,
        "paper_count": processed,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "failures": failures,
        "aggregate": {
            "block_type_counts": dict(sorted(aggregate_block_types.items())),
            "image_like_type_counts": dict(sorted(aggregate_image_types.items())),
            "image_extensions": dict(sorted(aggregate_image_extensions.items())),
            "equation_type_counts": dict(sorted(aggregate_equation_types.items())),
            "equation_count": equation_count,
        },
    }

    if args.report:
        if not args.dry_run:
            write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report["aggregate"], indent=2, sort_keys=True))

    if failures:
        raise SystemExit(1)


def reduce_paper_dir(paper_dir: Path, dry_run: bool) -> dict[str, Any]:
    try:
        if is_compacted_paper_dir(paper_dir):
            return {"status": "skipped", "paper_dir": str(paper_dir), "reason": "already_compacted"}

        paper_path = paper_dir / "paper.json"
        markdown_path = paper_dir / "markdown.md"
        figures_path = paper_dir / "figures.json"
        if not paper_path.exists() or not markdown_path.exists():
            return {"status": "skipped", "paper_dir": str(paper_dir)}

        paper = read_json(paper_path)
        paper_id = str(paper.get("paper_id") or paper_dir.name)
        structured_outputs = paper.get("structured_outputs") or {}

        content_results: list[dict[str, Any]] = []
        for name, relpath in sorted(structured_outputs.items()):
            content_path = paper_dir / str(relpath)
            if not content_path.exists():
                continue

            content = read_json(content_path)
            if is_v2_content_list(content):
                summary, found_equations = summarize_v2_content_list(content)
            else:
                summary, found_equations = summarize_v1_content_list(content)

            content_result = {"name": name, "path": str(relpath), **summary}
            content_results.append({"summary": content_result, "equations": found_equations})

        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        markdown_summary = summarize_markdown(markdown)
        figures = read_json(figures_path) if figures_path.exists() else []

        if not dry_run:
            paper["markdown_sha256"] = sha256_text(
                markdown_path.read_text(encoding="utf-8", errors="replace")
            )
            paper.pop("preprocessed_reduction", None)
            paper["preprocessed_analysis"] = {
                "content_policy": "full_document",
                "equations_relpath": "equations.json",
                "report_relpath": "../preprocessed_analysis_report.json",
            }
            write_json(paper_path, paper)
            write_json(paper_dir / "equations.json", primary_equations(content_results))

        primary = primary_summary(content_results)
        equations = primary_equations(content_results)
        paper_block_types = Counter(primary.get("block_type_counts", {}))
        paper_image_types = Counter(primary.get("image_like_type_counts", {}))
        paper_image_extensions = Counter(primary.get("image_extensions", {}))
        paper_equation_types = Counter(primary.get("equation_type_counts", {}))
        paper_summary = {
            "status": "processed",
            "paper_id": paper_id,
            "paper_dir": str(paper_dir),
            "original_num_pages": paper.get("num_pages_original", paper.get("num_pages")),
            "current_num_pages": paper.get("num_pages"),
            "figure_count": len(figures) if isinstance(figures, list) else None,
            "markdown": markdown_summary,
            "block_type_counts": dict(sorted(paper_block_types.items())),
            "image_like_type_counts": dict(sorted(paper_image_types.items())),
            "image_extensions": dict(sorted(paper_image_extensions.items())),
            "equation_type_counts": dict(sorted(paper_equation_types.items())),
            "equation_count": len(equations),
        }
        return paper_summary
    except Exception as exc:
        return {"status": "failed", "paper_dir": str(paper_dir), "error": str(exc)}


def is_compacted_paper_dir(paper_dir: Path) -> bool:
    required = (
        paper_dir / "paper.json",
        paper_dir / "figures.json",
        paper_dir / "markdown.md",
        paper_dir / "images",
    )
    if not all(path.exists() for path in required):
        return False
    if len(list(paper_dir.glob("*.pdf"))) != 1:
        return False
    allowed = {"paper.json", "figures.json", "markdown.md", "images", "_SUCCESS"} | {
        path.name for path in paper_dir.glob("*.pdf")
    }
    return all(child.name in allowed for child in paper_dir.iterdir())


def should_report_progress(index: int, total: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return index == 1 or index == total or index % progress_every == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root directory containing one subdirectory per paper.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/preprocessed_analysis_report.json"),
        help="Report JSON path. The report is printed even when this is set.",
    )
    parser.add_argument("--split-index", type=Path, default=None, help="Canonical train/test PDF split index.")
    parser.add_argument("--split", choices=["train", "test", "train+test", "all"], default="train+test")
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of paper directories to process concurrently.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed paper.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N completed papers; 0 disables.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_v2_content_list(content: Any) -> bool:
    return (
        isinstance(content, list)
        and (not content or isinstance(content[0], list))
    )


def summarize_v1_content_list(
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, equations = summarize_blocks(blocks, source_format="v1")
    summary.update({"format": "v1", "blocks": len(blocks)})
    return summary, equations


def summarize_v2_content_list(
    pages: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat_blocks: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        for block in page:
            copied = dict(block)
            copied.setdefault("page_idx", index)
            flat_blocks.append(copied)
    summary, equations = summarize_blocks(flat_blocks, source_format="v2")
    summary.update(
        {
            "format": "v2",
            "pages": len(pages),
            "blocks": len(flat_blocks),
        }
    )
    return summary, equations


def summarize_blocks(blocks: list[dict[str, Any]], source_format: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    block_types: Counter[str] = Counter()
    image_types: Counter[str] = Counter()
    image_extensions: Counter[str] = Counter()
    equation_types: Counter[str] = Counter()
    equations: list[dict[str, Any]] = []

    for block_index, block in enumerate(blocks):
        block_type = str(block.get("type") or "unknown")
        block_types[block_type] += 1
        if block_type in IMAGE_LIKE_TYPES:
            image_types[block_type] += 1
            image_path = image_path_from_block(block)
            if image_path:
                suffix = Path(image_path).suffix.lower() or "<no_extension>"
                image_extensions[suffix] += 1
        for equation in equations_from_block(block):
            equation_types[equation["type"]] += 1
            equations.append(
                {
                    "source_format": source_format,
                    "block_index": block_index,
                    "page_idx": block.get("page_idx"),
                    **equation,
                }
            )

    return (
        {
            "block_type_counts": dict(sorted(block_types.items())),
            "image_like_type_counts": dict(sorted(image_types.items())),
            "image_extensions": dict(sorted(image_extensions.items())),
            "equation_type_counts": dict(sorted(equation_types.items())),
        },
        equations,
    )


def summarize_markdown(markdown: str) -> dict[str, Any]:
    return {
        "trimmed": False,
        "lines": len(markdown.splitlines()),
        "characters": len(markdown),
    }


def equations_from_block(block: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    block_type = str(block.get("type") or "")
    if block_type in EQUATION_TYPES:
        content = extract_text(block.get("content")) or str(block.get("text") or "")
        if content.strip():
            found.append({"type": block_type, "content": content.strip()})

    for content_type, content in walk_content_items(block.get("content")):
        if content_type in EQUATION_TYPES:
            text = extract_text(content)
            if text.strip():
                found.append({"type": content_type, "content": text.strip()})
    return found


def walk_content_items(value: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        content_type = value.get("type")
        if isinstance(content_type, str):
            found.append((content_type, value.get("content", value)))
        for nested in value.values():
            found.extend(walk_content_items(nested))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_content_items(item))
    return found


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "latex", "html"):
            if key in value:
                text = extract_text(value[key])
                if text:
                    return text
        return " ".join(filter(None, (extract_text(item) for item in value.values())))
    if isinstance(value, list):
        return " ".join(filter(None, (extract_text(item) for item in value)))
    return ""


def image_path_from_block(block: dict[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("img_path", "image_path", "path"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("image_source", "table_source", "chart_source"):
            nested = content.get(key)
            if isinstance(nested, dict):
                value = image_path_from_block(nested)
                if value:
                    return value
    return None


def primary_summary(content_results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in content_results:
        summary = result["summary"]
        if summary.get("format") == "v2":
            return summary
    if content_results:
        return content_results[0]["summary"]
    return {}


def primary_equations(content_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in content_results:
        if result["summary"].get("format") == "v2":
            return result["equations"]
    if content_results:
        return content_results[0]["equations"]
    return []


if __name__ == "__main__":
    main()
