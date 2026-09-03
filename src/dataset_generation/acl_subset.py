"""Build a curated ACL Anthology subset from official XML metadata."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


LOGGER = logging.getLogger(__name__)

ACL_ANTHOLOGY_XML_BASE_URL = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml"
ACL_ANTHOLOGY_PAPER_BASE_URL = "https://aclanthology.org"
DEFAULT_OUTPUT_DIR = Path("data") / "acl_subset"
DEFAULT_XML_CACHE_DIR = DEFAULT_OUTPUT_DIR / "xml_cache"
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_MAX_WORKERS = 16

TARGET_VOLUMES: dict[str, str] = {
    "acl_2023": "2023.acl-long",
    "acl_2024": "2024.acl-long",
    "acl_2025": "2025.acl-long",
    "acl_2026": "2026.acl-long",
    "emnlp_2023": "2023.emnlp-main",
    "emnlp_2024": "2024.emnlp-main",
    "emnlp_2025": "2025.emnlp-main",
    "naacl_2025": "2025.naacl-long",
}


@dataclass(frozen=True)
class BuildResult:
    papers_path: Path
    metadata_path: Path
    paper_count: int
    counts_by_volume: dict[str, int]
    pdf_stats: dict[str, Any] | None = None


def build_acl_subset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    xml_cache_dir: str | Path | None = None,
    refresh_metadata: bool = False,
    download_pdfs: bool = False,
    pdf_dir: str | Path | None = None,
    overwrite_pdfs: bool = False,
    sleep_seconds: float = 0.0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> BuildResult:
    """Write papers.jsonl for the configured Anthology volumes and optionally PDFs."""
    output_path = Path(output_dir)
    cache_path = Path(xml_cache_dir) if xml_cache_dir is not None else output_path / "xml_cache"
    papers = list(iter_target_papers(cache_path, refresh_metadata=refresh_metadata))
    papers.sort(key=lambda paper: paper["anthology_id"])

    output_path.mkdir(parents=True, exist_ok=True)
    papers_path = output_path / "papers.jsonl"
    write_jsonl(papers, papers_path)

    counts_by_volume = dict(sorted(Counter(paper["volume_id"] for paper in papers).items()))
    metadata = {
        "source": "ACL Anthology official XML metadata",
        "source_repository": "https://github.com/acl-org/acl-anthology",
        "source_xml_base_url": ACL_ANTHOLOGY_XML_BASE_URL,
        "target_volumes": TARGET_VOLUMES,
        "paper_count": len(papers),
        "counts_by_volume": counts_by_volume,
        "excluded": [
            "frontmatter/proceedings records ending in .0",
            "Findings volumes",
            "short-paper volumes",
            "workshop volumes",
            "demo/system demonstration volumes",
            "industry-track volumes",
            "student research workshop volumes",
            "tutorial volumes",
            "shared-task and colocated-event volumes",
        ],
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    pdf_stats = None
    if download_pdfs:
        pdf_stats = download_acl_pdfs(
            papers,
            output_dir=Path(pdf_dir) if pdf_dir is not None else output_path / "pdfs",
            overwrite=overwrite_pdfs,
            sleep_seconds=sleep_seconds,
            max_workers=max_workers,
        )

    return BuildResult(
        papers_path=papers_path,
        metadata_path=metadata_path,
        paper_count=len(papers),
        counts_by_volume=counts_by_volume,
        pdf_stats=pdf_stats,
    )


def iter_target_papers(
    xml_cache_dir: str | Path = DEFAULT_XML_CACHE_DIR,
    *,
    refresh_metadata: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield actual paper records from the exact configured Anthology volumes."""
    cache_path = Path(xml_cache_dir)
    collections = sorted({volume_id.rsplit("-", 1)[0] for volume_id in TARGET_VOLUMES.values()})
    collection_roots = {
        collection_id: load_collection_xml(
            collection_id,
            cache_path,
            refresh_metadata=refresh_metadata,
        )
        for collection_id in collections
    }

    for volume_id in TARGET_VOLUMES.values():
        collection_id, local_volume_id = split_volume_id(volume_id)
        root = collection_roots[collection_id]
        volume = find_volume(root, collection_id, local_volume_id)
        volume_meta = parse_volume_meta(volume)
        for paper in volume.findall("paper"):
            anthology_id = child_text(paper, "url")
            if not anthology_id:
                anthology_id = f"{volume_id}.{paper.get('id')}"
            if anthology_id.endswith(".0"):
                continue
            yield paper_record(paper, anthology_id, volume_id, volume_meta)


def load_collection_xml(
    collection_id: str,
    xml_cache_dir: str | Path,
    *,
    refresh_metadata: bool = False,
) -> ElementTree.Element:
    """Load one official ACL Anthology collection XML file from cache or GitHub."""
    cache_path = Path(xml_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    xml_path = cache_path / f"{collection_id}.xml"
    if refresh_metadata or not xml_path.exists():
        url = f"{ACL_ANTHOLOGY_XML_BASE_URL}/{collection_id}.xml"
        LOGGER.info("Fetching %s", url)
        xml_path.write_bytes(fetch_bytes(url))
    return ElementTree.fromstring(xml_path.read_bytes())


def fetch_bytes(url: str, *, timeout: int = 60, max_retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "acl-subset-builder/1.0"})
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def split_volume_id(volume_id: str) -> tuple[str, str]:
    if "-" not in volume_id:
        raise ValueError(f"Volume ID must contain '-': {volume_id}")
    return volume_id.rsplit("-", 1)


def find_volume(root: ElementTree.Element, collection_id: str, local_volume_id: str) -> ElementTree.Element:
    if root.get("id") != collection_id:
        raise ValueError(f"Expected collection {collection_id}, found {root.get('id')}")
    for volume in root.findall("volume"):
        if volume.get("id") == local_volume_id:
            return volume
    raise ValueError(f"Volume {collection_id}-{local_volume_id} not found in {collection_id}.xml")


def parse_volume_meta(volume: ElementTree.Element) -> dict[str, Any]:
    meta = volume.find("meta")
    if meta is None:
        return {"year": None, "month": None, "venue": None}
    return {
        "year": parse_int(child_text(meta, "year")),
        "month": null_if_empty(child_text(meta, "month")),
        "venue": normalize_venue(child_text(meta, "venue")),
    }


def paper_record(
    paper: ElementTree.Element,
    anthology_id: str,
    volume_id: str,
    volume_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "anthology_id": anthology_id,
        "title": null_if_empty(child_text(paper, "title")),
        "authors": [person_record(author) for author in paper.findall("author")],
        "year": volume_meta["year"],
        "month": volume_meta["month"],
        "venue": volume_meta["venue"],
        "volume_id": volume_id,
        "pages": null_if_empty(child_text(paper, "pages")),
        "abstract": null_if_empty(child_text(paper, "abstract")),
        "doi": null_if_empty(child_text(paper, "doi")),
        "url": f"{ACL_ANTHOLOGY_PAPER_BASE_URL}/{anthology_id}/",
        "pdf_url": f"{ACL_ANTHOLOGY_PAPER_BASE_URL}/{anthology_id}.pdf",
    }


def person_record(person: ElementTree.Element) -> dict[str, str | None]:
    return {
        "first": null_if_empty(child_text(person, "first")),
        "last": null_if_empty(child_text(person, "last")),
    }


def child_text(parent: ElementTree.Element, tag: str) -> str | None:
    child = parent.find(tag)
    if child is None:
        return None
    return normalize_text("".join(child.itertext()))


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split())


def null_if_empty(value: str | None) -> str | None:
    value = normalize_text(value)
    return value if value else None


def parse_int(value: str | None) -> int | None:
    value = null_if_empty(value)
    if value is None:
        return None
    return int(value)


def normalize_venue(value: str | None) -> str | None:
    value = null_if_empty(value)
    return value.upper() if value else None


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def download_acl_pdfs(
    papers: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_PDF_DIR,
    overwrite: bool = False,
    sleep_seconds: float = 0.0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paper_rows = list(papers)
    total = len(paper_rows)
    downloaded = skipped = failed = 0
    failures: list[dict[str, str]] = []
    started_at = time.monotonic()
    worker_count = max(1, max_workers)
    print(f"Downloading {total} PDFs with {worker_count} workers into {output_path}", flush=True)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                download_pdf_task,
                paper,
                output_path,
                overwrite=overwrite,
                sleep_seconds=sleep_seconds,
            )
            for paper in paper_rows
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "downloaded":
                downloaded += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(
                    {
                        "anthology_id": result["anthology_id"],
                        "pdf_url": result["pdf_url"],
                        "error": result["error"],
                    }
                )
                LOGGER.warning("Failed to download %s: %s", result["anthology_id"], result["error"])
            print_progress(
                completed=completed,
                total=total,
                downloaded=downloaded,
                skipped=skipped,
                failed=failed,
                started_at=started_at,
            )
    print()

    if failures:
        (output_path / "download_failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return {"total": total, "downloaded": downloaded, "skipped": skipped, "failed": failed}


def download_pdf_task(
    paper: dict[str, Any],
    output_dir: Path,
    *,
    overwrite: bool,
    sleep_seconds: float,
) -> dict[str, str]:
    anthology_id = str(paper["anthology_id"])
    pdf_url = str(paper["pdf_url"])
    destination = output_dir / f"{anthology_id}.pdf"
    if destination.exists() and not overwrite and has_pdf_header(destination):
        return {"status": "skipped", "anthology_id": anthology_id, "pdf_url": pdf_url, "error": ""}
    try:
        download_one_pdf(pdf_url, destination)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return {"status": "downloaded", "anthology_id": anthology_id, "pdf_url": pdf_url, "error": ""}
    except Exception as exc:  # noqa: BLE001 - keep batch downloads moving.
        return {"status": "failed", "anthology_id": anthology_id, "pdf_url": pdf_url, "error": str(exc)}


def print_progress(
    *,
    completed: int,
    total: int,
    downloaded: int,
    skipped: int,
    failed: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = completed / elapsed
    percent = (completed / total * 100) if total else 100.0
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0
    message = (
        f"\r{completed}/{total} ({percent:5.1f}%) "
        f"downloaded={downloaded} skipped={skipped} failed={failed} "
        f"rate={rate:0.1f}/s eta={format_duration(eta_seconds)}"
    )
    print(message, end="", flush=True)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def download_one_pdf(url: str, destination: Path) -> None:
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.write_bytes(fetch_bytes(url, timeout=120))
    if not has_pdf_header(part_path):
        part_path.unlink(missing_ok=True)
        raise ValueError("downloaded file is not a PDF")
    part_path.replace(destination)


def has_pdf_header(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a curated ACL Anthology subset from official XML metadata.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write data/acl_subset/papers.jsonl only. This is the default.",
    )
    mode.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Write metadata and download all corresponding PDFs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xml-cache-dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--refresh-metadata", action="store_true", help="Refetch XML even when cached files exist.")
    parser.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay after each PDF download.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Concurrent PDF downloads for --download-pdfs.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    result = build_acl_subset(
        output_dir=args.output_dir,
        xml_cache_dir=args.xml_cache_dir,
        refresh_metadata=args.refresh_metadata,
        download_pdfs=args.download_pdfs,
        pdf_dir=args.pdf_dir,
        overwrite_pdfs=args.overwrite_pdfs,
        sleep_seconds=args.sleep_seconds,
        max_workers=args.max_workers,
    )
    print(f"Wrote {result.paper_count} papers to {result.papers_path}")
    print(f"Wrote build metadata to {result.metadata_path}")
    for volume_id, count in result.counts_by_volume.items():
        print(f"{volume_id}: {count}")
    if result.pdf_stats is not None:
        print("PDF downloads:")
        for key in ("total", "downloaded", "skipped", "failed"):
            print(f"{key}: {result.pdf_stats[key]}")


if __name__ == "__main__":
    main()
