#!/usr/bin/env python3
"""Compact preprocessed paper directories to markdown, source PDF, and visual crops."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


VISUAL_BLOCK_TYPES = {"chart", "image", "table"}


def main() -> None:
    args = parse_args()
    preprocessed_dir = args.preprocessed_dir
    pdf_dir = args.pdf_dir
    if not preprocessed_dir.exists():
        raise SystemExit(f"Preprocessed directory does not exist: {preprocessed_dir}")
    if not pdf_dir.exists():
        raise SystemExit(f"ACL subset PDF directory does not exist: {pdf_dir}")

    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    image_count = 0
    image_type_counts: Counter[str] = Counter()

    paper_dirs = sorted(path for path in preprocessed_dir.iterdir() if path.is_dir())
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(compact_paper_dir, paper_dir, pdf_dir, args.dry_run)
            for paper_dir in paper_dirs
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "processed":
                processed += 1
                image_count += result["image_count"]
                image_type_counts.update(result["image_type_counts"])
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failures.append({"paper_dir": result["paper_dir"], "error": result["error"]})
                if args.fail_fast:
                    raise RuntimeError(f"{result['paper_dir']}: {result['error']}")

            if args.progress_every and index % args.progress_every == 0:
                print(
                    json.dumps(
                        {"stage": "compact", "seen": index, "processed": processed, "skipped": skipped, "failures": len(failures)},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "paper_count": processed,
                "skipped_count": skipped,
                "failure_count": len(failures),
                "failures": failures,
                "image_count": image_count,
                "image_type_counts": dict(sorted(image_type_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if failures:
        raise SystemExit(1)


def compact_paper_dir(paper_dir: Path, pdf_dir: Path, dry_run: bool) -> dict[str, Any]:
    try:
        if is_compacted_paper_dir(paper_dir):
            return {"status": "skipped", "paper_dir": str(paper_dir), "reason": "already_compacted"}

        paper_path = paper_dir / "paper.json"
        markdown_path = paper_dir / "markdown.md"
        if not paper_path.exists() or not markdown_path.exists():
            return {"status": "skipped", "paper_dir": str(paper_dir), "reason": "missing_inputs"}

        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        paper_id = str(paper.get("paper_id") or paper_dir.name)
        v2_path = find_v2_content_list(paper_dir, paper)
        if not v2_path:
            raise RuntimeError(f"No v2 content list found for {paper_id}")

        visual_sources = collect_visual_sources(v2_path)
        staged_dir = paper_dir / ".compact_images_tmp"
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        if not dry_run:
            staged_dir.mkdir()

        copied_images: list[dict[str, Any]] = []
        for source in visual_sources:
            source_path = resolve_source_image(v2_path.parent, source["source"])
            if not source_path.exists():
                raise RuntimeError(f"Missing source image for {paper_id}: {source_path}")
            destination_name = destination_image_name(source_path)
            destination_path = staged_dir / destination_name
            if not dry_run:
                shutil.copy2(source_path, destination_path)
            copied_images.append(
                {
                    "type": source["type"],
                    "page_idx": source["page_idx"],
                    "source": source["source"],
                    "destination": f"images/{destination_name}",
                }
            )

        source_pdf = pdf_dir / f"{paper_id}.pdf"
        if not source_pdf.exists():
            raise RuntimeError(f"Missing ACL subset PDF for {paper_id}: {source_pdf}")

        if not dry_run:
            final_images_dir = paper_dir / "images"
            if final_images_dir.exists():
                shutil.rmtree(final_images_dir)
            staged_dir.rename(final_images_dir)
            shutil.copy2(source_pdf, paper_dir / source_pdf.name)
            remove_unwanted_files(paper_dir, keep={markdown_path.name, source_pdf.name, "images"})

        return {
            "status": "processed",
            "paper_id": paper_id,
            "paper_dir": str(paper_dir),
            "image_count": len(copied_images),
            "image_type_counts": count_by_type(copied_images),
        }
    except Exception as exc:
        staged_dir = paper_dir / ".compact_images_tmp"
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        return {"status": "failed", "paper_dir": str(paper_dir), "error": str(exc)}


def is_compacted_paper_dir(paper_dir: Path) -> bool:
    if not (paper_dir / "markdown.md").exists() or not (paper_dir / "images").is_dir():
        return False
    if len(list(paper_dir.glob("*.pdf"))) != 1:
        return False
    allowed = {"markdown.md", "images"} | {path.name for path in paper_dir.glob("*.pdf")}
    return all(child.name in allowed for child in paper_dir.iterdir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root directory containing one subdirectory per paper.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/acl_subset/pdfs"),
        help="Directory containing ACL subset PDFs named <paper_id>.pdf.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned compaction without writing.")
    parser.add_argument("--workers", type=int, default=1, help="Number of paper directories to process concurrently.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed paper.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N completed papers; 0 disables.")
    return parser.parse_args()


def find_v2_content_list(paper_dir: Path, paper: dict[str, Any]) -> Path | None:
    for relpath in (paper.get("structured_outputs") or {}).values():
        path = paper_dir / str(relpath)
        if path.name.endswith("_content_list_v2.json") and path.exists():
            return path
    matches = sorted(paper_dir.glob("mineru/*/hybrid_auto/*_content_list_v2.json"))
    return matches[0] if matches else None


def collect_visual_sources(v2_path: Path) -> list[dict[str, Any]]:
    pages = json.loads(v2_path.read_text(encoding="utf-8"))
    visual_sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for page_idx, page in enumerate(pages):
        for block in page:
            block_type = block.get("type")
            if block_type not in VISUAL_BLOCK_TYPES:
                continue
            image_source = image_source_from_block(block)
            if not image_source:
                continue
            key = (str(block_type), image_source)
            if key in seen:
                continue
            seen.add(key)
            visual_sources.append({"type": str(block_type), "page_idx": page_idx, "source": image_source})
    return visual_sources


def image_source_from_block(block: dict[str, Any]) -> str | None:
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
        nested = content.get("image_source")
        if isinstance(nested, dict):
            value = nested.get("path")
            if isinstance(value, str) and value:
                return value
    return None


def resolve_source_image(base_dir: Path, source: str) -> Path:
    path = Path(source)
    if path.is_absolute():
        return path
    return base_dir / path


def destination_image_name(source_path: Path) -> str:
    return source_path.name


def remove_unwanted_files(paper_dir: Path, keep: set[str]) -> None:
    for child in paper_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def count_by_type(images: list[dict[str, Any]]) -> dict[str, int]:
    counts = {block_type: 0 for block_type in sorted(VISUAL_BLOCK_TYPES)}
    for image in images:
        counts[str(image["type"])] += 1
    return {key: value for key, value in counts.items() if value}


if __name__ == "__main__":
    main()
