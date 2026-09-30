#!/usr/bin/env python3
"""Compact preprocessed paper directories to metadata, markdown, source PDF, and visual crops."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preprocessing.parse_tracking import write_incomplete_documents
from preprocessing.preprocessed import select_preprocessed_paper_dirs


VISUAL_BLOCK_TYPES = {"chart", "image", "table"}


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    preprocessed_dir = args.preprocessed_dir
    pdf_dir = args.pdf_dir or (args.root_dir / "pdf_datasets" if args.root_dir is not None else Path("data/pdf_datasets"))
    if not preprocessed_dir.exists():
        raise SystemExit(f"Preprocessed directory does not exist: {preprocessed_dir}")
    if not pdf_dir.exists():
        raise SystemExit(f"PDF dataset directory does not exist: {pdf_dir}")

    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    image_count = 0
    image_type_counts: Counter[str] = Counter()

    print(
        json.dumps(
            {
                "stage": "compact",
                "status": "selecting",
                "source": str(args.split_index) if args.split_index else "recursive_scan",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selection = select_preprocessed_paper_dirs(
        preprocessed_dir,
        split_index=args.split_index,
        split=args.split,
    )
    paper_dirs = selection.paper_dirs
    incomplete_documents: list[dict[str, Any]] = list(selection.missing_documents)
    total_papers = len(paper_dirs)
    print(
        json.dumps(
            {
                "stage": "compact",
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
    if selection.expected_count and not total_papers:
        if not args.dry_run:
            write_incomplete_documents(
                preprocessed_dir,
                incomplete_documents,
                stage="compaction",
                source=selection.source,
            )
        raise SystemExit(
            "No completed extraction directories matched the split index beneath "
            f"{preprocessed_dir}. Expected _SUCCESS, paper.json, and markdown.md."
        )
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
                if result.get("reason") != "already_compacted":
                    incomplete_documents.append(incomplete_document_from_compaction_result(result, "skipped"))
            else:
                failures.append({"paper_dir": result["paper_dir"], "error": result["error"]})
                incomplete_documents.append(incomplete_document_from_compaction_result(result, "failed"))
                if not args.dry_run:
                    append_compaction_failure(preprocessed_dir, result)
                if args.fail_fast:
                    raise RuntimeError(f"{result['paper_dir']}: {result['error']}")

            if should_report_progress(index, total_papers, args.progress_every):
                print(
                    json.dumps(
                        {
                            "stage": "compact",
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

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "paper_count": processed,
                "skipped_count": skipped,
                "failure_count": len(failures),
                "incomplete_count": len(incomplete_documents),
                "failures": failures,
                "image_count": image_count,
                "image_type_counts": dict(sorted(image_type_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if not args.dry_run:
        write_incomplete_documents(
            preprocessed_dir,
            incomplete_documents,
            stage="compaction",
            source=selection.source,
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
            return {
                "status": "skipped",
                "paper_id": paper_dir.name,
                "paper_dir": str(paper_dir),
                "reason": "missing_inputs",
            }

        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        paper_id = str(paper.get("paper_id") or paper_dir.name)
        v2_path = find_v2_content_list(paper_dir, paper)
        if not v2_path:
            raise RuntimeError(f"No v2 content list found for {paper_id}")

        visual_sources = collect_visual_sources_from_figures(paper_dir, paper)
        if not visual_sources:
            visual_sources = collect_visual_sources(v2_path)
        staged_dir = paper_dir / ".compact_images_tmp"
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        if not dry_run:
            staged_dir.mkdir()

        copied_images: list[dict[str, Any]] = []
        used_destination_names: set[str] = set()
        for source in visual_sources:
            source_path = resolve_source_image(Path(source.get("base_dir", v2_path.parent)), source["source"])
            if not source_path.exists():
                raise RuntimeError(f"Missing source image for {paper_id}: {source_path}")
            destination_name = destination_image_name(source_path, used_destination_names)
            used_destination_names.add(destination_name)
            destination_path = staged_dir / destination_name
            if not dry_run:
                shutil.copy2(source_path, destination_path)
            copied_images.append(
                {
                    "type": source["type"],
                    "page_idx": source["page_idx"],
                    "figure_id": source.get("figure_id"),
                    "source": source["source"],
                    "destination": f"images/{destination_name}",
                }
            )

        source_pdf = resolve_source_pdf(paper, pdf_dir, paper_id)
        if not source_pdf.exists():
            raise RuntimeError(f"Missing source PDF for {paper_id}: {source_pdf}")

        if not dry_run:
            final_images_dir = paper_dir / "images"
            if final_images_dir.exists():
                shutil.rmtree(final_images_dir)
            staged_dir.rename(final_images_dir)
            shutil.copy2(source_pdf, paper_dir / source_pdf.name)
            updated_paper, updated_figures = update_compacted_metadata(paper, copied_images, source_pdf.name)
            write_json(paper_path, updated_paper)
            write_json(paper_dir / "figures.json", updated_figures)
            remove_unwanted_files(
                paper_dir,
                keep={markdown_path.name, source_pdf.name, "images", "paper.json", "figures.json", "_SUCCESS"},
            )

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
        return {"status": "failed", "paper_id": paper_dir.name, "paper_dir": str(paper_dir), "error": str(exc)}


def is_compacted_paper_dir(paper_dir: Path) -> bool:
    if (
        not (paper_dir / "markdown.md").exists()
        or not (paper_dir / "paper.json").exists()
        or not (paper_dir / "figures.json").exists()
        or not (paper_dir / "images").is_dir()
    ):
        return False
    if len(list(paper_dir.glob("*.pdf"))) != 1:
        return False
    allowed = {"markdown.md", "paper.json", "figures.json", "images", "_SUCCESS"} | {
        path.name for path in paper_dir.glob("*.pdf")
    }
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
        "--root-dir",
        type=Path,
        default=None,
        help="Corpus root containing pdf_datasets/. Used to derive --pdf-dir when --pdf-dir is omitted.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing source PDFs. Defaults to ROOT/pdf_datasets with --root-dir, otherwise data/pdf_datasets.",
    )
    parser.add_argument("--split-index", type=Path, default=None, help="Canonical train/test PDF split index.")
    parser.add_argument("--split", choices=["train", "test", "train+test", "all"], default="train+test")
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


def collect_visual_sources_from_figures(paper_dir: Path, paper: dict[str, Any]) -> list[dict[str, Any]]:
    visual_sources = []
    for figure in paper.get("figures") or []:
        source = image_source_from_figure(figure)
        if not source:
            continue
        source_path = Path(source)
        if source_path.is_absolute():
            relative_source = str(source_path)
        else:
            relative_source = str((paper_dir / source_path).relative_to(paper_dir))
        visual_sources.append(
            {
                "type": str(figure.get("type") or "image"),
                "page_idx": figure.get("page_idx"),
                "figure_id": figure.get("figure_id"),
                "source": relative_source,
                "base_dir": str(paper_dir),
            }
        )
    return visual_sources


def image_source_from_figure(figure: dict[str, Any]) -> str | None:
    for key in ("image_relpath", "image_path", "path"):
        value = figure.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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


def destination_image_name(source_path: Path, used_names: set[str]) -> str:
    candidate = source_path.name
    if candidate not in used_names:
        return candidate
    stem = source_path.stem or "image"
    suffix = source_path.suffix
    index = 2
    while True:
        candidate = f"{stem}-{index}{suffix}"
        if candidate not in used_names:
            return candidate
        index += 1


def resolve_source_pdf(paper: dict[str, Any], pdf_dir: Path, paper_id: str) -> Path:
    candidates = []
    source_pdf_value = paper.get("source_pdf")
    if source_pdf_value:
        candidates.append(Path(str(source_pdf_value)).expanduser())
    source_pdf_relpath = paper.get("source_pdf_relpath")
    if source_pdf_relpath:
        candidates.append(pdf_dir / str(source_pdf_relpath))
    candidates.append(pdf_dir / f"{paper_id}.pdf")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def update_compacted_metadata(
    paper: dict[str, Any],
    copied_images: list[dict[str, Any]],
    source_pdf_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_figure_id = {
        str(image["figure_id"]): image
        for image in copied_images
        if image.get("figure_id") is not None
    }
    by_source = {str(image["source"]): image for image in copied_images}

    updated_figures = []
    for figure in paper.get("figures") or []:
        updated = dict(figure)
        copied = None
        figure_id = updated.get("figure_id")
        if figure_id is not None:
            copied = by_figure_id.get(str(figure_id))
        copied = copied or by_source.get(str(image_source_from_figure(updated)))
        if copied is not None:
            updated["image_relpath"] = copied["destination"]
            updated["image_path"] = copied["destination"]
            updated["filename"] = Path(str(copied["destination"])).name
        updated_figures.append(updated)

    updated_paper = dict(paper)
    updated_paper["pdf_relpath"] = source_pdf_name
    updated_paper["source_pdf_copy_relpath"] = source_pdf_name
    updated_paper["figures"] = updated_figures
    updated_paper.pop("structured_outputs", None)
    updated_paper.pop("mineru_raw_relpath", None)
    return updated_paper, updated_figures


def incomplete_document_from_compaction_result(result: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "stage": "compaction",
        "status": status,
        "paper_id": result.get("paper_id"),
        "paper_dir": result.get("paper_dir"),
        "reason": result.get("reason"),
        "message": result.get("error"),
    }


def append_compaction_failure(output_dir: Path, result: dict[str, Any]) -> None:
    failure = {
        "stage": "compaction",
        "paper_id": result.get("paper_id"),
        "paper_dir": result.get("paper_dir"),
        "message": result.get("error"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "compaction_failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def should_report_progress(index: int, total: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return index == 1 or index == total or index % progress_every == 0


if __name__ == "__main__":
    main()
