#!/usr/bin/env python3
"""Split arXiv PDFs into Physics and Engineering folders using the manifest domain."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


DOMAIN_DIRS = {
    "arxiv_physics": "physics",
    "arxiv_engineering": "engineering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or move PDFs from data/arxiv_open_reuse/pdfs into physics/ and engineering/ "
            "subdirectories using selection_domain from the arXiv manifest."
        )
    )
    parser.add_argument("--arxiv-root", type=Path, default=Path("data/arxiv_open_reuse"))
    parser.add_argument("--pdf-dir", type=Path, help="Source PDF directory. Defaults to ARXIV_ROOT/pdfs.")
    parser.add_argument("--output-dir", type=Path, help="Destination root. Defaults to ARXIV_ROOT/pdfs_by_domain.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest JSONL. Defaults to eligible_records.jsonl under ARXIV_ROOT.",
    )
    parser.add_argument("--move", action="store_true", help="Move PDFs instead of copying them.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned actions without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination PDFs.")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def filename_for_row(row: dict[str, object]) -> str | None:
    filename = row.get("pdf_filename")
    if isinstance(filename, str) and filename:
        return filename
    arxiv_id = row.get("pdf_arxiv_id") or row.get("arxiv_id")
    if isinstance(arxiv_id, str) and arxiv_id:
        return f"{arxiv_id.replace('/', '_')}.pdf"
    return None


def arxiv_subset_from_manifest_row(row: dict[str, object]) -> str:
    selection_domain = str(row.get("selection_domain") or "").lower()
    if selection_domain.startswith("eess."):
        return "arxiv_engineering"
    if selection_domain.startswith("physics."):
        return "arxiv_physics"

    broad_domains = row.get("broad_domains")
    domains = [str(value).lower() for value in broad_domains] if isinstance(broad_domains, list) else []
    source_query = str(row.get("source_api_query") or "").lower()
    primary_category = str(row.get("primary_category") or "").lower()
    categories = row.get("categories") or row.get("subject_categories") or row.get("category_matches")
    category_values = [str(value).lower() for value in categories] if isinstance(categories, list) else []
    has_eess = any(category.startswith("eess.") for category in category_values)
    has_physics = any(category.startswith("physics.") for category in category_values)

    if "engineering" in domains and "physics" not in domains:
        return "arxiv_engineering"
    if "physics" in domains and "engineering" not in domains:
        return "arxiv_physics"
    if source_query.startswith("eess"):
        return "arxiv_engineering"
    if source_query.startswith("physics"):
        return "arxiv_physics"
    if has_eess and not has_physics:
        return "arxiv_engineering"
    if has_physics and not has_eess:
        return "arxiv_physics"
    if primary_category.startswith("eess"):
        return "arxiv_engineering"
    return "arxiv_physics"


def split_pdfs(
    *,
    manifest_path: Path,
    pdf_dir: Path,
    output_dir: Path,
    move: bool,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in iter_jsonl(manifest_path):
        filename = filename_for_row(row)
        if not filename:
            counts["missing_filename"] += 1
            continue
        source = pdf_dir / filename
        subset = arxiv_subset_from_manifest_row(row)
        domain_dir = DOMAIN_DIRS.get(subset)
        if domain_dir is None:
            counts["unknown_domain"] += 1
            continue
        destination = output_dir / domain_dir / filename
        if not source.exists():
            counts[f"missing_source_{domain_dir}"] += 1
            continue
        if destination.exists() and not overwrite:
            counts[f"existing_{domain_dir}"] += 1
            continue
        counts[f"planned_{domain_dir}"] += 1
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if move:
            if overwrite and destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
        counts[f"written_{domain_dir}"] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    arxiv_root = args.arxiv_root
    manifest_path = args.manifest or arxiv_root / "eligible_records.jsonl"
    pdf_dir = args.pdf_dir or arxiv_root / "pdfs"
    output_dir = args.output_dir or arxiv_root / "pdfs_by_domain"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    counts = split_pdfs(
        manifest_path=manifest_path,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        move=args.move,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    verb = "Would write" if args.dry_run else "Wrote"
    action = "move" if args.move else "copy"
    print(f"{verb} arXiv PDF {action} split under {output_dir}")
    for key, count in counts.items():
        print(f"{key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
