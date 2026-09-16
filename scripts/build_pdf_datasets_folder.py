#!/usr/bin/env python3
"""Build data/pdf_datasets with ACL, Physics, Engineering, Biology, and Medicine PDFs."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    name: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create data/pdf_datasets/{ACL,Physics,Engineering,Biology,Medicine} "
            "from the existing downloaded PDF corpora."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pdf_datasets"))
    parser.add_argument("--acl-dir", type=Path, help="Defaults to DATA_ROOT/acl_subset/pdfs.")
    parser.add_argument(
        "--physics-dir",
        type=Path,
        help="Defaults to DATA_ROOT/arxiv_open_reuse/pdfs_by_domain/physics.",
    )
    parser.add_argument(
        "--engineering-dir",
        type=Path,
        help="Defaults to DATA_ROOT/arxiv_open_reuse/pdfs_by_domain/engineering.",
    )
    parser.add_argument("--biology-dir", type=Path, help="Defaults to DATA_ROOT/pmc_oa_strict/pdfs/Biology.")
    parser.add_argument(
        "--medicine-dir",
        type=Path,
        help="Defaults to DATA_ROOT/pmc_oa_strict/pdfs/Medical_Clinical_Research.",
    )
    parser.add_argument("--move", action="store_true", help="Move PDFs instead of copying them.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination PDFs.")
    parser.add_argument("--clean", action="store_true", help="Remove DATA_ROOT/pdf_datasets before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned actions without writing files.")
    return parser.parse_args()


def default_sources(data_root: Path, args: argparse.Namespace) -> list[DatasetSource]:
    return [
        DatasetSource("ACL", args.acl_dir or data_root / "acl_subset" / "pdfs"),
        DatasetSource("Physics", args.physics_dir or data_root / "arxiv_open_reuse" / "pdfs_by_domain" / "physics"),
        DatasetSource(
            "Engineering",
            args.engineering_dir or data_root / "arxiv_open_reuse" / "pdfs_by_domain" / "engineering",
        ),
        DatasetSource("Biology", args.biology_dir or data_root / "pmc_oa_strict" / "pdfs" / "Biology"),
        DatasetSource(
            "Medicine",
            args.medicine_dir or data_root / "pmc_oa_strict" / "pdfs" / "Medical_Clinical_Research",
        ),
    ]


def materialize_pdf_datasets(
    *,
    sources: list[DatasetSource],
    output_dir: Path,
    move: bool,
    overwrite: bool,
    clean: bool,
    dry_run: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if clean and output_dir.exists():
        counts["clean_output_dir"] += 1
        if not dry_run:
            shutil.rmtree(output_dir)

    for source in sources:
        destination_root = output_dir / source.name
        if not dry_run:
            destination_root.mkdir(parents=True, exist_ok=True)
        else:
            counts[f"ensure_{source.name}"] += 1

        if not source.source.exists():
            counts[f"missing_source_{source.name}"] += 1
            continue

        pdfs = sorted(source.source.rglob("*.pdf"))
        if not pdfs:
            counts[f"empty_source_{source.name}"] += 1
            continue

        for pdf in pdfs:
            relative = pdf.relative_to(source.source)
            destination = destination_root / relative
            if destination.exists() and not overwrite:
                counts[f"existing_{source.name}"] += 1
                continue
            counts[f"planned_{source.name}"] += 1
            if dry_run:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if move:
                if overwrite and destination.exists():
                    destination.unlink()
                shutil.move(str(pdf), str(destination))
            else:
                shutil.copy2(pdf, destination)
            counts[f"written_{source.name}"] += 1

    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    sources = default_sources(args.data_root, args)
    counts = materialize_pdf_datasets(
        sources=sources,
        output_dir=args.output_dir,
        move=args.move,
        overwrite=args.overwrite,
        clean=args.clean,
        dry_run=args.dry_run,
    )
    verb = "Would build" if args.dry_run else "Built"
    action = "move" if args.move else "copy"
    print(f"{verb} {args.output_dir} using {action} mode")
    for source in sources:
        print(f"{source.name}: {source.source} -> {args.output_dir / source.name}")
    print("")
    for key, count in counts.items():
        print(f"{key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
