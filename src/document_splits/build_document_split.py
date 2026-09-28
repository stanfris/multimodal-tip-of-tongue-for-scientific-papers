#!/usr/bin/env python
"""Write a fixed-seed stratified train/test document split index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_splits.document_splits import (
    DEFAULT_SPLIT_NAME,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_TEST_FRACTION,
    build_split_index,
    write_split_index,
)
from preprocessing.preprocessed import read_preprocessed_papers

DEFAULT_DATASET = Path("data/preprocessed")
DEFAULT_OUTPUT = Path("data/splits") / f"{DEFAULT_SPLIT_NAME}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        help=(
            "Dataset storage root. Defaults --dataset to ROOT/processed and "
            "--output to ROOT/data/splits/document_split.json."
        ),
    )
    parser.add_argument("--dataset", type=Path, help=f"Preprocessed papers directory (default: {DEFAULT_DATASET}).")
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Output JSON split index (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help=f"Use fraction mode instead of fixed sizes. Previous default was {DEFAULT_TEST_FRACTION}.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--stratify-field",
        default=None,
        help="Optional paper.json field to stratify by. Defaults to venue/year when available.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root_dir = args.root_dir.expanduser().resolve() if args.root_dir is not None else None
    dataset = args.dataset or (root_dir / "processed" if root_dir is not None else DEFAULT_DATASET)
    output = args.output or (
        root_dir / "data/splits" / f"{DEFAULT_SPLIT_NAME}.json"
        if root_dir is not None
        else DEFAULT_OUTPUT
    )
    return dataset, output


def main() -> int:
    args = build_parser().parse_args()
    dataset, output = resolve_paths(args)
    if args.test_fraction is None:
        split_index = build_split_index(
            read_preprocessed_papers(dataset),
            train_size=args.train_size,
            test_size=args.test_size,
            test_fraction=None,
            seed=args.seed,
            stratify_field=args.stratify_field,
        )
    else:
        split_index = build_split_index(
            read_preprocessed_papers(dataset),
            test_fraction=args.test_fraction,
            seed=args.seed,
            stratify_field=args.stratify_field,
        )
    output_path = write_split_index(split_index, output)
    print(json.dumps({"output": str(output_path), **split_index.metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
