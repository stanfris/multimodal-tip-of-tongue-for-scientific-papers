#!/usr/bin/env python
"""Write a fixed-seed stratified train/test document split index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_generation.document_splits import (
    DEFAULT_SPLIT_NAME,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_FRACTION,
    build_split_index,
    write_split_index,
)
from dataset_generation.preprocessed import read_preprocessed_papers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/preprocessed"), help="Preprocessed papers directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits") / f"{DEFAULT_SPLIT_NAME}.json",
        help="Output JSON split index.",
    )
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--stratify-field",
        default=None,
        help="Optional paper.json field to stratify by. Defaults to venue/year when available.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split_index = build_split_index(
        read_preprocessed_papers(args.dataset),
        test_fraction=args.test_fraction,
        seed=args.seed,
        stratify_field=args.stratify_field,
    )
    output_path = write_split_index(split_index, args.output)
    print(json.dumps({"output": str(output_path), **split_index.metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
