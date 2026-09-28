#!/usr/bin/env python
"""Deprecated: write a post-extraction document split index for legacy datasets."""

from __future__ import annotations

import argparse
import json
import sys
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/preprocessed"), help="Preprocessed papers directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits") / f"{DEFAULT_SPLIT_NAME}.json",
        help="Output JSON split index.",
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


def main() -> int:
    args = build_parser().parse_args()
    print(
        "Deprecated: use pdf_corpus.build_pdf_dataset_split before extraction; "
        "newly extracted paper.json files retain that canonical split.",
        file=sys.stderr,
    )
    if args.test_fraction is None:
        split_index = build_split_index(
            read_preprocessed_papers(args.dataset),
            train_size=args.train_size,
            test_size=args.test_size,
            test_fraction=None,
            seed=args.seed,
            stratify_field=args.stratify_field,
        )
    else:
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
