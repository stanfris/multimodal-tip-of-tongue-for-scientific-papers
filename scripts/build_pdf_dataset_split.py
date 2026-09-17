#!/usr/bin/env python3
"""Write a random per-dataset train/test split for PDFs under data/pdf_datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("ACL", "Biology", "Engineering", "Medicine", "Physics")
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE_PER_DATASET = 1000
DEFAULT_TEST_SIZE_PER_DATASET = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/pdf_datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/pdf_dataset_split.json"),
        help="Output JSON split index.",
    )
    parser.add_argument("--train-size-per-dataset", type=int, default=DEFAULT_TRAIN_SIZE_PER_DATASET)
    parser.add_argument("--test-size-per-dataset", type=int, default=DEFAULT_TEST_SIZE_PER_DATASET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subset folders to sample from.",
    )
    return parser.parse_args()


def build_pdf_dataset_split(
    *,
    input_dir: Path,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    train_size_per_dataset: int = DEFAULT_TRAIN_SIZE_PER_DATASET,
    test_size_per_dataset: int = DEFAULT_TEST_SIZE_PER_DATASET,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if train_size_per_dataset < 1:
        raise ValueError("train_size_per_dataset must be at least 1")
    if test_size_per_dataset < 1:
        raise ValueError("test_size_per_dataset must be at least 1")

    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []
    counts_by_dataset: dict[str, dict[str, int]] = {}
    dataset_names = list(datasets)

    for dataset in dataset_names:
        dataset_dir = input_dir / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

        pdfs = sorted(path.relative_to(dataset_dir).as_posix() for path in dataset_dir.rglob("*.pdf"))
        requested = train_size_per_dataset + test_size_per_dataset
        if len(pdfs) < requested:
            raise ValueError(
                f"{dataset} has only {len(pdfs)} PDFs, but {requested} are required "
                f"({train_size_per_dataset} train + {test_size_per_dataset} test)."
            )

        sampled = rng.sample(pdfs, requested)
        dataset_train = sorted(f"{dataset}/{relative_path}" for relative_path in sampled[:train_size_per_dataset])
        dataset_test = sorted(f"{dataset}/{relative_path}" for relative_path in sampled[train_size_per_dataset:])
        train.extend(dataset_train)
        test.extend(dataset_test)
        counts_by_dataset[dataset] = {
            "available": len(pdfs),
            "train": len(dataset_train),
            "test": len(dataset_test),
        }

    train.sort()
    test.sort()
    return {
        "train": train,
        "test": test,
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "datasets": dataset_names,
            "seed": seed,
            "train_size_per_dataset": train_size_per_dataset,
            "test_size_per_dataset": test_size_per_dataset,
            "train_count": len(train),
            "test_count": len(test),
            "counts_by_dataset": counts_by_dataset,
        },
    }


def write_pdf_dataset_split(split_index: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split_index, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    split_index = build_pdf_dataset_split(
        input_dir=args.input_dir,
        datasets=args.datasets,
        train_size_per_dataset=args.train_size_per_dataset,
        test_size_per_dataset=args.test_size_per_dataset,
        seed=args.seed,
    )
    output_path = write_pdf_dataset_split(split_index, args.output)
    print(json.dumps({"output": str(output_path), **split_index["metadata"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
