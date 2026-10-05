#!/usr/bin/env python3
"""Write a random per-dataset train/test PDF split under data/pdf_datasets.

The default five-group corpus contributes 2,200 training documents and 110
test documents from each of ACL, Biology, Engineering, Medicine, and Physics.
These are source-document counts; downstream query generation may produce one
or more queries from each selected document.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DEFAULT_DATASETS = ("ACL", "Biology", "Engineering", "Medicine", "Physics")
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE_PER_DATASET = 2200
DEFAULT_TEST_SIZE_PER_DATASET = 110
DEFAULT_INPUT_DIR = Path("data/pdf_datasets")
DEFAULT_OUTPUT = Path("data/splits/pdf_dataset_split.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        help=(
            "Dataset storage root. Defaults --input-dir to ROOT/pdf_datasets "
            "and --output to ROOT/splits/pdf_dataset_split.json."
        ),
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON split index.",
    )
    parser.add_argument(
        "--train-size-per-dataset",
        type=int,
        default=DEFAULT_TRAIN_SIZE_PER_DATASET,
        help="Training PDFs to sample from each dataset folder (default: 2200).",
    )
    parser.add_argument(
        "--test-size-per-dataset",
        type=int,
        default=DEFAULT_TEST_SIZE_PER_DATASET,
        help="Test PDFs to sample from each dataset folder (default: 110).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing split with different membership."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subset folders to sample from.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root_dir = args.root_dir.expanduser().resolve() if args.root_dir is not None else None
    input_dir = args.input_dir or (root_dir / "pdf_datasets" if root_dir is not None else DEFAULT_INPUT_DIR)
    output = args.output or (
        root_dir / "splits/pdf_dataset_split.json" if root_dir is not None else DEFAULT_OUTPUT
    )
    return input_dir, output


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


def write_pdf_dataset_split(split_index: dict[str, Any], output: Path, *, overwrite: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing split is invalid JSON: {output}; use --overwrite to replace it") from exc
        if (
            isinstance(existing, dict)
            and existing.get("train") == split_index["train"]
            and existing.get("test") == split_index["test"]
        ):
            if not isinstance(existing.get("metadata"), dict):
                raise ValueError(f"Existing split lacks metadata: {output}; use --overwrite to replace it")
            return output
        raise FileExistsError(f"Existing split has different membership: {output}; use --overwrite to replace it")
    with NamedTemporaryFile(
        "w", encoding="utf-8", prefix=f".{output.name}.", suffix=".tmp",
        dir=output.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(split_index, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> int:
    args = build_parser().parse_args()
    input_dir, output = resolve_paths(args)
    split_index = build_pdf_dataset_split(
        input_dir=input_dir,
        datasets=args.datasets,
        train_size_per_dataset=args.train_size_per_dataset,
        test_size_per_dataset=args.test_size_per_dataset,
        seed=args.seed,
    )
    output_path = write_pdf_dataset_split(split_index, output, overwrite=args.overwrite)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output_path), **saved["metadata"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
