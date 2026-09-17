from __future__ import annotations

import pytest

from scripts.build_pdf_dataset_split import DEFAULT_DATASETS
from scripts.build_pdf_dataset_split import build_pdf_dataset_split


def make_pdf_dataset(root, dataset: str, count: int) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True)
    for index in range(count):
        (dataset_dir / f"{dataset.lower()}_{index:03d}.pdf").write_bytes(b"%PDF\n")


def test_build_pdf_dataset_split_samples_requested_count_per_dataset(tmp_path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    for dataset in DEFAULT_DATASETS:
        make_pdf_dataset(input_dir, dataset, 12)

    split_index = build_pdf_dataset_split(
        input_dir=input_dir,
        train_size_per_dataset=10,
        test_size_per_dataset=1,
        seed=7,
    )

    assert len(split_index["train"]) == 50
    assert len(split_index["test"]) == 5
    assert set(split_index["train"]).isdisjoint(split_index["test"])
    for dataset in DEFAULT_DATASETS:
        assert sum(path.startswith(f"{dataset}/") for path in split_index["train"]) == 10
        assert sum(path.startswith(f"{dataset}/") for path in split_index["test"]) == 1
        assert split_index["metadata"]["counts_by_dataset"][dataset] == {
            "available": 12,
            "train": 10,
            "test": 1,
        }


def test_build_pdf_dataset_split_is_reproducible_for_seed(tmp_path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    for dataset in ("ACL", "Physics"):
        make_pdf_dataset(input_dir, dataset, 8)

    first = build_pdf_dataset_split(
        input_dir=input_dir,
        datasets=("ACL", "Physics"),
        train_size_per_dataset=3,
        test_size_per_dataset=2,
        seed=99,
    )
    second = build_pdf_dataset_split(
        input_dir=input_dir,
        datasets=("ACL", "Physics"),
        train_size_per_dataset=3,
        test_size_per_dataset=2,
        seed=99,
    )

    assert first["train"] == second["train"]
    assert first["test"] == second["test"]


def test_build_pdf_dataset_split_requires_enough_pdfs(tmp_path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    make_pdf_dataset(input_dir, "ACL", 2)

    with pytest.raises(ValueError, match="ACL has only 2 PDFs"):
        build_pdf_dataset_split(
            input_dir=input_dir,
            datasets=("ACL",),
            train_size_per_dataset=2,
            test_size_per_dataset=1,
        )
