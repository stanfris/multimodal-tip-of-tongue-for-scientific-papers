from __future__ import annotations

import pytest

from pdf_corpus.build_pdf_dataset_split import DEFAULT_DATASETS
from pdf_corpus.build_pdf_dataset_split import DEFAULT_TEST_SIZE_PER_DATASET
from pdf_corpus.build_pdf_dataset_split import DEFAULT_TRAIN_SIZE_PER_DATASET
from pdf_corpus.build_pdf_dataset_split import build_parser, build_pdf_dataset_split, resolve_paths


def make_pdf_dataset(root, dataset: str, count: int) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True)
    for index in range(count):
        (dataset_dir / f"{dataset.lower()}_{index:03d}.pdf").write_bytes(b"%PDF\n")


def test_default_split_sizes_are_per_dataset_group() -> None:
    assert DEFAULT_TRAIN_SIZE_PER_DATASET == 2200
    assert DEFAULT_TEST_SIZE_PER_DATASET == 110
    assert len(DEFAULT_DATASETS) * DEFAULT_TRAIN_SIZE_PER_DATASET == 11_000
    assert len(DEFAULT_DATASETS) * DEFAULT_TEST_SIZE_PER_DATASET == 550


def test_root_dir_sets_scratch_layout_defaults(tmp_path) -> None:
    args = build_parser().parse_args(["--root-dir", str(tmp_path)])

    input_dir, output = resolve_paths(args)

    assert input_dir == tmp_path / "pdf_datasets"
    assert output == tmp_path / "data/splits/pdf_dataset_split.json"


def test_explicit_paths_override_root_dir(tmp_path) -> None:
    input_override = tmp_path / "custom-input"
    output_override = tmp_path / "custom-output.json"
    args = build_parser().parse_args(
        [
            "--root-dir",
            str(tmp_path / "root"),
            "--input-dir",
            str(input_override),
            "--output",
            str(output_override),
        ]
    )

    assert resolve_paths(args) == (input_override, output_override)


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
