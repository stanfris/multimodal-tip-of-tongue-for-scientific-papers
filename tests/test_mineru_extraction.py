from __future__ import annotations

import json
from pathlib import Path

from dataset_generation.mineru_extraction import build_pdf_inputs
from dataset_generation.mineru_extraction import discover_pdfs


def write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")


def test_discover_pdfs_uses_relative_path_ids_for_multi_dataset_roots(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    write_pdf(input_dir / "Physics" / "paper.pdf")

    pdfs = discover_pdfs(input_dir)

    assert [pdf.paper_id for pdf in pdfs] == ["ACL_paper", "Physics_paper"]
    assert [pdf.source_dataset for pdf in pdfs] == ["ACL", "Physics"]
    assert [pdf.relative_path.as_posix() for pdf in pdfs] == ["ACL/paper.pdf", "Physics/paper.pdf"]


def test_discover_pdfs_can_follow_train_test_split_index(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "train.pdf")
    write_pdf(input_dir / "Biology" / "test.pdf")
    split_index = tmp_path / "splits" / "pdf_dataset_split.json"
    split_index.parent.mkdir()
    split_index.write_text(
        json.dumps(
            {
                "train": ["ACL/train.pdf"],
                "test": ["Biology/test.pdf"],
            }
        ),
        encoding="utf-8",
    )

    train = discover_pdfs(input_dir, split_index=split_index, split="train")
    all_pdfs = discover_pdfs(input_dir, split_index=split_index, split="all")

    assert [(pdf.paper_id, pdf.split) for pdf in train] == [("ACL_train", "train")]
    assert [(pdf.paper_id, pdf.split) for pdf in all_pdfs] == [
        ("ACL_train", "train"),
        ("Biology_test", "test"),
    ]


def test_build_pdf_inputs_adds_hash_when_sanitized_ids_collide(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    first = input_dir / "A B.pdf"
    second = input_dir / "A/B.pdf"
    write_pdf(first)
    write_pdf(second)

    pdfs = build_pdf_inputs([first, second], input_dir)

    assert len({pdf.paper_id for pdf in pdfs}) == 2
    assert all(pdf.paper_id.startswith("A_B.") for pdf in pdfs)
