from __future__ import annotations

import json
from pathlib import Path

from extraction.mineru_extraction import build_pdf_inputs
from extraction.mineru_extraction import build_extract_parser
from extraction.mineru_extraction import discover_pdfs
from extraction.mineru_extraction import resolve_split_index


def write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")


def test_extract_parser_accepts_explicit_split_union_and_domains() -> None:
    args = build_extract_parser().parse_args(["--split", "train+test", "--domains", "ACL", "Biology"])

    assert args.split == "train+test"
    assert args.domains == ["ACL", "Biology"]
    assert args.all_domain_pdfs is False


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


def test_discover_pdfs_can_select_train_and_test_for_one_domain(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "train.pdf")
    write_pdf(input_dir / "ACL" / "test.pdf")
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(
        json.dumps(
            {
                "train": ["ACL/train.pdf", "Physics/train.pdf"],
                "test": ["ACL/test.pdf"],
            }
        ),
        encoding="utf-8",
    )

    pdfs = discover_pdfs(
        input_dir,
        split_index=split_index,
        split="train+test",
        domains=["ACL"],
    )

    assert [(pdf.relative_path.as_posix(), pdf.split) for pdf in pdfs] == [
        ("ACL/train.pdf", "train"),
        ("ACL/test.pdf", "test"),
    ]


def test_discover_pdfs_can_select_all_pdfs_in_domain_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "Biology" / "one.pdf")
    write_pdf(input_dir / "Biology" / "nested" / "two.pdf")
    write_pdf(input_dir / "Medicine" / "three.pdf")

    pdfs = discover_pdfs(input_dir, domains=["Biology"])

    assert [pdf.relative_path.as_posix() for pdf in pdfs] == [
        "Biology/nested/two.pdf",
        "Biology/one.pdf",
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


def test_resolve_split_index_requires_canonical_index_for_named_split() -> None:
    assert resolve_split_index(Path("data/pdf_datasets"), None, "train") == Path(
        "data/splits/pdf_dataset_split.json"
    )


def test_resolve_split_index_requires_canonical_index_for_train_test_union() -> None:
    assert resolve_split_index(Path("data/pdf_datasets"), None, "train+test") == Path(
        "data/splits/pdf_dataset_split.json"
    )


def test_resolve_split_index_can_be_bypassed_for_complete_domain_folders() -> None:
    assert (
        resolve_split_index(
            Path("data/pdf_datasets"),
            Path("custom-split.json"),
            "train+test",
            all_domain_pdfs=True,
        )
        is None
    )


def test_resolve_split_index_leaves_custom_all_split_discovery_unsplit(tmp_path: Path) -> None:
    assert resolve_split_index(tmp_path / "custom_pdfs", None, "all") is None
