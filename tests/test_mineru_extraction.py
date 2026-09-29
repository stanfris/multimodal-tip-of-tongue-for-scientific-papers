from __future__ import annotations

import json
from pathlib import Path

from extraction.mineru_extraction import build_pdf_inputs
from extraction.mineru_extraction import build_extraction_resume_report
from extraction.mineru_extraction import build_extract_parser
from extraction.mineru_extraction import discover_pdfs
from extraction.mineru_extraction import filter_pdfs_for_retry
from extraction.mineru_extraction import paper_output_dir
from extraction.mineru_extraction import resolve_extraction_output_dir
from extraction.mineru_extraction import resolve_split_index
from extraction.mineru_extraction import write_extraction_incomplete_manifest


def write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")


def test_extract_parser_accepts_explicit_split_union_and_domains() -> None:
    args = build_extract_parser().parse_args(["--split", "train+test", "--domains", "ACL", "Biology"])

    assert args.split == "train+test"
    assert args.domains == ["ACL", "Biology"]
    assert args.all_domain_pdfs is False


def test_extract_parser_accepts_retry_incomplete_only() -> None:
    args = build_extract_parser().parse_args(["--retry-incomplete-only"])

    assert args.retry_incomplete_only is True


def test_single_domain_extraction_keeps_shared_output_root() -> None:
    assert resolve_extraction_output_dir(Path("data/processed"), ["Engineering"]) == Path("data/processed")


def test_paper_output_dir_always_includes_source_subset(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    write_pdf(input_dir / "Engineering" / "paper.pdf")

    pdfs = discover_pdfs(input_dir)

    assert [paper_output_dir(tmp_path / "processed", pdf) for pdf in pdfs] == [
        tmp_path / "processed/papers/ACL/ACL_paper",
        tmp_path / "processed/papers/Engineering/Engineering_paper",
    ]


def test_discover_pdfs_uses_relative_path_ids_for_multi_dataset_roots(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    write_pdf(input_dir / "Physics" / "paper.pdf")

    pdfs = discover_pdfs(input_dir)

    assert [pdf.paper_id for pdf in pdfs] == ["ACL_paper", "Physics_paper"]
    assert [pdf.source_dataset for pdf in pdfs] == ["ACL", "Physics"]
    assert [pdf.relative_path.as_posix() for pdf in pdfs] == ["ACL/paper.pdf", "Physics/paper.pdf"]


def test_direct_subset_input_still_uses_subset_output_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets" / "ACL"
    write_pdf(input_dir / "paper.pdf")

    pdf = discover_pdfs(input_dir)[0]

    assert pdf.source_dataset == "ACL"
    assert paper_output_dir(tmp_path / "processed", pdf) == tmp_path / "processed/papers/ACL/paper"


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


def test_resolve_split_index_leaves_default_all_selection_unsplit() -> None:
    assert resolve_split_index(Path("data/pdf_datasets"), None, "all") is None


def test_extraction_incomplete_manifest_tracks_failed_and_pending_documents(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "done.pdf")
    write_pdf(input_dir / "ACL" / "failed.pdf")
    write_pdf(input_dir / "ACL" / "pending.pdf")
    pdfs = discover_pdfs(input_dir)
    output_dir = tmp_path / "processed"

    done = next(pdf for pdf in pdfs if pdf.relative_path.name == "done.pdf")
    done_dir = paper_output_dir(output_dir, done)
    done_dir.mkdir(parents=True)
    (done_dir / "markdown.md").write_text("# Done\n", encoding="utf-8")
    (done_dir / "_SUCCESS").touch()
    (done_dir / "paper.json").write_text(
        json.dumps({"paper_id": done.paper_id, "markdown_relpath": "markdown.md", "figures": []}),
        encoding="utf-8",
    )
    failed = next(pdf for pdf in pdfs if pdf.relative_path.name == "failed.pdf")
    (output_dir / "failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "failures.jsonl").write_text(
        json.dumps(
            {
                "paper_id": failed.paper_id,
                "source_pdf_relpath": failed.relative_path.as_posix(),
                "error_type": "timeout",
                "message": "timed out",
                "attempts": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = write_extraction_incomplete_manifest(output_dir, pdfs)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["incomplete_count"] == 2
    assert {row["paper_id"]: row["status"] for row in payload["documents"]} == {
        "ACL_failed": "failed",
        "ACL_pending": "pending",
    }


def test_retry_incomplete_only_filters_selected_pdfs_from_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "retry.pdf")
    write_pdf(input_dir / "ACL" / "skip.pdf")
    pdfs = discover_pdfs(input_dir)
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "incomplete_documents.json").write_text(
        json.dumps({"documents": [{"source_pdf_relpath": "ACL/retry.pdf"}]}),
        encoding="utf-8",
    )

    retry = filter_pdfs_for_retry(pdfs, output_dir)

    assert [pdf.relative_path.as_posix() for pdf in retry] == ["ACL/retry.pdf"]


def test_resume_report_preserves_complete_and_queues_incomplete(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "done.pdf")
    write_pdf(input_dir / "ACL" / "retry.pdf")
    pdfs = discover_pdfs(input_dir)
    output_dir = tmp_path / "processed"

    done = next(pdf for pdf in pdfs if pdf.relative_path.name == "done.pdf")
    done_dir = paper_output_dir(output_dir, done)
    done_dir.mkdir(parents=True)
    (done_dir / "markdown.md").write_text("# Done\n", encoding="utf-8")
    (done_dir / "_SUCCESS").touch()
    (done_dir / "paper.json").write_text(
        json.dumps({"markdown_relpath": "markdown.md", "figures": []}),
        encoding="utf-8",
    )
    retry = next(pdf for pdf in pdfs if pdf.relative_path.name == "retry.pdf")
    (output_dir / "failures.jsonl").write_text(
        json.dumps({"paper_id": retry.paper_id, "error_type": "timeout"}) + "\n",
        encoding="utf-8",
    )

    report = build_extraction_resume_report(output_dir, pdfs)

    assert report == {
        "mode": "continue",
        "selected": 2,
        "preserved_complete": 1,
        "queued_incomplete": 1,
        "prior_failures": 1,
        "overwrite_completed": False,
        "completed_output_policy": "preserve",
        "incomplete_output_policy": "replace_after_successful_retry",
    }
