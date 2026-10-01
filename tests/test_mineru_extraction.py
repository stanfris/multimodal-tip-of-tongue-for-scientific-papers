from __future__ import annotations

import json
import asyncio
from pathlib import Path

import httpx
import pytest
from PIL import Image

import extraction.mineru_extraction as mineru_extraction
from extraction.mineru_extraction import build_pdf_inputs
from extraction.mineru_extraction import build_benchmark_parser
from extraction.mineru_extraction import build_extraction_resume_report
from extraction.mineru_extraction import build_extract_parser
from extraction.mineru_extraction import discover_pdfs
from extraction.mineru_extraction import filter_pdfs_for_retry
from extraction.mineru_extraction import paper_output_dir
from extraction.mineru_extraction import resolve_extraction_output_dir
from extraction.mineru_extraction import resolve_split_index
from extraction.mineru_extraction import options_from_args, submit_task
from extraction.mineru_extraction import recommend_benchmark_result
from extraction.mineru_extraction import write_extraction_incomplete_manifest
from extraction.mineru_extraction import MinerUOptions, PDFInput, normalize_figures


def test_normalize_figures_reads_nested_mineru_image_source(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    base_dir = work_dir / "mineru" / "raw"
    image_path = base_dir / "images" / "figure.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (12, 8)).save(image_path)
    pdf_input = PDFInput(
        path=tmp_path / "paper.pdf",
        relative_path=Path("ACL/paper.pdf"),
        paper_id="ACL_paper",
        source_dataset="ACL",
        split="train",
    )

    figures = normalize_figures(
        [[{"type": "image", "content": {"image_source": {"path": "images/figure.png"}}}]],
        base_dir,
        pdf_input.paper_id,
        pdf_input,
        work_dir,
        tmp_path / "final",
    )

    assert len(figures) == 1
    assert figures[0]["image_relpath"] == "mineru/raw/images/figure.png"
    assert figures[0]["image_width"] == 12
    assert figures[0]["image_height"] == 8


def write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7\n")


def test_extract_parser_accepts_explicit_split_union_and_domains() -> None:
    args = build_extract_parser().parse_args(["--split", "train+test", "--domains", "ACL", "Biology"])

    assert args.split == "train+test"
    assert args.domains == ["ACL", "Biology"]
    assert args.all_domain_pdfs is False


def test_extract_parser_accepts_other_split() -> None:
    assert build_extract_parser().parse_args(["--split", "other"]).split == "other"


def test_extraction_defaults_to_no_last_page() -> None:
    args = build_extract_parser().parse_args([])
    assert options_from_args(args).end_page_id is None


def test_benchmark_covers_high_concurrency_and_recommends_only_stable_runs() -> None:
    assert build_benchmark_parser().parse_args([]).max_in_flight_values == [4, 8, 16, 24, 32]
    results = [
        {"pdfs": 20, "completed": 20, "failed": 0, "max_in_flight": 16,
         "papers_per_second": 1.0, "pages_per_second": 10.0},
        {"pdfs": 20, "completed": 19, "failed": 1, "max_in_flight": 32,
         "papers_per_second": 2.0, "pages_per_second": 20.0},
    ]
    assert recommend_benchmark_result(results)["max_in_flight"] == 16
    assert recommend_benchmark_result(results[1:]) is None


def test_extraction_run_config_records_both_concurrency_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "pdfs"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    output_dir = tmp_path / "processed"
    monkeypatch.setenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "32")

    async def fake_extract_many(pdfs, output, options, *, state_dir):
        assert options.max_in_flight == 24
        assert state_dir == output_dir
        return mineru_extraction.ExtractionStats(total=len(pdfs))

    monkeypatch.setattr(mineru_extraction, "extract_many", fake_extract_many)
    args = build_extract_parser().parse_args([
        "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--max-in-flight", "24",
    ])
    mineru_extraction.run_extract(args)
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["server_concurrency"] == 32
    assert run_config["max_in_flight"] == 24


def test_submission_omits_page_cutoff_by_default(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    write_pdf(pdf)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"task_id": "task-1"})

    async def submit(options: MinerUOptions) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://mineru") as client:
            assert await submit_task(client, pdf, options) == "task-1"

    asyncio.run(submit(MinerUOptions()))
    assert b'name="end_page_id"' not in requests[-1].content
    asyncio.run(submit(MinerUOptions(end_page_id=3)))
    assert b'name="end_page_id"' in requests[-1].content


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


def test_discover_pdfs_selects_only_documents_outside_train_and_test(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    for relative_path in (
        "ACL/train.pdf", "ACL/test.pdf", "ACL/other.pdf",
        "ACL/nested/extra.pdf", "Biology/other.pdf",
    ):
        write_pdf(input_dir / relative_path)
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(json.dumps({
        "train": ["ACL/train.pdf"],
        "test": ["ACL/test.pdf"],
    }), encoding="utf-8")

    pdfs = discover_pdfs(input_dir, split_index=split_index, split="other", domains=["ACL"])

    assert [(pdf.relative_path.as_posix(), pdf.split) for pdf in pdfs] == [
        ("ACL/nested/extra.pdf", "other"),
        ("ACL/other.pdf", "other"),
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


def test_resolve_split_index_uses_custom_input_root_for_indexed_runs(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "custom" / "pdf_datasets"
    assert resolve_split_index(pdf_dir, None, "train") == (
        tmp_path / "custom" / "splits" / "pdf_dataset_split.json"
    )


def test_other_split_requires_index_and_cannot_bypass_it() -> None:
    input_dir = Path("data/pdf_datasets")
    assert resolve_split_index(input_dir, None, "other") == Path("data/splits/pdf_dataset_split.json")
    with pytest.raises(ValueError, match="requires a train/test split index"):
        discover_pdfs(input_dir, split="other")
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_split_index(input_dir, None, "other", all_domain_pdfs=True)


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


def test_full_domain_run_does_not_read_missing_index_and_uses_selected_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "corpus" / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    write_pdf(input_dir / "Biology" / "other.pdf")
    output_dir = tmp_path / "corpus" / "processed"

    async def fake_extract_many(pdfs, output, options, *, state_dir):
        assert output == output_dir
        assert state_dir == output_dir / "_runs" / "ACL"
        assert [pdf.relative_path.as_posix() for pdf in pdfs] == ["ACL/paper.pdf"]
        return mineru_extraction.ExtractionStats(total=len(pdfs))

    monkeypatch.setattr(mineru_extraction, "extract_many", fake_extract_many)
    args = build_extract_parser().parse_args([
        "--input-dir", str(input_dir), "--output-dir", str(output_dir),
        "--split", "train+test", "--split-index", str(tmp_path / "missing.json"),
        "--all-domain-pdfs", "--domains", "ACL",
    ])
    mineru_extraction.run_extract(args)
    saved = json.loads((output_dir / "_runs" / "ACL" / "run_config.json").read_text())
    assert saved["split_index"] is None
    assert saved["all_domain_pdfs"] is True
    assert saved["output_dir"] == str(output_dir)
    assert saved["state_dir"] == str(output_dir / "_runs" / "ACL")
    assert not (tmp_path / "corpus" / "splits").exists()


def test_missing_indexed_input_reports_launcher_derived_path_before_writing(tmp_path: Path) -> None:
    input_dir = tmp_path / "corpus" / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "paper.pdf")
    output_dir = tmp_path / "corpus" / "processed"
    args = build_extract_parser().parse_args([
        "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--split", "train",
    ])
    with pytest.raises(FileNotFoundError, match="corpus/splits/pdf_dataset_split.json"):
        mineru_extraction.run_extract(args)
    assert not output_dir.exists()


def test_full_domain_run_rejects_wrong_pdf_root_before_writing(tmp_path: Path) -> None:
    input_dir = tmp_path / "corpus" / "pdf_datasets"
    write_pdf(input_dir / "Biology" / "paper.pdf")
    output_dir = tmp_path / "corpus" / "processed"
    args = build_extract_parser().parse_args([
        "--input-dir", str(input_dir), "--output-dir", str(output_dir),
        "--split", "train+test", "--all-domain-pdfs", "--domains", "ACL",
    ])
    with pytest.raises(FileNotFoundError, match="No PDFs found for domains"):
        mineru_extraction.run_extract(args)
    assert not output_dir.exists()


def test_domain_run_reports_do_not_replace_each_other(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "pdf_datasets"
    for domain in ("ACL", "Biology"):
        write_pdf(input_dir / domain / "paper.pdf")
    output_dir = tmp_path / "processed"

    async def fake_extract_many(pdfs, output, options, *, state_dir):
        return mineru_extraction.ExtractionStats(total=len(pdfs))

    monkeypatch.setattr(mineru_extraction, "extract_many", fake_extract_many)
    for domain in ("ACL", "Biology"):
        args = build_extract_parser().parse_args([
            "--input-dir", str(input_dir), "--output-dir", str(output_dir),
            "--split", "train+test", "--all-domain-pdfs", "--domains", domain,
        ])
        mineru_extraction.run_extract(args)

    for domain in ("ACL", "Biology"):
        state_dir = output_dir / "_runs" / domain
        saved = json.loads((state_dir / "run_config.json").read_text())
        assert saved["domains"] == [domain]
        assert (state_dir / "last_run_summary.json").exists()
        assert (state_dir / "incomplete_documents.json").exists()
    assert not (output_dir / "run_config.json").exists()


def test_domain_state_preserves_completed_paper_and_reads_legacy_retry_state(tmp_path: Path) -> None:
    input_dir = tmp_path / "pdf_datasets"
    write_pdf(input_dir / "ACL" / "done.pdf")
    write_pdf(input_dir / "ACL" / "retry.pdf")
    pdfs = discover_pdfs(input_dir, domains=["ACL"])
    output_dir = tmp_path / "processed"
    state_dir = output_dir / "_runs" / "ACL"
    done = next(pdf for pdf in pdfs if pdf.relative_path.name == "done.pdf")
    done_dir = paper_output_dir(output_dir, done)
    done_dir.mkdir(parents=True)
    (done_dir / "markdown.md").write_text("# Original\n", encoding="utf-8")
    (done_dir / "paper.json").write_text(
        json.dumps({"markdown_relpath": "markdown.md", "figures": []}), encoding="utf-8",
    )
    (done_dir / "_SUCCESS").touch()
    (output_dir / "incomplete_documents.json").write_text(
        json.dumps({"documents": [{"source_pdf_relpath": "ACL/retry.pdf"}]}), encoding="utf-8",
    )

    selected = filter_pdfs_for_retry(pdfs, output_dir, state_dir=state_dir)
    assert [pdf.relative_path.as_posix() for pdf in selected] == ["ACL/retry.pdf"]
    stats = asyncio.run(mineru_extraction.extract_many(
        [done], output_dir, MinerUOptions(), state_dir=state_dir,
    ))
    assert stats.already_complete == 1
    assert stats.submitted == 0
    assert (done_dir / "markdown.md").read_text() == "# Original\n"
    assert (state_dir / "incomplete_documents.json").exists()


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
