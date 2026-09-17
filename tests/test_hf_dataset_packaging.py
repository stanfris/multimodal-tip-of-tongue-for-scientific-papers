from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path

import pandas as pd

from dataset_generation.hf_dataset_packaging import SourceSpec
from dataset_generation.hf_dataset_packaging import build_parser
from dataset_generation.hf_dataset_packaging import prepare_dataset
from dataset_generation.hf_dataset_packaging import run
from dataset_generation.hf_dataset_packaging import shard_size_bytes
from dataset_generation.hf_dataset_packaging import validate_dataset


def write_pdf(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + body + b"\n%%EOF\n")


def test_prepare_dataset_writes_webdataset_shards_and_metadata(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="dataset_generation.hf_dataset_packaging")
    input_root = tmp_path / "pdf_datasets"
    write_pdf(input_root / "ACL" / "2023.acl-long.1.pdf", b"acl")
    write_pdf(input_root / "Physics" / "2609.17126v1.pdf", b"physics")

    data_root = input_root.parent
    acl_manifest = data_root / "acl_subset" / "papers.jsonl"
    acl_manifest.parent.mkdir(parents=True)
    acl_manifest.write_text(
        json.dumps(
            {
                "anthology_id": "2023.acl-long.1",
                "title": "ACL title",
                "year": 2023,
                "doi": "10.18653/v1/2023.acl-long.1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    arxiv_manifest = data_root / "arxiv_open_reuse" / "eligible_records.jsonl"
    arxiv_manifest.parent.mkdir(parents=True)
    arxiv_manifest.write_text(
        json.dumps(
            {
                "arxiv_id": "2609.17126v1",
                "pdf_filename": "2609.17126v1.pdf",
                "title": "arXiv title",
                "created": "2026-09-15",
                "normalized_license_url": "https://creativecommons.org/licenses/by/4.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "huggingface_dataset"
    sources = [SourceSpec("ACL", input_root / "ACL"), SourceSpec("Physics", input_root / "Physics")]
    prepare_dataset(
        sources=sources,
        output_dir=output_dir,
        shard_size_bytes=shard_size_bytes(0.001),
    )

    metadata = pd.read_parquet(output_dir / "metadata.parquet")
    assert list(metadata.columns)[:3] == ["document_id", "source", "original_filename"]
    assert set(metadata["source"]) == {"ACL", "Physics"}
    assert (output_dir / "duplicates.parquet").exists()
    assert (output_dir / "README.md").exists()

    row = metadata.loc[metadata["source"] == "ACL"].iloc[0]
    with tarfile.open(output_dir / row["shard"], "r") as tar:
        names = set(tar.getnames())
        assert row["member_path"] in names
        assert row["member_path"].replace(".pdf", ".json") in names
        sidecar = json.loads(tar.extractfile(row["member_path"].replace(".pdf", ".json")).read())
        assert sidecar["title"] == "ACL title"
        assert sidecar["doi"] == "10.18653/v1/2023.acl-long.1"

    result = validate_dataset(output_dir=output_dir, sources=sources)
    assert result.errors == []
    assert result.packaged_pdf_count == 2
    assert result.unique_content_count == 2
    assert "Validation: checking metadata and SHA-256 of 2 source PDFs" in caplog.text
    assert "Validation complete: 2 PDFs" in caplog.text


def test_prepare_dataset_records_duplicates_and_can_extract_by_document_id(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    pdf_body = b"same content"
    write_pdf(input_root / "Biology" / "PMC1.pdf", pdf_body)
    write_pdf(input_root / "Medicine" / "PMC1.pdf", pdf_body)
    sources = [SourceSpec("Biology", input_root / "Biology"), SourceSpec("Medicine", input_root / "Medicine")]
    output_dir = tmp_path / "out"

    prepare_dataset(sources=sources, output_dir=output_dir, shard_size_bytes=shard_size_bytes(0.001))

    duplicates = pd.read_parquet(output_dir / "duplicates.parquet")
    assert len(duplicates) == 1
    metadata = pd.read_parquet(output_dir / "metadata.parquet")
    document_id = metadata.iloc[0]["document_id"]
    located = metadata.loc[metadata["document_id"] == document_id].iloc[0]
    with tarfile.open(output_dir / located["shard"], "r") as tar:
        assert tar.extractfile(located["member_path"]).read().startswith(b"%PDF")


def test_prepare_dataset_reuses_completed_shards(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    write_pdf(input_root / "ACL" / "paper.pdf", b"first")
    sources = [SourceSpec("ACL", input_root / "ACL")]
    output_dir = tmp_path / "out"

    prepare_dataset(sources=sources, output_dir=output_dir, shard_size_bytes=shard_size_bytes(0.001))
    shard = next((output_dir / "data" / "ACL").glob("*.tar"))
    first_mtime = shard.stat().st_mtime_ns

    prepare_dataset(sources=sources, output_dir=output_dir, shard_size_bytes=shard_size_bytes(0.001))
    assert shard.stat().st_mtime_ns == first_mtime


def test_validation_reports_invalid_pdf_without_packaging_it(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    bad_pdf = input_root / "ACL" / "bad.pdf"
    bad_pdf.parent.mkdir(parents=True)
    bad_pdf.write_bytes(b"not a pdf")

    output_dir = tmp_path / "out"
    sources = [SourceSpec("ACL", input_root / "ACL")]
    prepare_dataset(sources=sources, output_dir=output_dir, shard_size_bytes=shard_size_bytes(0.001))

    report = json.loads((output_dir / "preparation_report.json").read_text(encoding="utf-8"))
    assert report["packaged_pdf_count"] == 0
    assert report["invalid_or_unreadable"][0]["error"] == "invalid_pdf_header"


def test_upload_only_can_skip_local_validation(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in ("README.md", "metadata.parquet", "duplicates.parquet", "preparation_report.json"):
        (output_dir / name).touch()
    (output_dir / "data").mkdir()
    calls = []
    monkeypatch.setattr(
        "dataset_generation.hf_dataset_packaging.upload_dataset",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "dataset_generation.hf_dataset_packaging.validate_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("validation should be skipped")),
    )

    args = build_parser().parse_args(["--upload-only", "--skip-validation", "--output-dir", str(output_dir)])
    result = run(args)

    assert result["validation_skipped"] is True
    assert len(calls) == 1
    assert calls[0]["output_dir"] == output_dir
