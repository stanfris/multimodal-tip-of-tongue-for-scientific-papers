from __future__ import annotations

import json
import logging
import tarfile
import threading
from pathlib import Path

import pandas as pd
import pytest

from dataset_packaging.hf_dataset_packaging import SourceSpec
from dataset_packaging.hf_dataset_packaging import build_parser
from dataset_packaging.hf_dataset_packaging import hf_cli_command
from dataset_packaging.hf_dataset_packaging import prepare_dataset
from dataset_packaging.hf_dataset_packaging import prepare_additional_data
from dataset_packaging.hf_dataset_packaging import run
from dataset_packaging.hf_dataset_packaging import shard_size_bytes
from dataset_packaging.hf_dataset_packaging import validate_dataset
from dataset_packaging.hf_dataset_packaging import upload_additional_data
from dataset_packaging.hf_dataset_packaging import walk_additional_files
from dataset_packaging.restore_hf_dataset import restore_dataset


def write_pdf(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n" + body + b"\n%%EOF\n")


def test_prepare_dataset_writes_webdataset_shards_and_metadata(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="dataset_packaging.hf_dataset_packaging")
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


def test_restore_dataset_recreates_source_folders_and_nested_paths(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    acl_pdf = input_root / "ACL" / "2023" / "paper.pdf"
    physics_pdf = input_root / "Physics" / "2609.17126v1.pdf"
    write_pdf(acl_pdf, b"acl")
    write_pdf(physics_pdf, b"physics")
    packaged_dir = tmp_path / "huggingface_dataset"
    prepare_dataset(
        sources=[SourceSpec("ACL", input_root / "ACL"), SourceSpec("Physics", input_root / "Physics")],
        output_dir=packaged_dir,
        shard_size_bytes=shard_size_bytes(0.001),
    )

    restored_root = tmp_path / "restored_pdf_datasets"
    result = restore_dataset(dataset_dir=packaged_dir, target_dir=restored_root, workers=2)

    assert result["restored_pdf_count"] == 2
    assert result["workers"] == 2
    assert (restored_root / "ACL" / "2023" / "paper.pdf").read_bytes() == acl_pdf.read_bytes()
    assert (restored_root / "Physics" / "2609.17126v1.pdf").read_bytes() == physics_pdf.read_bytes()


def test_restore_dataset_refuses_to_overwrite_existing_pdfs(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    write_pdf(input_root / "ACL" / "paper.pdf", b"source")
    packaged_dir = tmp_path / "huggingface_dataset"
    prepare_dataset(
        sources=[SourceSpec("ACL", input_root / "ACL")],
        output_dir=packaged_dir,
        shard_size_bytes=shard_size_bytes(0.001),
    )
    restored_root = tmp_path / "restored_pdf_datasets"
    write_pdf(restored_root / "ACL" / "paper.pdf", b"existing")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        restore_dataset(dataset_dir=packaged_dir, target_dir=restored_root)


def test_restore_dataset_can_skip_checksum_verification(tmp_path) -> None:
    input_root = tmp_path / "pdf_datasets"
    source_pdf = input_root / "ACL" / "paper.pdf"
    write_pdf(source_pdf, b"source")
    packaged_dir = tmp_path / "huggingface_dataset"
    prepare_dataset(
        sources=[SourceSpec("ACL", input_root / "ACL")],
        output_dir=packaged_dir,
        shard_size_bytes=shard_size_bytes(0.001),
    )
    metadata_path = packaged_dir / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata.loc[:, "sha256"] = "0" * 64
    metadata.to_parquet(metadata_path, index=False)

    with pytest.raises(ValueError, match="Checksum or size mismatch"):
        restore_dataset(dataset_dir=packaged_dir, target_dir=tmp_path / "checked")

    result = restore_dataset(
        dataset_dir=packaged_dir,
        target_dir=tmp_path / "unchecked",
        verify_checksum=False,
    )
    assert result["checksum_verified"] is False
    assert (tmp_path / "unchecked" / "ACL" / "paper.pdf").read_bytes() == source_pdf.read_bytes()


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
        "dataset_packaging.hf_dataset_packaging.upload_dataset",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        "dataset_packaging.hf_dataset_packaging.validate_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("validation should be skipped")),
    )

    args = build_parser().parse_args(["--upload-only", "--skip-validation", "--output-dir", str(output_dir)])
    result = run(args)

    assert result["validation_skipped"] is True
    assert len(calls) == 1
    assert calls[0]["output_dir"] == output_dir


def test_hf_cli_command_accepts_uv_run_hf() -> None:
    assert hf_cli_command("uv run hf") == ["uv", "run", "hf"]
    assert build_parser().parse_args(["--additional-only"]).additional_shard_size_gb == 1.0


def test_additional_upload_preserves_pdf_package_and_uses_separate_remote_paths(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "dataset"
    (data_root / "preprocessed" / "ACL" / "paper-1").mkdir(parents=True)
    (data_root / "preprocessed" / "ACL" / "paper-1" / "markdown.md").write_text("paper")
    (data_root / "clues" / "ACL" / "paper-1" / "base").mkdir(parents=True)
    (data_root / "clues" / "ACL" / "paper-1" / "base" / "textual_clues.jsonl").write_text("{}\n")
    (data_root / "query_collections" / "ACL" / "set-1").mkdir(parents=True)
    (data_root / "query_collections" / "ACL" / "set-1" / "queries.jsonl").write_text("{}\n")
    (data_root / "splits").mkdir()
    (data_root / "splits" / "pdf_dataset_split.json").write_text("{}")
    output_dir = tmp_path / "package"
    (output_dir / "data" / "ACL").mkdir(parents=True)
    pdf_shard = output_dir / "data" / "ACL" / "shard-00000.tar"
    pdf_shard.write_bytes(b"existing PDF package")
    (output_dir / "metadata.parquet").write_bytes(b"existing metadata")

    uploads = list(prepare_additional_data(data_root=data_root, output_dir=output_dir, shard_size_bytes=1024))
    assert pdf_shard.read_bytes() == b"existing PDF package"
    assert (output_dir / "metadata.parquet").read_bytes() == b"existing metadata"
    assert {upload.remote_path for upload in uploads} == {
        "preprocessed/ACL/shard-00000.tar", "clues/ACL/shard-00000.tar", "query_collections", "splits"
    }
    assert not (tmp_path / "package_additional").exists()
    assert [upload.source for upload in uploads[-2:]] == [data_root / "query_collections", data_root / "splits"]

    commands = []
    temporary_shards = []
    def fake_upload(cmd, **kwargs):
        commands.append(cmd)
        if "hf-additional-" in cmd[3]:
            shard = next(Path(cmd[3]).glob("*.tar"))
            temporary_shards.append(shard)
            assert shard.exists()
            with tarfile.open(shard) as tar:
                assert tar.getnames() == (["paper-1/markdown.md"] if cmd[4].startswith("preprocessed") else ["paper-1/base/textual_clues.jsonl"])
    monkeypatch.setattr("dataset_packaging.hf_dataset_packaging.subprocess.run", fake_upload)
    upload_additional_data(uploads=uploads, repo_id="owner/dataset", hf_cli="hf")
    assert len(commands) == 4
    assert all(not shard.exists() for shard in temporary_shards)
    assert all(cmd[:3] == ["hf", "upload", "owner/dataset"] for cmd in commands)
    assert all(cmd[4] not in {"data", ".", "metadata.parquet", "README.md"} for cmd in commands)


def test_additional_scan_yields_before_finishing_a_domain(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    for name in ("preprocessed", "clues", "query_collections", "splits"):
        (data_root / name).mkdir(parents=True)
    domain = data_root / "preprocessed" / "ACL"
    domain.mkdir()
    (domain / "a.txt").write_text("a")
    (domain / "b.txt").write_text("b")
    (domain / "z-link").symlink_to(domain / "a.txt")
    uploads = prepare_additional_data(data_root=data_root, output_dir=tmp_path / "package", shard_size_bytes=1)
    assert next(iter(uploads)).members == (domain / "a.txt",)
    with pytest.raises(ValueError, match="Symlinks"):
        list(uploads)


def test_additional_file_limit_starts_small_shards_early(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    for name in ("preprocessed", "clues", "query_collections", "splits"):
        (data_root / name).mkdir(parents=True)
    domain = data_root / "preprocessed" / "ACL"
    domain.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (domain / name).write_text("x")
    (domain / "z-link").symlink_to(domain / "a.txt")
    uploads = prepare_additional_data(
        data_root=data_root,
        output_dir=tmp_path / "package",
        shard_size_bytes=1024,
        max_files_per_shard=2,
    )
    assert next(iter(uploads)).members == (domain / "a.txt", domain / "b.txt")
    with pytest.raises(ValueError, match="Symlinks"):
        list(uploads)


def test_streaming_walk_preserves_previous_sorted_path_order(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.txt").write_text("nested")
    (tmp_path / "a.txt").write_text("flat")
    assert [path.relative_to(tmp_path).as_posix() for path, _ in walk_additional_files(tmp_path)] == [
        "a.txt", "a/nested.txt"
    ]


def test_additional_uploads_run_concurrently(tmp_path, monkeypatch) -> None:
    from dataset_packaging.hf_dataset_packaging import AdditionalUpload

    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.txt"
    second = source / "second.txt"
    first.write_text("first")
    second.write_text("second")
    second_started = threading.Event()

    def fake_upload(cmd, **kwargs):
        if (Path(cmd[3]) / "shard-00000.tar").exists():
            assert second_started.wait(2), "second upload did not start while the first was running"
        else:
            second_started.set()

    monkeypatch.setattr("dataset_packaging.hf_dataset_packaging.subprocess.run", fake_upload)
    paths = upload_additional_data(
        uploads=[
            AdditionalUpload(source, "preprocessed/ACL/shard-00000.tar", (first,)),
            AdditionalUpload(source, "preprocessed/ACL/shard-00001.tar", (second,)),
        ],
        repo_id="owner/dataset",
        hf_cli="hf",
        workers=2,
    )
    assert paths == ["preprocessed/ACL/shard-00000.tar", "preprocessed/ACL/shard-00001.tar"]


def test_additional_upload_rejects_flat_paper_layout(tmp_path) -> None:
    data_root = tmp_path / "dataset"
    for name in ("preprocessed", "clues", "query_collections", "splits"):
        (data_root / name).mkdir(parents=True)
    (data_root / "preprocessed" / "paper-1.json").write_text("{}")
    with pytest.raises(ValueError, match="domain directories only"):
        list(prepare_additional_data(data_root=data_root, output_dir=tmp_path / "package", shard_size_bytes=1024))


def test_additional_upload_cleans_node_local_shard_after_failure(tmp_path, monkeypatch) -> None:
    from dataset_packaging.hf_dataset_packaging import AdditionalUpload

    source = tmp_path / "dataset" / "preprocessed" / "ACL"
    source.mkdir(parents=True)
    paper = source / "paper.md"
    paper.write_text("paper")
    local_temp = tmp_path / "node-local"
    local_temp.mkdir()
    monkeypatch.setenv("PBS_LOCALDIR", str(local_temp))
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    seen = []

    def fail_upload(cmd, **kwargs):
        shard = Path(cmd[3]) / "shard-00000.tar"
        seen.append(shard)
        assert shard.is_relative_to(local_temp)
        assert shard.exists()
        assert kwargs["env"]["HF_XET_CACHE"] == str(local_temp / "hf-xet-cache")
        raise RuntimeError("upload failed")

    monkeypatch.setattr("dataset_packaging.hf_dataset_packaging.subprocess.run", fail_upload)
    with pytest.raises(RuntimeError, match="upload failed"):
        upload_additional_data(
            uploads=[AdditionalUpload(source, "preprocessed/ACL/shard-00000.tar", (paper,))],
            repo_id="owner/dataset",
            hf_cli="hf",
        )
    assert not seen[0].exists()
    assert (local_temp / "hf-xet-cache").is_dir()
    assert paper.read_text() == "paper"


def test_additional_upload_preserves_explicit_xet_cache(tmp_path, monkeypatch) -> None:
    from dataset_packaging.hf_dataset_packaging import AdditionalUpload

    local_temp = tmp_path / "node-local"
    local_temp.mkdir()
    monkeypatch.setenv("PBS_LOCALDIR", str(local_temp))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "custom-cache"))
    seen = []
    monkeypatch.setattr("dataset_packaging.hf_dataset_packaging.subprocess.run", lambda cmd, **kwargs: seen.append(kwargs["env"]["HF_XET_CACHE"]))
    upload_additional_data(
        uploads=[AdditionalUpload(tmp_path, "splits")],
        repo_id="owner/dataset",
        hf_cli="hf",
        workers=1,
    )
    assert seen == [str(tmp_path / "custom-cache")]


def test_additional_upload_starts_before_scanning_later_domains(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "dataset"
    for name in ("preprocessed", "clues", "query_collections", "splits"):
        (data_root / name).mkdir(parents=True)
    acl = data_root / "preprocessed" / "ACL"
    acl.mkdir()
    (acl / "paper.md").write_text("paper")
    biology = data_root / "preprocessed" / "Biology"
    biology.mkdir()
    (biology / "link").symlink_to(acl / "paper.md")
    commands = []
    monkeypatch.setattr("dataset_packaging.hf_dataset_packaging.subprocess.run", lambda cmd, **kwargs: commands.append(cmd))

    with pytest.raises(ValueError, match="Symlinks"):
        upload_additional_data(
            uploads=prepare_additional_data(data_root=data_root, output_dir=tmp_path / "package", shard_size_bytes=1024),
            repo_id="owner/dataset",
            hf_cli="hf",
        )
    assert [cmd[4] for cmd in commands] == ["preprocessed/ACL"]
