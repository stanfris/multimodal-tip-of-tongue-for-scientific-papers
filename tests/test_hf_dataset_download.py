from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from dataset_packaging.download_hf_dataset import allow_patterns, download_dataset, main, restore_archive_component


def write_tar(path: Path, members: dict[str, bytes], *, link: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if link:
            info = tarfile.TarInfo("linked")
            info.type = tarfile.SYMTYPE
            info.linkname = link
            archive.addfile(info)


def make_hub_fixture(root: Path) -> None:
    pdf = b"%PDF-1.4\nfixture\n%%EOF\n"
    write_tar(root / "data" / "ACL" / "shard-00000.tar", {"doc.pdf": pdf})
    pd.DataFrame([{
        "source": "ACL", "original_relative_path": "nested/paper.pdf",
        "shard": "data/ACL/shard-00000.tar", "member_path": "doc.pdf",
        "size_bytes": len(pdf), "sha256": hashlib.sha256(pdf).hexdigest(),
    }]).to_parquet(root / "metadata.parquet")
    write_tar(root / "preprocessed" / "ACL" / "shard-00000.tar", {"paper/text.json": b"{}"})
    write_tar(root / "clues" / "Physics" / "shard-00000.tar", {"paper/clue.jsonl": b"{}\n"})
    (root / "preprocessed" / "metadata.json").write_text("{}")
    for name in ("query_collections", "splits"):
        path = root / name / "ACL" / "items.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")


def test_selected_downloads_restore_layout_and_resume(tmp_path: Path, monkeypatch) -> None:
    fixture = tmp_path / "fixture"
    make_hub_fixture(fixture)
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        for path in fixture.rglob("*"):
            if path.is_file():
                relative = path.relative_to(fixture)
                # Mock the Hub's allow_patterns filtering.
                from fnmatch import fnmatch
                if any(fnmatch(relative.as_posix(), pattern) for pattern in kwargs["allow_patterns"]):
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        destination.write_bytes(path.read_bytes())
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    data = tmp_path / "data"
    download_dataset(data_root=data, repo_id="owner/fixture", components={"clues"})
    assert (data / "clues" / "Physics" / "paper" / "clue.jsonl").read_bytes() == b"{}\n"
    assert not (data / "pdf_datasets").exists()
    assert calls[0]["allow_patterns"] == ["clues/**"]
    assert "*.manifest.json" in calls[0]["ignore_patterns"]

    download_dataset(data_root=data, repo_id="owner/fixture", components=set(("pdfs", "preprocessed", "clues", "query_collections", "splits")))
    assert (data / "pdf_datasets" / "ACL" / "nested" / "paper.pdf").read_bytes().startswith(b"%PDF")
    assert (data / "preprocessed" / "ACL" / "paper" / "text.json").read_bytes() == b"{}"
    assert (data / "preprocessed" / "metadata.json").read_text() == "{}"
    assert (data / "query_collections" / "ACL" / "items.jsonl").exists()
    assert (data / "splits" / "ACL" / "items.jsonl").exists()
    assert "metadata.parquet" in calls[1]["allow_patterns"]

    download_dataset(data_root=data, repo_id="owner/fixture", components={"pdfs", "preprocessed"})
    assert len(calls) == 3

    pdf_path = data / "pdf_datasets" / "ACL" / "nested" / "paper.pdf"
    pdf_path.write_bytes(b"x" * pdf_path.stat().st_size)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        download_dataset(data_root=data, repo_id="owner/fixture", components={"pdfs"})
    download_dataset(data_root=data, repo_id="owner/fixture", components={"pdfs"}, overwrite=True)
    assert pdf_path.read_bytes().startswith(b"%PDF")

    clue_path = data / "clues" / "Physics" / "paper" / "clue.jsonl"
    clue_path.write_bytes(b"bad")
    with pytest.raises(FileExistsError, match="Existing file differs"):
        download_dataset(data_root=data, repo_id="owner/fixture", components={"clues"})
    download_dataset(data_root=data, repo_id="owner/fixture", components={"clues"}, overwrite=True)
    assert clue_path.read_bytes() == b"{}\n"


@pytest.mark.parametrize("member", ["../escape", "/absolute", "sub/../../escape", "sub\\escape"])
def test_archive_rejects_unsafe_paths(tmp_path: Path, member: str) -> None:
    source = tmp_path / "source"
    write_tar(source / "ACL" / "shard-00000.tar", {"okay.txt": b"okay", member: b"bad"})
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="Unsafe TAR member"):
        restore_archive_component(source, destination)
    assert not destination.exists()


def test_archive_rejects_links_and_destination_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_tar(source / "ACL" / "shard-00000.tar", {"paper/file": b"data"}, link="../../outside")
    with pytest.raises(ValueError, match="Unsupported"):
        restore_archive_component(source, tmp_path / "destination")
    write_tar(source / "ACL" / "shard-00000.tar", {"paper/file": b"data"})
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "ACL").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Symlink"):
        restore_archive_component(source, destination)


def test_component_selection_cli(tmp_path: Path, monkeypatch) -> None:
    captured = []
    monkeypatch.setattr("dataset_packaging.download_hf_dataset.download_dataset", lambda **kwargs: captured.append(kwargs))
    assert main([str(tmp_path), "--generated"]) == 0
    assert captured[0]["components"] == {"preprocessed", "clues", "query_collections", "splits"}
    assert allow_patterns({"pdfs"}) == ["metadata.parquet", "data/*/shard-*.tar"]
    with pytest.raises(SystemExit):
        main([str(tmp_path)])
