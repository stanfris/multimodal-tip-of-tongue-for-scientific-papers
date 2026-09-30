from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.stage_preprocessed import copy_completed_papers


def write_paper(root: Path, *, complete: bool = True) -> Path:
    paper = root / "papers" / "ACL" / "ACL_paper"
    paper.mkdir(parents=True)
    (paper / "paper.json").write_text(json.dumps({"paper_id": "ACL_paper"}), encoding="utf-8")
    (paper / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    if complete:
        (paper / "_SUCCESS").touch()
    return paper


def test_copy_completed_mineru_papers_to_direct_preprocessed_layout(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed)
    destination = tmp_path / "preprocessed" / "ACL" / "ACL_paper"

    assert copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train") == 1
    assert (paper / "_SUCCESS").exists()
    assert (destination / "_SUCCESS").exists()
    assert copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train") == 0


def test_copy_leaves_incomplete_papers_for_mineru_retry(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed, complete=False)

    assert copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train") == 0
    assert paper.exists()


def test_copy_also_accepts_domain_directly_under_processed(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed)
    direct_paper = processed / "ACL" / "ACL_paper"
    direct_paper.parent.mkdir()
    paper.rename(direct_paper)

    assert copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train") == 1
    assert direct_paper.exists()
    assert (tmp_path / "preprocessed" / "ACL" / "ACL_paper" / "_SUCCESS").exists()


def test_copy_preserves_existing_preprocessed_paper(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed)
    destination = tmp_path / "preprocessed" / "ACL" / "ACL_paper"
    destination.mkdir(parents=True)
    (destination / "_SUCCESS").touch()
    (destination / "paper.json").write_text("{}", encoding="utf-8")
    (destination / "markdown.md").write_text("# Existing\n", encoding="utf-8")

    marker = destination / "marker"
    marker.touch()
    assert copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train") == 0
    assert marker.exists()
    assert paper.exists()


def test_copy_rejects_incomplete_existing_destination(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed)
    destination = tmp_path / "preprocessed" / "ACL" / "ACL_paper"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Incomplete preprocessed paper"):
        copy_completed_papers(processed, tmp_path / "preprocessed", split_index=None, split="train")
    assert paper.exists()
