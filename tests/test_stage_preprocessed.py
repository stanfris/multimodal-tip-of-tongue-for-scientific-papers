from __future__ import annotations

import json
from pathlib import Path

import pytest

from preprocessing.stage_preprocessed import copy_markdown_and_images


def write_paper(root: Path, *, complete: bool = True) -> Path:
    paper = root / "papers" / "ACL" / "ACL_paper"
    paper.mkdir(parents=True)
    (paper / "paper.json").write_text(json.dumps({"paper_id": "ACL_paper"}), encoding="utf-8")
    (paper / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    if complete:
        (paper / "_SUCCESS").touch()
    return paper


def test_markdown_images_domain_layout_and_idempotence(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    paper = write_paper(processed)
    raw_images = paper / "mineru" / "raw" / "hybrid_auto" / "images"
    raw_images.mkdir(parents=True)
    (raw_images / "fig.png").write_bytes(b"image")
    (raw_images / "ignore.json").write_text("{}", encoding="utf-8")
    other = processed / "papers" / "Physics" / "Physics_paper"
    other.mkdir(parents=True)
    (other / "_SUCCESS").touch()
    (other / "paper.json").write_text("{}", encoding="utf-8")
    (other / "markdown.md").write_text("Other", encoding="utf-8")
    target = tmp_path / "preprocessed"

    options = dict(domains=["ACL"], workers=2)
    assert copy_markdown_and_images(processed, target, dry_run=True, **options) == 1
    assert not target.exists()
    assert copy_markdown_and_images(processed, target, **options) == 1
    assert copy_markdown_and_images(processed, target, **options) == 0
    destination = target / "ACL" / "ACL_paper"
    assert {path.name for path in destination.iterdir()} == {"markdown.md", "images"}
    assert [path.name for path in (destination / "images").iterdir()] == ["fig.png"]
    assert not (target / "Physics").exists()
    assert (paper / "paper.json").exists()


def test_markdown_images_rejects_incomplete_existing_destination(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_paper(processed)
    destination = tmp_path / "preprocessed" / "ACL" / "ACL_paper"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Incomplete or incompatible destination"):
        copy_markdown_and_images(processed, tmp_path / "preprocessed", domains=["ACL"])


def test_markdown_images_requires_completed_paper(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_paper(processed, complete=False)

    with pytest.raises(ValueError, match="No completed papers"):
        copy_markdown_and_images(processed, tmp_path / "preprocessed", domains=["ACL"])
