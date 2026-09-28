from __future__ import annotations

import json
from pathlib import Path

from preprocessing.preprocessed import iter_preprocessed_paper_dirs, read_preprocessed_papers


def write_paper(root: Path, subset: str, paper_id: str) -> Path:
    paper_dir = root / "papers" / subset / paper_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "markdown.md").write_text(f"# {paper_id}\n", encoding="utf-8")
    (paper_dir / "paper.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "source_paper_dataset": subset,
                "split": "train",
                "figures": [],
            }
        ),
        encoding="utf-8",
    )
    return paper_dir


def test_nested_subset_papers_are_discovered_from_output_root(tmp_path: Path) -> None:
    acl_dir = write_paper(tmp_path, "ACL", "ACL_paper")
    engineering_dir = write_paper(tmp_path, "Engineering", "Engineering_paper")

    assert iter_preprocessed_paper_dirs(tmp_path) == [acl_dir, engineering_dir]
    assert [paper["paper_id"] for paper in read_preprocessed_papers(tmp_path)] == [
        "ACL_paper",
        "Engineering_paper",
    ]
