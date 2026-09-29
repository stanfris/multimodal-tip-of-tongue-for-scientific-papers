from __future__ import annotations

import json
from pathlib import Path

from preprocessing.preprocessed import iter_preprocessed_paper_dirs, read_preprocessed_papers
from preprocessing.preprocessed import select_preprocessed_paper_dirs


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
    (paper_dir / "_SUCCESS").touch()
    return paper_dir


def test_nested_subset_papers_are_discovered_from_output_root(tmp_path: Path) -> None:
    acl_dir = write_paper(tmp_path, "ACL", "ACL_paper")
    engineering_dir = write_paper(tmp_path, "Engineering", "Engineering_paper")

    assert iter_preprocessed_paper_dirs(tmp_path) == [acl_dir, engineering_dir]
    assert [paper["paper_id"] for paper in read_preprocessed_papers(tmp_path)] == [
        "ACL_paper",
        "Engineering_paper",
    ]


def test_split_selection_resolves_only_extracted_papers(tmp_path: Path) -> None:
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(
        json.dumps({"train": ["ACL/train.pdf"], "test": ["Biology/test.pdf"]}),
        encoding="utf-8",
    )
    train_dir = write_paper(tmp_path / "processed", "ACL", "ACL_train")

    selection = select_preprocessed_paper_dirs(
        tmp_path / "processed",
        split_index=split_index,
        split="train+test",
    )

    assert selection.paper_dirs == [train_dir]
    assert selection.expected_count == 2
    assert selection.missing_count == 1
    assert selection.source == str(split_index)


def test_split_selection_supports_direct_domain_layout(tmp_path: Path) -> None:
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(json.dumps({"train": ["ACL/train.pdf"], "test": []}), encoding="utf-8")
    write_paper(tmp_path / "processed", "ACL", "ACL_train")
    (tmp_path / "processed" / "papers").rename(tmp_path / "nested")
    direct_dir = tmp_path / "processed" / "ACL" / "ACL_train"
    direct_dir.parent.mkdir()
    (tmp_path / "nested" / "ACL" / "ACL_train").rename(direct_dir)

    selection = select_preprocessed_paper_dirs(
        tmp_path / "processed",
        split_index=split_index,
        split="train",
    )

    assert selection.paper_dirs == [direct_dir]


def test_split_selection_supports_domain_then_papers_layout(tmp_path: Path) -> None:
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(json.dumps({"train": ["Physics/2112.05988.pdf"], "test": []}), encoding="utf-8")
    nested_dir = write_paper(tmp_path / "staging", "Physics", "Physics_2112.05988")
    paper_dir = tmp_path / "processed" / "Physics" / "papers" / "Physics_2112.05988"
    paper_dir.parent.mkdir(parents=True)
    nested_dir.rename(paper_dir)

    selection = select_preprocessed_paper_dirs(
        tmp_path / "processed",
        split_index=split_index,
        split="train",
    )

    assert selection.paper_dirs == [paper_dir]
