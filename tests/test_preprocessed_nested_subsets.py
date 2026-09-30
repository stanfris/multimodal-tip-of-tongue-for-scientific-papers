from __future__ import annotations

import json
from pathlib import Path

from preprocessing.preprocessed import iter_preprocessed_paper_dirs, read_preprocessed_papers
from preprocessing.preprocessed import select_preprocessed_paper_dirs
from document_splits.document_splits import filter_papers_by_split
from clues.vl_figure_descriptions import iter_samples_from_preprocessed_dataset


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


def test_existing_paper_json_with_empty_figures_uses_images_for_visual_samples(tmp_path: Path) -> None:
    paper_dir = write_paper(tmp_path, "ACL", "ACL_paper")
    images_dir = paper_dir / "images"
    images_dir.mkdir()
    (images_dir / "first.jpg").write_bytes(b"image")
    (images_dir / "second.png").write_bytes(b"image")
    (images_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    split_index = tmp_path / "split.json"
    split_index.write_text(json.dumps({"train": ["ACL_paper"], "test": []}), encoding="utf-8")

    samples = iter_samples_from_preprocessed_dataset(
        tmp_path,
        limit=None,
        split_index=split_index,
        split_name="train",
    )

    assert [sample.record_id for sample in samples] == ["first", "second"]
    assert [sample.image_path for sample in samples] == [
        images_dir / "first.jpg",
        images_dir / "second.png",
    ]


def test_snellius_processed_layout_uses_pdf_split_paths(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    paper_dir = write_paper(root, "ACL", "ACL_first")
    paper_json = paper_dir / "paper.json"
    paper = json.loads(paper_json.read_text(encoding="utf-8"))
    paper.update(source_pdf_relpath="ACL/first.pdf", split=None)
    paper_json.write_text(json.dumps(paper), encoding="utf-8")
    (root / "ACL").mkdir(exist_ok=True)
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(json.dumps({"train": ["ACL/first.pdf"], "test": []}), encoding="utf-8")

    papers = read_preprocessed_papers(root)
    assert [paper["paper_id"] for paper in papers] == ["ACL_first"]
    assert filter_papers_by_split(
        papers, split_index_path=split_index, split_name="train"
    ) == papers


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
    assert selection.missing_documents == [
        {
            "stage": "extraction",
            "status": "missing_extraction",
            "source_pdf_relpath": "Biology/test.pdf",
            "source_paper_dataset": "Biology",
            "paper_id": "Biology_test",
            "candidate_paper_ids": ("Biology_test", "Biology_test.6ee2972bc564"),
        }
    ]


def test_split_selection_supports_direct_domain_layout(tmp_path: Path) -> None:
    split_index = tmp_path / "pdf_dataset_split.json"
    split_index.write_text(json.dumps({"train": ["ACL/train.pdf"], "test": []}), encoding="utf-8")
    root = tmp_path / "preprocessed"
    direct_dir = root / "ACL" / "ACL_train"
    direct_dir.mkdir(parents=True)
    (direct_dir / "markdown.md").write_text("# ACL_train\n", encoding="utf-8")
    (direct_dir / "paper.json").write_text(json.dumps({"paper_id": "ACL_train"}), encoding="utf-8")
    (direct_dir / "_SUCCESS").touch()

    selection = select_preprocessed_paper_dirs(
        root,
        split_index=split_index,
        split="train",
    )

    assert selection.paper_dirs == [direct_dir]
    assert [paper["paper_dir"] for paper in read_preprocessed_papers(root)] == [str(direct_dir)]


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
