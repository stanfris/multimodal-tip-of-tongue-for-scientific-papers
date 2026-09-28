from __future__ import annotations

import json

import pytest

from document_splits.document_splits import filter_papers_by_split


def test_filter_uses_canonical_split_embedded_by_extraction() -> None:
    papers = [
        {"paper_id": "train-paper", "split": "train"},
        {"paper_id": "test-paper", "split": "test"},
    ]

    assert filter_papers_by_split(
        papers,
        split_index_path=None,
        split_name="train",
    ) == [papers[0]]
    assert filter_papers_by_split(
        papers,
        split_index_path=None,
        split_name="test",
    ) == [papers[1]]


def test_filter_rejects_missing_canonical_split_metadata() -> None:
    with pytest.raises(ValueError, match="canonical PDF split metadata"):
        filter_papers_by_split(
            [{"paper_id": "unsplit-paper"}],
            split_index_path=None,
            split_name="train",
        )


def test_filter_retains_legacy_index_compatibility(tmp_path) -> None:
    split_index = tmp_path / "document_split.json"
    split_index.write_text(json.dumps({"train": ["paper-b"], "test": ["paper-a"]}), encoding="utf-8")
    papers = [{"paper_id": "paper-a"}, {"paper_id": "paper-b"}]

    assert filter_papers_by_split(
        papers,
        split_index_path=split_index,
        split_name="train",
    ) == [papers[1]]
