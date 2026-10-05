from __future__ import annotations

import pytest

from pathlib import Path

from common.validation import validate_index_window, validate_pdf_split_index, validate_split_pdf_path


def test_validate_index_window_accepts_open_ended_range() -> None:
    validate_index_window(3, None)


def test_validate_index_window_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_index must be non-negative"):
        validate_index_window(-1, None)


def test_validate_index_window_rejects_reversed_window_with_cli_names() -> None:
    with pytest.raises(ValueError, match="end-index must be greater than or equal to start-index"):
        validate_index_window(5, 4, start_name="start-index", end_name="end-index")


@pytest.mark.parametrize("entry", ["../outside.pdf", "/tmp/outside.pdf", "ACL//paper.pdf", "ACL/./paper.pdf", "ACL\\paper.pdf", "ACL/paper.txt"])
def test_split_pdf_paths_cannot_escape_or_misidentify_corpus(entry: str) -> None:
    with pytest.raises(ValueError):
        validate_split_pdf_path(entry)


def test_split_pdf_path_accepts_nested_source_paths() -> None:
    assert validate_split_pdf_path("ACL/2024/paper.pdf").as_posix() == "ACL/2024/paper.pdf"


def test_pdf_split_rejects_overlap_and_duplicates() -> None:
    source = Path("split.json")
    with pytest.raises(ValueError, match="both train and test"):
        validate_pdf_split_index({"train": ["ACL/a.pdf"], "test": ["ACL/a.pdf"]}, source)
    with pytest.raises(ValueError, match="duplicate paths"):
        validate_pdf_split_index({"train": ["ACL/a.pdf", "ACL/a.pdf"], "test": []}, source)
