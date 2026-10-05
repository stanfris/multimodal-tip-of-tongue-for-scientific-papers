"""Small validation helpers shared by generation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_split_pdf_path(value: str) -> Path:
    """Validate a canonical split entry before joining it to a corpus root."""
    if not value or "\\" in value or value.startswith("/") or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError(f"Invalid relative PDF path in split index: {value!r}")
    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Split index entry is not a PDF path: {value!r}")
    return path


def validate_pdf_split_index(index: Any, source: Path) -> dict[str, list[Path]]:
    """Reject ambiguous train/test membership before any stage reads a split."""
    if not isinstance(index, dict):
        raise ValueError(f"PDF split index must be a JSON object: {source}")
    paths: dict[str, list[Path]] = {}
    for name in ("train", "test"):
        rows = index.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError(f"Split index {source} does not contain a string list for {name!r}")
        entries = [validate_split_pdf_path(row) for row in rows]
        if len(entries) != len(set(entries)):
            raise ValueError(f"Split index {source} contains duplicate paths in {name!r}")
        paths[name] = entries
    overlap = set(paths["train"]) & set(paths["test"])
    if overlap:
        raise ValueError(f"Split index {source} assigns {min(overlap)} to both train and test")
    return paths


def validate_index_window(
    start_index: int,
    end_index: int | None,
    *,
    start_name: str = "start_index",
    end_name: str = "end_index",
) -> None:
    if start_index < 0:
        raise ValueError(f"{start_name} must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError(f"{end_name} must be greater than or equal to {start_name}")
