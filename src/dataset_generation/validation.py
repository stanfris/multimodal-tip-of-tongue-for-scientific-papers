"""Small validation helpers shared by generation commands."""

from __future__ import annotations


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
