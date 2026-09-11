from __future__ import annotations

import pytest

from dataset_generation.validation import validate_index_window


def test_validate_index_window_accepts_open_ended_range() -> None:
    validate_index_window(3, None)


def test_validate_index_window_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_index must be non-negative"):
        validate_index_window(-1, None)


def test_validate_index_window_rejects_reversed_window_with_cli_names() -> None:
    with pytest.raises(ValueError, match="end-index must be greater than or equal to start-index"):
        validate_index_window(5, 4, start_name="start-index", end_name="end-index")
