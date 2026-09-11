from __future__ import annotations

from pathlib import Path

import pytest

from dataset_generation.jsonl import append_jsonl_object, read_jsonl_objects


def test_read_jsonl_objects_missing_ok(tmp_path: Path) -> None:
    assert read_jsonl_objects(tmp_path / "missing.jsonl", missing_ok=True) == []


def test_read_jsonl_objects_rejects_non_objects(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": true}\n["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        read_jsonl_objects(path)


def test_append_jsonl_object_round_trips_utf8(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"

    append_jsonl_object(path, {"query": "visual clue", "paper_id": "p1"})
    append_jsonl_object(path, {"query": "textual clue", "paper_id": "p2"})

    assert read_jsonl_objects(path) == [
        {"paper_id": "p1", "query": "visual clue"},
        {"paper_id": "p2", "query": "textual clue"},
    ]
