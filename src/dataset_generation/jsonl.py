"""Shared JSONL readers and appenders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_jsonl_objects(path: str | Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    """Read a JSONL file whose nonblank lines must all be JSON objects."""
    data_path = Path(path)
    if not data_path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"JSONL file does not exist: {data_path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {data_path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {data_path}:{line_number}")
        rows.append(row)
    return rows


def append_jsonl_object(
    path: str | Path,
    row: dict[str, Any],
    *,
    sort_keys: bool = True,
    fsync: bool = False,
) -> None:
    """Append one JSON object to a JSONL file."""
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=sort_keys, ensure_ascii=False))
        handle.write("\n")
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())
