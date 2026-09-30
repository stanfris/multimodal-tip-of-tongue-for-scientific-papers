"""Small shared utilities for long-running generation scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TypeVar

from clues.interpretations import append_failure_record


T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> list[list[T]]:
    if batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def progress(
    items: Iterable[T],
    *,
    total: int,
    enabled: bool,
    description: str,
    unit: str = "batch",
) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    unit_label = {"batch": "batches", "query": "queries"}.get(unit, f"{unit}s")
    for processed, item in enumerate(items, start=1):
        yield item
        percent = int((processed / total) * 100) if total else 100
        print(f"{description}: {processed}/{total} {unit_label} ({percent}%)", flush=True)


def record_generation_failure(
    output_file: str | Path,
    *,
    record_id: str,
    kind: Literal["visual", "textual"],
    error: Exception,
    metadata: dict[str, Any],
    **extra: Any,
) -> None:
    append_failure_record(
        {
            "record_id": record_id,
            "kind": kind,
            "metadata": metadata,
            "error_type": type(error).__name__,
            "error": str(error),
            **extra,
        },
        output_file,
    )
