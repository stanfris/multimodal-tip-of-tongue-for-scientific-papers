"""Small shared utilities for long-running generation scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def batched(items: list[T], batch_size: int) -> list[list[T]]:
    if batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def progress(items: Iterable[T], *, total: int, enabled: bool, description: str) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        processed = 0
        next_report = 10
        for item in items:
            yield item
            processed += 1
            percent = int((processed / total) * 100) if total else 100
            if percent >= next_report or processed == total:
                print(f"{description}: {processed}/{total} rows ({percent}%)", flush=True)
                next_report += 10
        return
    yield from tqdm(items, total=total, desc=description, unit="batch")
