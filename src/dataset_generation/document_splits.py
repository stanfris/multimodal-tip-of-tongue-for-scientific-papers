"""Build and apply stable train/test document split indexes."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_TRAIN_SIZE = 1000
DEFAULT_TEST_SIZE = 200
DEFAULT_SPLIT_NAME = "document_split"


@dataclass(frozen=True)
class SplitIndex:
    train: list[str]
    test: list[str]
    metadata: dict[str, Any]


def build_split_index(
    papers: Iterable[dict[str, Any]],
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    test_fraction: float | None = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    stratify_field: str | None = None,
) -> SplitIndex:
    if train_size is not None and train_size < 1:
        raise ValueError("train_size must be at least 1 when set")
    if test_size is not None and test_size < 1:
        raise ValueError("test_size must be at least 1 when set")
    if (train_size is None) != (test_size is None):
        raise ValueError("train_size and test_size must be provided together")
    if test_fraction is not None and not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    paper_rows = list(papers)
    groups: dict[str, list[str]] = defaultdict(list)
    for paper in paper_rows:
        paper_id = str(paper["paper_id"])
        groups[_stratum_for_paper(paper, stratify_field)].append(paper_id)

    train: list[str] = []
    test: list[str] = []
    rng = random.Random(seed)
    shuffled_groups = {stratum: _shuffled(ids, rng) for stratum, ids in sorted(groups.items())}
    if train_size is not None and test_size is not None:
        if train_size + test_size > len(paper_rows):
            raise ValueError(
                f"Requested {train_size} train and {test_size} test documents, "
                f"but only {len(paper_rows)} papers are available."
            )
        test_counts = _allocate_counts(
            {stratum: len(ids) for stratum, ids in shuffled_groups.items()},
            test_size,
        )
        remaining_groups: dict[str, list[str]] = {}
        for stratum, ids in shuffled_groups.items():
            test_count = test_counts[stratum]
            test.extend(ids[:test_count])
            remaining_groups[stratum] = ids[test_count:]
        train_counts = _allocate_counts(
            {stratum: len(ids) for stratum, ids in remaining_groups.items()},
            train_size,
        )
        for stratum, ids in remaining_groups.items():
            train.extend(ids[: train_counts[stratum]])
    else:
        if test_fraction is None:
            raise ValueError("test_fraction is required when fixed train/test sizes are not set")
        for stratum in sorted(shuffled_groups):
            ids = shuffled_groups[stratum]
            test_count = _test_count(len(ids), test_fraction)
            test.extend(ids[:test_count])
            train.extend(ids[test_count:])

    train.sort()
    test.sort()
    counts_by_stratum = {
        stratum: {"total": len(ids), "train": 0, "test": 0}
        for stratum, ids in sorted(groups.items())
    }
    stratum_by_id = {paper_id: stratum for stratum, ids in groups.items() for paper_id in ids}
    for paper_id in train:
        counts_by_stratum[stratum_by_id[paper_id]]["train"] += 1
    for paper_id in test:
        counts_by_stratum[stratum_by_id[paper_id]]["test"] += 1

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_fraction": test_fraction,
        "train_size": train_size,
        "test_size": test_size,
        "stratify_field": stratify_field,
        "paper_count": len(paper_rows),
        "train_count": len(train),
        "test_count": len(test),
        "counts_by_stratum": counts_by_stratum,
    }
    return SplitIndex(train=train, test=test, metadata=metadata)


def write_split_index(split_index: SplitIndex, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "train": split_index.train,
                "test": split_index.test,
                "metadata": split_index.metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def read_split_paper_ids(path: str | Path, split_name: str) -> list[str]:
    split_path = Path(path)
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Split index must be a JSON object: {split_path}")
    values = raw.get(split_name)
    if not isinstance(values, list):
        raise ValueError(f"Split index {split_path} does not contain list split {split_name!r}")
    paper_ids = [str(value) for value in values]
    duplicates = [paper_id for paper_id, count in Counter(paper_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Split index {split_path} contains duplicate paper IDs in {split_name}: {duplicates[:5]}")
    return paper_ids


def filter_papers_by_split(
    papers: Iterable[dict[str, Any]],
    *,
    split_index_path: str | Path | None,
    split_name: str | None,
) -> list[dict[str, Any]]:
    paper_rows = list(papers)
    if split_index_path is None:
        return paper_rows
    if split_name is None:
        raise ValueError("split_name is required when split_index_path is set")
    by_id = {str(paper["paper_id"]): paper for paper in paper_rows}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for paper_id in read_split_paper_ids(split_index_path, split_name):
        paper = by_id.get(paper_id)
        if paper is None:
            missing.append(paper_id)
            continue
        selected.append(paper)
    if missing:
        raise ValueError(
            f"Split index {split_index_path} references {len(missing)} paper IDs that are not in the dataset; "
            f"first missing ID: {missing[0]}"
        )
    return selected


def _test_count(group_size: int, test_fraction: float) -> int:
    if group_size <= 1:
        return 0
    return min(group_size - 1, max(1, round(group_size * test_fraction)))


def _shuffled(ids: list[str], rng: random.Random) -> list[str]:
    copied = list(ids)
    rng.shuffle(copied)
    return copied


def _allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    capacity = sum(group_sizes.values())
    if total > capacity:
        raise ValueError(f"Cannot allocate {total} documents across only {capacity} available documents.")
    if total == 0:
        return {stratum: 0 for stratum in group_sizes}

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, size in sorted(group_sizes.items()):
        exact = total * (size / capacity) if capacity else 0
        count = min(size, int(exact))
        allocations[stratum] = count
        remainders.append((exact - count, stratum))

    remaining = total - sum(allocations.values())
    for _, stratum in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        if allocations[stratum] >= group_sizes[stratum]:
            continue
        allocations[stratum] += 1
        remaining -= 1

    while remaining > 0:
        changed = False
        for stratum in sorted(group_sizes):
            if allocations[stratum] >= group_sizes[stratum]:
                continue
            allocations[stratum] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            raise RuntimeError("Failed to allocate requested document count.")
    return allocations


def _stratum_for_paper(paper: dict[str, Any], stratify_field: str | None) -> str:
    if stratify_field:
        value = paper.get(stratify_field)
        return str(value) if value not in (None, "") else "unknown"
    venue = paper.get("venue") or paper.get("source_paper_dataset") or paper.get("source_fig_dataset") or "unknown"
    year = paper.get("year") or "unknown"
    return f"{venue}:{year}"
