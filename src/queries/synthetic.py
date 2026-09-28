"""Interfaces for future synthetic retrieval/evaluation collection generation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TestCollectionExample:
    query_id: str
    query: str
    relevant_ids: list[str]
    negative_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyntheticCollectionConfig:
    seed: int = 13
    max_examples: int | None = None
    negative_examples_per_query: int = 0
    method: str = "label-template"
    model: str | None = None
    prompt_version: str | None = None


def generate_synthetic_queries(
    records: Iterable[dict[str, Any]],
    config: SyntheticCollectionConfig,
) -> list[TestCollectionExample]:
    """Generate deterministic placeholder queries without binding to an LLM provider."""
    rng = random.Random(config.seed)
    materialized = [record for record in records if record.get("match_status") == "matched"]
    rng.shuffle(materialized)
    if config.max_examples is not None:
        materialized = materialized[: config.max_examples]

    all_ids = [record["record_id"] for record in materialized]
    examples: list[TestCollectionExample] = []
    for index, record in enumerate(materialized):
        label = record.get("label") or "figure"
        relevant_id = record["record_id"]
        candidates = [record_id for record_id in all_ids if record_id != relevant_id]
        negative_ids = candidates[: config.negative_examples_per_query]
        examples.append(
            TestCollectionExample(
                query_id=f"q{index:05d}",
                query=f"Find ACL paper figures labeled {label}.",
                relevant_ids=[relevant_id],
                negative_ids=negative_ids,
                metadata={
                    "method": config.method,
                    "model": config.model,
                    "prompt_version": config.prompt_version,
                    "seed": config.seed,
                    "source_record_id": relevant_id,
                    "resolved_paper_id": record.get("resolved_paper_id", ""),
                },
            )
        )
    return examples


def write_test_collection(collection: list[TestCollectionExample], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = [asdict(example) for example in collection]
    (output_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(example, sort_keys=True) for example in data) + ("\n" if data else ""),
        encoding="utf-8",
    )
    return output_path
