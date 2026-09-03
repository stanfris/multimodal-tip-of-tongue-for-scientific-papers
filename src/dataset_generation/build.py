"""Reproducible build pipeline for ACL-fig plus ACL Anthology markdown."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_generation.matching import match_records_to_papers
from dataset_generation.schema import validate_records
from dataset_generation.sources import (
    ACL_ANTHOLOGY_MD_CONFIG,
    ACL_ANTHOLOGY_MD_DATASET,
    ACL_FIG_DATASET,
    DEFAULT_DATA_DIR,
    load_acl_fig_records,
    load_acl_markdown_papers,
)


@dataclass(frozen=True)
class BuildConfig:
    fig_dataset: str = ACL_FIG_DATASET
    fig_split: str = "train"
    paper_dataset: str = ACL_ANTHOLOGY_MD_DATASET
    paper_config: str = ACL_ANTHOLOGY_MD_CONFIG
    paper_split: str = "train"
    data_dir: str = str(DEFAULT_DATA_DIR)
    prefer_local_sources: bool = True
    streaming: bool = False
    limit: int | None = None
    paper_limit: int | None = None
    seed: int = 13


def build_combined_dataset(
    config: BuildConfig,
    records: Iterable[dict[str, Any]] | None = None,
    papers: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build enriched records, coverage stats, and reproducibility metadata."""
    figure_records = list(records) if records is not None else load_acl_fig_records(
        split=config.fig_split,
        limit=config.limit,
        data_dir=config.data_dir,
        prefer_local=config.prefer_local_sources,
    )
    paper_records = papers if papers is not None else load_acl_markdown_papers(
        split=config.paper_split,
        streaming=config.streaming,
        limit=config.paper_limit,
        data_dir=config.data_dir,
        prefer_local=config.prefer_local_sources,
    )

    enriched, match_stats = match_records_to_papers(figure_records, paper_records)
    for record in enriched:
        record["source_paper_dataset"] = config.paper_dataset
        record["source_paper_config"] = config.paper_config
        record["source_paper_split"] = config.paper_split

    validate_records(enriched)

    stats = {
        **match_stats,
        "matched_percent": round(match_stats["match_rate"] * 100, 4),
        "source_fig_dataset": config.fig_dataset,
        "source_fig_split": config.fig_split,
        "source_paper_dataset": config.paper_dataset,
        "source_paper_config": config.paper_config,
        "source_paper_split": config.paper_split,
    }
    metadata = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "sources": {
            "figure": {"name": config.fig_dataset, "split": config.fig_split},
            "paper_markdown": {
                "name": config.paper_dataset,
                "config": config.paper_config,
                "split": config.paper_split,
            },
        },
        "seed": config.seed,
        "stats": stats,
        "layout": {
            "raw_sources": str(Path(config.data_dir) / "raw" / "hf"),
            "processed_artifact": "data.parquet plus metadata.json and stats.json",
            "interpretations": "data/interim/interpretations/<run_id>/interpretations.jsonl",
            "query_collections": "data/query_collections/<collection_id>/queries.jsonl",
        },
    }
    return enriched, stats, metadata
