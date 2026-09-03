"""Persistence helpers for generated dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_generation.schema import records_to_dataframe, validate_records


SUPPORTED_FORMATS = {"parquet", "jsonl", "csv"}


def write_dataset_artifact(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    metadata: dict[str, Any],
    stats: dict[str, Any],
    file_format: str = "parquet",
) -> Path:
    file_format = file_format.lower()
    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {file_format!r}; expected one of {sorted(SUPPORTED_FORMATS)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validate_records(records)
    frame = records_to_dataframe(records)
    data_path = output_path / f"data.{file_format}"
    if file_format == "parquet":
        frame.to_parquet(data_path, index=False)
    elif file_format == "jsonl":
        frame.to_json(data_path, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_csv(data_path, index=False)

    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def read_dataset_artifact(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_dir)
    for file_format in ("parquet", "jsonl", "csv"):
        data_path = dataset_path / f"data.{file_format}"
        if data_path.exists():
            if file_format == "parquet":
                return pd.read_parquet(data_path)
            if file_format == "jsonl":
                return pd.read_json(data_path, orient="records", lines=True)
            return pd.read_csv(data_path)
    raise FileNotFoundError(f"No data.parquet, data.jsonl, or data.csv found in {dataset_path}")


def read_stats(dataset_dir: str | Path) -> dict[str, Any]:
    stats_path = Path(dataset_dir) / "stats.json"
    return json.loads(stats_path.read_text(encoding="utf-8"))
