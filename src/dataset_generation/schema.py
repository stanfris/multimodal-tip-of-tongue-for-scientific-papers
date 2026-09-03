"""Output schema validation for the combined dataset."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


REQUIRED_FIELDS = (
    "record_id",
    "filename",
    "extracted_paper_id",
    "normalized_paper_id",
    "resolved_paper_id",
    "label",
    "markdown",
    "match_status",
    "source_fig_dataset",
    "source_fig_split",
    "source_paper_dataset",
    "source_paper_config",
)

MATCH_STATUSES = {"matched", "unmatched"}


def validate_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record is missing required fields: {', '.join(missing)}")
    if record["match_status"] not in MATCH_STATUSES:
        raise ValueError(f"Invalid match_status: {record['match_status']!r}")


def validate_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        validate_record(record)


def records_to_dataframe(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    materialized = list(records)
    validate_records(materialized)
    return pd.DataFrame(materialized)
