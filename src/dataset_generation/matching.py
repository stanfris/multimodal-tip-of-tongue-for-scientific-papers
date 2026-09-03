"""Matching utilities for ACL figure records and ACL Anthology markdown."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse


def normalize_id(paper_id: str | None) -> str:
    """Normalize an ACL Anthology ID to a canonical lookup key."""
    if not paper_id:
        return ""

    normalized = str(paper_id).strip()

    if normalized.startswith(("http://", "https://")):
        parsed = urlparse(normalized)
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            normalized = parts[-1]

    if normalized.endswith(".pdf"):
        normalized = normalized[:-4]

    normalized = normalized.lower()

    if normalized.endswith(".dataset"):
        normalized = normalized[:-8]

    normalized = re.sub(r"v\d+$", "", normalized)

    return normalized.strip()


def extract_paper_id_from_filename(filename: str | None) -> str:
    """Extract a paper ID from an ACL-fig filename or path."""
    if not filename:
        return ""

    basename = os.path.basename(str(filename))
    if ".pdf-" in basename:
        return basename.split(".pdf-")[0]
    return os.path.splitext(basename)[0]


def _paper_markdown_map(paper_iterator: Iterable[dict[str, Any]]) -> dict[str, str]:
    paper_map: dict[str, str] = {}
    for paper in paper_iterator:
        anthology_id = (
            paper.get("anthology_id")
            or paper.get("paper_id")
            or paper.get("id")
            or paper.get("acl_id")
            or ""
        )
        normalized_id = normalize_id(anthology_id)
        if normalized_id:
            paper_map[normalized_id] = paper.get("markdown") or paper.get("text") or ""
    return paper_map


def _fallback_ids(normalized_id: str) -> tuple[str, ...]:
    return (
        f"{normalized_id}.0",
        f"{normalized_id}00",
        f"{normalized_id}000",
        f"{normalized_id}0",
    )


def match_records_to_papers(
    records: Iterable[dict[str, Any]],
    paper_iterator: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match figure records to markdown records using normalized ACL IDs.

    This preserves the prototype matching behavior, including the proceedings
    fallback suffixes used by the original `matcher.py`.
    """
    paper_map = _paper_markdown_map(paper_iterator)
    enriched_records: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    unmatched_ids: set[str] = set()

    for record in records:
        enriched = dict(record)
        normalized_id = (
            enriched.get("normalized_paper_id")
            or enriched.get("normalized_id")
            or normalize_id(enriched.get("extracted_paper_id") or enriched.get("extracted_id"))
        )
        normalized_id = str(normalized_id)
        enriched["normalized_paper_id"] = normalized_id
        enriched.setdefault("normalized_id", normalized_id)

        markdown = paper_map.get(normalized_id)
        matched_id = normalized_id

        if markdown is None and normalized_id:
            for fallback_id in _fallback_ids(normalized_id):
                if fallback_id in paper_map:
                    matched_id = fallback_id
                    markdown = paper_map[fallback_id]
                    break

        if markdown is not None:
            enriched["markdown"] = markdown
            enriched["match_status"] = "matched"
            enriched["resolved_paper_id"] = matched_id
            matched_ids.add(normalized_id)
        else:
            enriched["markdown"] = ""
            enriched["match_status"] = "unmatched"
            enriched["resolved_paper_id"] = ""
            unmatched_ids.add(normalized_id)

        enriched_records.append(enriched)

    total_records = len(enriched_records)
    matched_records = sum(1 for record in enriched_records if record["match_status"] == "matched")

    stats = {
        "total_records": total_records,
        "matched_records": matched_records,
        "unmatched_records": total_records - matched_records,
        "match_rate": matched_records / total_records if total_records else 0.0,
        "unique_matched_ids": len(matched_ids),
        "unique_unmatched_ids": len(unmatched_ids),
    }
    return enriched_records, stats
