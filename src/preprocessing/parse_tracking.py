"""Track documents that still need extraction or compaction work."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INCOMPLETE_DOCUMENTS_FILENAME = "incomplete_documents.json"


def write_incomplete_documents(
    output_dir: Path,
    documents: list[dict[str, Any]],
    *,
    stage: str,
    source: str,
) -> Path:
    """Write a resumable manifest of documents that are not fully processed.

    The manifest is deliberately a snapshot rather than an append-only log: each
    run rewrites it with the current unfinished set, so a clean rerun naturally
    removes documents that have since completed.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / INCOMPLETE_DOCUMENTS_FILENAME
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "source": source,
        "incomplete_count": len(documents),
        "documents": sorted(documents, key=_document_sort_key),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_incomplete_documents(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Incomplete document manifest does not contain a documents list: {path}")
    return [document for document in documents if isinstance(document, dict)]


def _document_sort_key(document: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(document.get("source_paper_dataset") or ""),
        str(document.get("paper_id") or ""),
        str(document.get("source_pdf_relpath") or ""),
    )
