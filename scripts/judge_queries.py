#!/usr/bin/env python
"""Judge groundedness for existing generated query collections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dataset_generation.document_splits import filter_papers_by_split
from dataset_generation.preprocessed import DEFAULT_DATA_DIR, read_preprocessed_papers
from dataset_generation.query_generation import (
    DEFAULT_JUDGEMENT_MODEL,
    DEFAULT_JUDGEMENT_PROMPT,
    MemoryComponent,
    QueryGenerationConfig,
    judge_query,
    load_query_judge,
)


MODE_DIRS = {
    "visual-only": "visual_only",
    "visual-and-text": "visual_and_text",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge groundedness for existing query JSONL files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split-index", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--collection-id", default=None)
    parser.add_argument("--mode", choices=sorted(MODE_DIRS), action="append", default=None)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_JUDGEMENT_PROMPT)
    parser.add_argument("--prompt-id", default="query_judgement")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--model", default=DEFAULT_JUDGEMENT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset or (args.data_dir / "preprocessed")
    input_dir = args.input_dir or (args.data_dir / "query_collections" / (args.collection_id or f"query_generation_{args.split}"))
    modes = args.mode or ["visual-only", "visual-and-text"]
    papers = {
        str(paper["paper_id"]): paper
        for paper in filter_papers_by_split(
            read_preprocessed_papers(dataset),
            split_index_path=args.split_index,
            split_name=args.split if args.split_index is not None else None,
        )
    }
    config = QueryGenerationConfig(
        dataset=dataset,
        visual_interpretations=None,
        textual_interpretations=None,
        clues_dir=args.data_dir / "clues",
        output_dir=input_dir.parent,
        judge_queries=True,
        judgement_prompt=args.prompt,
        judgement_prompt_id=args.prompt_id,
        judgement_prompt_version=args.prompt_version,
        judgement_model=args.model,
        judgement_temperature=args.temperature,
        judgement_max_tokens=args.max_tokens,
        judgement_device_map=args.device_map,
        judgement_dtype=args.dtype,
        judgement_attn_implementation=args.attn_implementation,
    )
    loaded_judge = load_query_judge(config)
    processed = 0
    skipped = 0
    missing = 0
    for mode in modes:
        query_path = input_dir / MODE_DIRS[mode] / "queries.jsonl"
        if not query_path.exists():
            raise FileNotFoundError(f"Query file does not exist: {query_path}")
        rows = read_jsonl(query_path)
        updated = []
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("query_judgement") is not None and not args.overwrite:
                skipped += 1
                updated.append(row)
                continue
            if args.limit is not None and processed >= args.limit:
                updated.append(row)
                continue
            paper_id = str(metadata.get("paper_id") or (row.get("relevant_ids") or [""])[0])
            paper = papers.get(paper_id)
            if paper is None:
                missing += 1
                updated.append(row)
                continue
            selected = components_from_metadata(metadata)
            judgement = judge_query(
                mode,
                paper,
                selected,
                str(row.get("query") or ""),
                config,
                loaded_judge,
            )
            row = dict(row)
            metadata = dict(metadata)
            metadata["query_judgement"] = judgement
            row["metadata"] = metadata
            updated.append(row)
            processed += 1
        write_jsonl_atomic(query_path, updated)
    print(json.dumps({"processed": processed, "skipped": skipped, "missing_papers": missing}, indent=2, sort_keys=True))
    return 0


def components_from_metadata(metadata: dict[str, Any]) -> list[MemoryComponent]:
    components = []
    raw_components = metadata.get("selected_components")
    if not isinstance(raw_components, list):
        return components
    for raw in raw_components:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("kind")
        if kind not in ("visual", "textual"):
            continue
        components.append(
            MemoryComponent(
                record_id=str(raw.get("record_id") or ""),
                kind=kind,
                text=str(raw.get("text") or ""),
            )
        )
    return components


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
