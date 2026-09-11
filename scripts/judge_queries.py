#!/usr/bin/env python
"""Judge groundedness for existing generated query collections."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from dataset_generation.document_splits import filter_papers_by_split
from dataset_generation.generation_utils import progress
from dataset_generation.jsonl import append_jsonl_object, read_jsonl_objects
from dataset_generation.managed_settings import DEFAULT_SETTINGS_PATH, load_managed_settings, section
from dataset_generation.preprocessed import DEFAULT_DATA_DIR, read_preprocessed_papers
from dataset_generation.query_generation import (
    DEFAULT_JUDGEMENT_MODEL,
    DEFAULT_JUDGEMENT_PROMPT,
    MemoryComponent,
    QueryGenerationConfig,
    _config_from_yaml,
    judge_query,
    load_query_judge,
)


MODE_DIRS = {
    "visual-only": "visual_only",
    "visual-and-text": "visual_and_text",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge groundedness for existing query JSONL files.")
    parser.add_argument("--settings", type=Path, default=None, help="Managed settings YAML for standard judgement runs.")
    parser.add_argument("--set", choices=["train", "test"], default="train", help="Managed query set to judge.")
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
    if args.settings is not None:
        args, config = load_managed_judgement_args(args)
    else:
        config = None
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
    if config is None:
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
        judgement_path = query_path.with_name("query_judgements.jsonl")
        if args.overwrite:
            judgement_path.unlink(missing_ok=True)
        completed_query_ids = read_completed_query_ids(judgement_path)
        rows = read_jsonl_objects(query_path)
        indexed_rows = list(enumerate(rows))
        for _, row in progress(
            indexed_rows,
            total=len(indexed_rows),
            enabled=True,
            description=f"Judging {mode} queries",
            unit="query",
        ):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            query_id = str(row.get("query_id") or "")
            if query_id in completed_query_ids and not args.overwrite:
                skipped += 1
                continue
            if args.limit is not None and processed >= args.limit:
                continue
            paper_id = str(metadata.get("paper_id") or (row.get("relevant_ids") or [""])[0])
            paper = papers.get(paper_id)
            if paper is None:
                missing += 1
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
            append_judgement_row(
                judgement_path,
                {
                    "query_id": query_id,
                    "mode": mode,
                    "paper_id": paper_id,
                    "query": str(row.get("query") or ""),
                    "relevant_ids": row.get("relevant_ids") or [],
                    "judgement": judgement,
                },
            )
            completed_query_ids.add(query_id)
            processed += 1
    print(json.dumps({"processed": processed, "skipped": skipped, "missing_papers": missing}, indent=2, sort_keys=True))
    return 0


def load_managed_judgement_args(args: argparse.Namespace) -> tuple[argparse.Namespace, QueryGenerationConfig]:
    settings_path = args.settings or DEFAULT_SETTINGS_PATH
    raw, _ = load_managed_settings(settings_path)
    query_config = replace(_config_from_yaml(settings_path, query_set=args.set), judge_queries=True)

    visual_query = section(raw, "visual_query")
    judgement = section(visual_query, "judgement")
    run = judgement.get("run") if isinstance(judgement.get("run"), dict) else {}

    managed = argparse.Namespace(**vars(args))
    managed.data_dir = query_config.output_dir.parent
    managed.dataset = query_config.dataset
    managed.split_index = query_config.split_index
    managed.split = query_config.split_name or args.set
    managed.input_dir = query_config.output_dir / query_config.collection_id
    managed.collection_id = query_config.collection_id
    managed.mode = list(run.get("modes", query_config.modes))
    managed.prompt = query_config.judgement_prompt
    managed.prompt_id = query_config.judgement_prompt_id
    managed.prompt_version = query_config.judgement_prompt_version
    managed.model = query_config.judgement_model
    managed.temperature = query_config.judgement_temperature
    managed.max_tokens = query_config.judgement_max_tokens
    managed.device_map = query_config.judgement_device_map
    managed.dtype = query_config.judgement_dtype
    managed.attn_implementation = query_config.judgement_attn_implementation
    managed.limit = run.get("limit")
    managed.overwrite = bool(run.get("overwrite", False))
    return managed, query_config


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


def read_completed_query_ids(path: Path) -> set[str]:
    completed = set()
    for row in read_jsonl_objects(path, missing_ok=True):
        query_id = row.get("query_id")
        if query_id:
            completed.add(str(query_id))
    return completed


def append_judgement_row(path: Path, row: dict[str, Any]) -> None:
    append_jsonl_object(path, row, sort_keys=True, fsync=True)


if __name__ == "__main__":
    raise SystemExit(main())
