#!/usr/bin/env python
"""Judge groundedness for existing generated query collections."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from document_splits.document_splits import filter_papers_by_split
from common.generation_utils import progress, batched
from inference import GenerationRequest
from inference.base import ordered_results
from common.jsonl import append_jsonl_object, read_jsonl_objects
from common.managed_settings import DEFAULT_SETTINGS_PATH, load_managed_settings, section
from preprocessing.preprocessed import DEFAULT_DATA_DIR, clue_domain_dir, read_preprocessed_papers
from queries.query_generation import (
    DEFAULT_JUDGEMENT_MODEL,
    DEFAULT_JUDGEMENT_PROMPT,
    MemoryComponent,
    QueryGenerationConfig,
    _config_from_yaml,
    judge_query,
    load_query_judge,
    format_judgement_prompt,
    paper_image_paths,
    parse_judgement_output,
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
    parser.add_argument("--provider", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    started = time.monotonic()
    args = args or build_parser().parse_args()
    if args.settings is not None:
        args, config = load_managed_judgement_args(args)
    else:
        config = None
    dataset = args.dataset or (args.data_dir / "preprocessed")
    input_dir = args.input_dir or (args.data_dir / "query_collections")
    modes = args.mode or ["visual-only", "visual-and-text"]
    papers = {
        str(paper["paper_id"]): paper
        for paper in filter_papers_by_split(
            read_preprocessed_papers(dataset),
            split_index_path=args.split_index,
            split_name=args.split,
        )
    }
    if config is not None or args.input_dir is None:
        domains = {
            clue_domain_dir(args.data_dir / "clues", paper_id, paper.get("source_paper_dataset")).name
            for paper_id, paper in papers.items()
        }
        collection_id = config.collection_id if config is not None else (args.collection_id or f"query_generation_{args.split}")
        input_dirs = [input_dir / domain / collection_id for domain in sorted(domains)]
    else:
        input_dirs = [input_dir]
    if config is None:
        config = QueryGenerationConfig(
            dataset=dataset,
            visual_interpretations=None,
            textual_interpretations=None,
            clues_dir=args.data_dir / "clues",
            output_dir=input_dir if args.input_dir is None else input_dir.parent,
            judge_queries=True,
            judgement_prompt=args.prompt,
            judgement_prompt_id=args.prompt_id,
            judgement_prompt_version=args.prompt_version,
            judgement_model=args.model,
            judgement_model_provider=args.provider,
            judgement_batch_size=args.batch_size,
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
    failed = 0
    found_queries = False
    for collection_dir, mode in ((directory, mode) for directory in input_dirs for mode in modes):
        query_path = collection_dir / MODE_DIRS[mode] / "queries.jsonl"
        if not query_path.exists():
            if config is not None or args.input_dir is None:
                continue
            raise FileNotFoundError(f"Query file does not exist: {query_path}")
        found_queries = True
        judgement_path = query_path.with_name("query_judgements.jsonl")
        if args.overwrite:
            judgement_path.unlink(missing_ok=True)
        completed_query_ids = read_completed_query_ids(judgement_path)
        rows = read_jsonl_objects(query_path)
        indexed_rows = list(enumerate(rows))
        pending = []
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
            if args.limit is not None and processed + len(pending) >= args.limit:
                continue
            paper_id = str(metadata.get("paper_id") or (row.get("relevant_ids") or [""])[0])
            paper = papers.get(paper_id)
            if paper is None:
                missing += 1
                continue
            pending.append((row, query_id, paper_id, paper, components_from_metadata(metadata)))
        for batch in batched(pending, config.judgement_batch_size):
            if loaded_judge is not None:
                template = config.judgement_prompt.read_text(encoding="utf-8")
                requests = [GenerationRequest(query_id,
                    format_judgement_prompt(template, mode, paper, selected, str(row.get("query") or "")),
                    images=tuple(paper_image_paths(paper)), max_tokens=config.judgement_max_tokens,
                    temperature=config.judgement_temperature)
                    for row, query_id, _, paper, selected in batch]
                results = ordered_results(requests, loaded_judge.generate(requests))
                judgements = [parse_judgement_output(result.text.strip(), config) if result.error is None
                              else {"error": str(result.error)}
                              for result in results]
            else:
                judgements = [judge_query(mode, paper, selected, str(row.get("query") or ""), config, loaded_judge)
                              for row, _, _, paper, selected in batch]
            for (row, query_id, paper_id, _, _), judgement in zip(batch, judgements, strict=True):
                if isinstance(judgement, dict) and "error" in judgement:
                    failed += 1
                    print(f"Judgement failed for {query_id}: {judgement['error']}", flush=True)
                    continue
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
    if not found_queries:
        raise FileNotFoundError(f"No domain query files found under {input_dir}")
    elapsed = time.monotonic() - started
    print(json.dumps({"processed": processed, "skipped": skipped, "failed": failed,
                      "missing_papers": missing,
                      "elapsed_seconds": round(elapsed, 3),
                      "requests_per_second": round(processed / max(elapsed, 0.001), 3)},
                     indent=2, sort_keys=True))
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
    managed.input_dir = query_config.output_dir
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
