"""Generate tip-of-the-tongue query collections from interpretation sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from clues.component_parsing import split_component_text
from document_splits.document_splits import filter_papers_by_split
from common.jsonl import append_jsonl_object, read_jsonl_objects
from common.managed_settings import (
    DEFAULT_SETTINGS_PATH,
    ManagedSet,
    load_managed_settings,
    resolve_dataset_path,
    resolve_optional_dataset_path,
    resolve_path,
    section,
)
from common.model_output import parse_json_object
from preprocessing.preprocessed import (
    clue_domain_dir,
    read_clue_rows,
    read_preprocessed_markdown,
    read_preprocessed_papers,
    textual_clue_path,
    visual_clue_path,
)
from queries.synthetic import TestCollectionExample, write_test_collection
from common.validation import validate_index_window
from common.generation_utils import batched
from inference.base import ordered_results
from inference import GenerationRequest, InferenceBackend, load_backend


QueryMode = Literal["visual-only", "visual-and-text"]
DEFAULT_PROMPT = Path("prompts/query_generation.v1.txt")
DEFAULT_JUDGEMENT_PROMPT = Path("prompts/query_judgement.v1.txt")
DEFAULT_VISUAL_INTERPRETATIONS = "qwen3_vl_figure_description"
DEFAULT_TEXTUAL_INTERPRETATIONS = "qwen3_textual_clue_description"
DEFAULT_VISUAL_COMPONENT_BUDGET = 3
DEFAULT_TEXTUAL_COMPONENT_BUDGET = 3
DEFAULT_MAX_IMAGES = 15
DEFAULT_VISUAL_QUERY_NAME = "query_generation"
DEFAULT_MODELS = {
    "null": "null",
    "transformers": "Qwen/Qwen3-4B",
    "vllm": "microsoft/phi-4",
}
DEFAULT_JUDGEMENT_MODEL = "google/gemma-3-27b-it"


@dataclass(frozen=True)
class QueryGenerationConfig:
    dataset: Path
    visual_interpretations: Path | None
    textual_interpretations: Path | None
    clues_dir: Path
    output_dir: Path
    split_index: Path | None = None
    split_name: str | None = None
    prompt: Path = DEFAULT_PROMPT
    prompt_id: str = "query_generation"
    prompt_version: str = "v1"
    model_provider: str = "null"
    model: str = "null"
    temperature: float = 0.0
    max_tokens: int = 220
    device_map: str = "auto"
    dtype: str = "bfloat16"
    attn_implementation: str | None = "sdpa"
    thinking: bool = False
    batch_size: int = 1
    runtime: dict[str, Any] = field(default_factory=dict)
    judge_queries: bool = False
    judgement_prompt: Path = DEFAULT_JUDGEMENT_PROMPT
    judgement_prompt_id: str = "query_judgement"
    judgement_prompt_version: str = "v1"
    judgement_model_provider: str = "transformers"
    judgement_model: str = DEFAULT_JUDGEMENT_MODEL
    judgement_temperature: float = 0.0
    judgement_max_tokens: int = 900
    judgement_device_map: str = "auto"
    judgement_dtype: str = "bfloat16"
    judgement_attn_implementation: str | None = "sdpa"
    judgement_batch_size: int = 1
    judgement_runtime: dict[str, Any] = field(default_factory=dict)
    modes: tuple[QueryMode, ...] = ("visual-and-text",)
    seed: int = 13
    visual_component_budget: int = DEFAULT_VISUAL_COMPONENT_BUDGET
    textual_component_budget: int = DEFAULT_TEXTUAL_COMPONENT_BUDGET
    max_images: int = DEFAULT_MAX_IMAGES
    max_examples: int | None = None
    allow_partial_components: bool = False
    allow_missing_figure_clues: bool = False
    start_index: int = 0
    end_index: int | None = None
    resume: bool = False
    collection_id: str = "query_generation_train"
    config_path: Path | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryComponent:
    record_id: str
    kind: Literal["visual", "textual"]
    text: str


@dataclass(frozen=True)
class ExistingQueryState:
    examples: list[TestCollectionExample]
    paper_ids: set[str]
    query_ids: set[str]




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate query collections from visual/textual clue sidecars.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="Managed settings YAML. All generation settings are read from this file.",
    )
    parser.add_argument(
        "--set",
        choices=["train", "test"],
        default="train",
        help="Managed query set to generate. Defaults to train.",
    )
    parser.add_argument("--limit", type=int, help="Target number of queries per mode, including existing queries when resuming.")
    return parser


def run(args: argparse.Namespace) -> Path:
    config = load_query_generation_config(args)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    return generate_query_collections(config, limit=args.limit)


def load_query_generation_config(args: argparse.Namespace) -> QueryGenerationConfig:
    config = _config_from_yaml(args.settings, query_set=args.set)
    _validate_config(config)
    return config


def generate_query_collections(config: QueryGenerationConfig, *, limit: int | None = None) -> Path:
    started = time.monotonic()
    prompt_text = config.prompt.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    loaded_generator = load_query_generator(config)
    loaded_judge = load_query_judge(config)
    output = _generate_query_collections_from_preprocessed(
        config, prompt_text, prompt_sha256, loaded_generator, loaded_judge, limit=limit
    )
    elapsed = time.monotonic() - started
    print(f"Query generation elapsed: {elapsed:.1f}s", flush=True)
    return output





def _generate_query_collections_from_preprocessed(
    config: QueryGenerationConfig,
    prompt_text: str,
    prompt_sha256: str,
    loaded_generator: InferenceBackend | None,
    loaded_judge: InferenceBackend | None,
    *,
    limit: int | None = None,
) -> Path:
    papers = [_limit_paper_images(paper, config.max_images) for paper in read_preprocessed_papers(config.dataset)]
    components_by_paper = _components_by_paper(
        config.dataset,
        clues_dir=config.clues_dir,
        visual_clues_path=config.visual_interpretations,
        textual_clues_path=config.textual_interpretations,
        max_images=config.max_images,
    )
    selected_papers = _eligible_papers(papers, config)
    papers_by_domain: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for paper_index, paper in selected_papers:
        domain = clue_domain_dir(config.clues_dir, str(paper["paper_id"]), paper.get("source_paper_dataset")).name
        papers_by_domain.setdefault(domain, []).append((paper_index, paper))
    target = config.max_examples
    if limit is not None:
        target = min(limit, target) if target is not None else limit
    totals_by_mode = {mode: 0 for mode in config.modes}
    for domain, domain_papers in sorted(papers_by_domain.items()):
        root = config.output_dir / domain / config.collection_id
        root.mkdir(parents=True, exist_ok=True)
        root_query_path = config.clues_dir / domain / "queries.jsonl"
        query_keys = _read_existing_query_keys(root_query_path)
        for mode in sorted(config.modes, key=lambda value: value != "visual-and-text"):
            collection_dir = root / mode.replace("-", "_")
            collection_dir.mkdir(parents=True, exist_ok=True)
            collection_query_path = collection_dir / "queries.jsonl"
            if not config.resume:
                collection_query_path.unlink(missing_ok=True)
            existing = _read_existing_query_state(collection_query_path, mode) if config.resume else None
            mode_config = replace(config, max_examples=max(0, target - totals_by_mode[mode])) if target is not None else config
            collection_query_keys = _read_existing_query_keys(collection_query_path)
            examples = _generate_mode_examples_from_papers(
                mode,
                domain_papers,
                components_by_paper,
                mode_config,
                prompt_text,
                loaded_generator,
                loaded_judge,
                existing=existing,
                query_keys=query_keys,
                root_query_path=root_query_path,
                collection_query_keys=collection_query_keys,
                collection_query_path=collection_query_path,
                require_target=False,
            )
            collection = (existing.examples if existing is not None else []) + examples
            totals_by_mode[mode] += len(collection)
            write_test_collection(collection, collection_dir)
            _write_collection_metadata(collection_dir, mode, config, prompt_sha256, len(collection))
        _write_root_metadata(root, config, prompt_sha256)
    if target is not None:
        for mode, total in totals_by_mode.items():
            if total < target:
                raise RuntimeError(f"Only have {total} {mode} queries across domains, but {target} were requested.")
    return config.output_dir


def _generate_mode_examples_from_papers(
    mode: QueryMode,
    papers: list[tuple[int, dict[str, Any]]],
    components_by_paper: dict[str, list[MemoryComponent]],
    config: QueryGenerationConfig,
    prompt_template: str,
    loaded_generator: InferenceBackend | None,
    loaded_judge: InferenceBackend | None,
    *,
    existing: ExistingQueryState | None = None,
    query_keys: set[str] | None = None,
    root_query_path: Path | None = None,
    collection_query_keys: set[str] | None = None,
    collection_query_path: Path | None = None,
    require_target: bool = True,
) -> list[TestCollectionExample]:
    started = time.monotonic()
    examples: list[TestCollectionExample] = []
    existing_count = len(existing.examples) if existing is not None else 0
    target_new_examples = None
    if config.max_examples is not None:
        target_new_examples = max(0, config.max_examples - existing_count)
    progress = QueryProgress(
        total=target_new_examples if target_new_examples is not None else max(0, len(papers) - existing_count),
        description=f"Generating {mode} paper queries",
    )
    scanned = 0
    underfilled = 0
    empty = 0
    incomplete_clues = 0
    resumed = 0
    failed = 0
    candidates = []
    for paper_index, paper in papers:
        if target_new_examples == 0:
            break
        scanned += 1
        paper_id = str(paper["paper_id"])
        query_id = _query_id(mode, paper_index, config)
        if existing is not None and (paper_id in existing.paper_ids or query_id in existing.query_ids):
            resumed += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled + resumed)
            continue
        components = components_by_paper.get(paper_id, [])
        if not _has_complete_clue_coverage(paper, components, config.allow_missing_figure_clues):
            incomplete_clues += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled + incomplete_clues + resumed)
            continue
        selected = select_components(components, mode=mode)
        if not selected:
            empty += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled + incomplete_clues)
            continue
        if not config.allow_partial_components and not _has_required_modalities(selected, mode):
            underfilled += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled + incomplete_clues)
            continue
        candidates.append((paper_index, paper, paper_id, query_id, selected))
    for batch in batched(candidates, config.batch_size):
        if target_new_examples is not None and len(examples) >= target_new_examples:
            break
        if target_new_examples is not None:
            batch = batch[:target_new_examples - len(examples)]
        if loaded_generator is not None:
            requests = [GenerationRequest(query_id, format_query_prompt(prompt_template, selected),
                                          max_tokens=config.max_tokens, temperature=config.temperature,
                                          thinking=config.thinking)
                        for _, _, _, query_id, selected in batch]
            results = ordered_results(requests, loaded_generator.generate(requests))
            query_values = [result.error if result.error else result.text.strip() for result in results]
        else:
            query_values = [generate_query_text(selected, config, prompt_template, None)
                            for _, _, _, _, selected in batch]
        judgements_by_id = {}
        if loaded_judge is not None:
            judgement_template = config.judgement_prompt.read_text(encoding="utf-8")
            judgement_requests = [GenerationRequest(
                query_id,
                format_judgement_prompt(judgement_template, mode, paper, selected, query),
                images=tuple(paper_image_paths(paper)), max_tokens=config.judgement_max_tokens,
                temperature=config.judgement_temperature,
            ) for (_, paper, _, query_id, selected), query in zip(batch, query_values, strict=True)
                if not isinstance(query, Exception)]
            if judgement_requests:
                for result in ordered_results(judgement_requests, loaded_judge.generate(judgement_requests)):
                    judgements_by_id[result.request_id] = (
                        parse_judgement_output(result.text.strip(), config) if result.error is None else
                        {"error": str(result.error), "model_provider": config.judgement_model_provider,
                         "model": config.judgement_model, "prompt_id": config.judgement_prompt_id,
                         "prompt_version": config.judgement_prompt_version}
                    )
        for (paper_index, paper, paper_id, query_id, selected), query in zip(batch, query_values, strict=True):
            if isinstance(query, Exception):
                failed += 1
                print(f"Query generation failed for {query_id}: {query}", flush=True)
                continue
            generation_prompt = format_query_prompt(prompt_template, selected)
            visual_count = sum(component.kind == "visual" for component in selected)
            text_count = sum(component.kind == "textual" for component in selected)
            judgement = judgements_by_id.get(query_id)
            metadata = {
                "mode": mode,
                "method": "preprocessed_acl_paper_query_generation",
                "model_provider": config.model_provider,
                "model": config.model,
                "seed": config.seed,
                "paper_id": paper_id,
                "source_paper_dataset": paper.get("source_paper_dataset"),
                "split": config.split_name,
                "split_index": str(config.split_index) if config.split_index is not None else None,
                "split_paper_index": paper_index if config.split_name is not None else None,
                "selected_component_count": len(selected),
                "selected_visual_count": visual_count,
                "selected_text_count": text_count,
                "component_selection": "all_available",
                "visual_component_budget": config.visual_component_budget,
                "textual_component_budget": config.textual_component_budget,
                "max_images": config.max_images,
                "prompt_id": config.prompt_id,
                "prompt_version": config.prompt_version,
                "prompt": generation_prompt,
                "selected_components": [asdict(component) for component in selected],
            }
            if judgement is not None:
                metadata["query_judgement"] = judgement
            example = TestCollectionExample(query_id=query_id, query=query,
                                            relevant_ids=[paper_id], metadata=metadata)
            examples.append(example)
            if collection_query_path is not None:
                _append_query(example, collection_query_path, collection_query_keys)
            if root_query_path is not None:
                _append_query(example, root_query_path, query_keys)
            progress.update(scanned=scanned, skipped=empty + underfilled + incomplete_clues + resumed)
    progress.close()
    elapsed = time.monotonic() - started
    print(f"{mode}: completed={len(examples)} skipped={empty + underfilled + incomplete_clues + resumed} "
          f"(resumed={resumed}, incomplete_clues={incomplete_clues}, "
          f"missing_modalities={underfilled}, empty_components={empty}) "
          f"failed={failed} elapsed={elapsed:.1f}s requests_per_second={len(examples) / max(elapsed, 0.001):.2f}",
          flush=True)
    final_count = existing_count + len(examples)
    if require_target and config.max_examples is not None and final_count < config.max_examples:
        raise RuntimeError(
            f"Only have {final_count} {mode} queries, but {config.max_examples} were requested. "
            f"Generated {len(examples)} new queries and resumed {existing_count} existing queries. "
            f"Scanned {scanned} papers; {incomplete_clues} lacked required clue coverage, "
            f"{underfilled} were missing required modalities, "
            f"{empty} had no usable {mode} components, and {resumed} were already generated. "
            f"Regenerate clues or adjust visual_query.selection.allow_missing_figure_clues."
        )
    return examples


def _read_existing_query_state(path: Path, mode: QueryMode) -> ExistingQueryState:
    examples: list[TestCollectionExample] = []
    paper_ids: set[str] = set()
    query_ids: set[str] = set()
    for row in read_jsonl_objects(path, missing_ok=True):
        metadata = row.get("metadata", {})
        row_mode = metadata.get("mode") if isinstance(metadata, dict) else None
        if row_mode is not None and row_mode != mode:
            continue
        query_id = str(row.get("query_id", ""))
        query = str(row.get("query", ""))
        relevant_ids = [str(value) for value in row.get("relevant_ids", [])]
        negative_ids = [str(value) for value in row.get("negative_ids", [])]
        metadata = metadata if isinstance(metadata, dict) else {}
        examples.append(
            TestCollectionExample(
                query_id=query_id,
                query=query,
                relevant_ids=relevant_ids,
                negative_ids=negative_ids,
                metadata=metadata,
            )
        )
        if query_id:
            query_ids.add(query_id)
        paper_id = _query_paper_id(row)
        if paper_id is not None:
            paper_ids.add(paper_id)
    return ExistingQueryState(examples=examples, paper_ids=paper_ids, query_ids=query_ids)


def _read_existing_query_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in read_jsonl_objects(path, missing_ok=True):
        keys.update(_query_identity_keys(row))
    return keys


def _append_query(
    example: TestCollectionExample,
    path: Path,
    existing_keys: set[str] | None,
) -> None:
    row = asdict(example)
    if existing_keys is not None:
        keys = _query_identity_keys(row)
        if keys & existing_keys:
            return
        existing_keys.update(keys)
    append_jsonl_object(path, row, sort_keys=False, fsync=True)


def _query_identity_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    query_id = row.get("query_id")
    if query_id:
        keys.add(f"query_id:{query_id}")
    metadata = row.get("metadata", {})
    mode = metadata.get("mode") if isinstance(metadata, dict) else None
    paper_id = _query_paper_id(row)
    if mode and paper_id:
        keys.add(f"mode-paper:{mode}:{paper_id}")
    return keys


def _query_paper_id(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("paper_id"):
        return str(metadata["paper_id"])
    relevant_ids = row.get("relevant_ids")
    if isinstance(relevant_ids, list) and relevant_ids:
        return str(relevant_ids[0])
    return None


def _eligible_papers(papers: list[dict[str, Any]], config: QueryGenerationConfig) -> list[tuple[int, dict[str, Any]]]:
    papers = filter_papers_by_split(
        papers,
        split_index_path=config.split_index,
        split_name=config.split_name,
    )
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, paper in enumerate(papers):
        if index < config.start_index:
            continue
        if config.end_index is not None and index >= config.end_index:
            break
        selected.append((index, paper))
    return selected


def _query_id(mode: QueryMode, paper_index: int, config: QueryGenerationConfig) -> str:
    prefix = mode.replace("-", "_")
    if config.split_name:
        prefix = f"{config.split_name}_{prefix}"
    return f"{prefix}_q{paper_index:05d}"


def _limit_paper_images(paper: dict[str, Any], max_images: int) -> dict[str, Any]:
    return {**paper, "figures": paper.get("figures", [])[:max_images]}


def _components_by_paper(
    dataset: Path,
    *,
    clues_dir: Path,
    visual_clues_path: Path | None,
    textual_clues_path: Path | None,
    max_images: int | None = None,
) -> dict[str, list[MemoryComponent]]:
    papers = read_preprocessed_papers(dataset)
    if max_images is not None:
        papers = [_limit_paper_images(paper, max_images) for paper in papers]
    paper_ids = set()
    figure_ids_by_paper: dict[str, set[str]] = {}
    figure_names: dict[tuple[str, str], str] = {}
    for paper in papers:
        paper_id = str(paper["paper_id"])
        paper_ids.add(paper_id)
        for index, figure in enumerate(paper.get("figures", [])):
            fid = str(figure["figure_id"])
            figure_ids_by_paper.setdefault(paper_id, set()).add(fid)
            filename = str(figure.get("filename", ""))
            match = re.search(r"Figure\s*(\d+[a-zA-Z]?)", filename, re.IGNORECASE)
            if match:
                figure_names[(paper_id, fid)] = f"Figure {match.group(1)}"
            else:
                figure_names[(paper_id, fid)] = f"Figure {index + 1}"
    
    components: dict[str, list[MemoryComponent]] = {}
    if textual_clues_path is not None and textual_clues_path.is_file():
        textual_rows = read_clue_rows(textual_clues_path)
    else:
        textual_rows = [
            clue
            for paper in papers
            for clue in read_clue_rows(textual_clue_path(clues_dir, str(paper["paper_id"]), paper.get("source_paper_dataset")))
        ]
    if visual_clues_path is not None and visual_clues_path.is_file():
        visual_rows = read_clue_rows(visual_clues_path)
    else:
        visual_rows = [
            clue
            for paper in papers
            for figure in paper.get("figures", [])
            for clue in read_clue_rows(
                visual_clue_path(
                    clues_dir,
                    str(paper["paper_id"]),
                    str(figure["figure_id"]),
                    paper.get("source_paper_dataset"),
                )
            )
        ]
    for clue in textual_rows:
        if clue.get("kind") != "textual":
            continue
        paper_id = str(clue.get("paper_id", ""))
        if paper_id in paper_ids:
            text = str(clue.get("output", "")).strip()
            for component_text in split_component_text(text, "textual"):
                components.setdefault(paper_id, []).append(
                    MemoryComponent(record_id=paper_id, kind="textual", text=component_text)
                )
    for clue in visual_rows:
        if clue.get("kind") != "visual":
            continue
        figure_id = str(clue.get("figure_id", ""))
        paper_id = str(clue.get("paper_id", ""))
        if paper_id in paper_ids and figure_id in figure_ids_by_paper.get(paper_id, set()):
            text = str(clue.get("output", "")).strip()
            figure_name = figure_names.get((paper_id, figure_id), "Figure")
            for component_text in split_component_text(text, "visual"):
                components.setdefault(paper_id, []).append(
                    MemoryComponent(record_id=figure_id, kind="visual", text=f"{figure_name}: {component_text}")
                )
    return components


def _has_required_modalities(selected: list[MemoryComponent], mode: QueryMode) -> bool:
    has_visual = any(component.kind == "visual" for component in selected)
    has_textual = any(component.kind == "textual" for component in selected)
    if mode == "visual-only":
        return has_visual
    return has_visual and has_textual


def _has_complete_clue_coverage(
    paper: dict[str, Any], components: list[MemoryComponent], allow_missing_figure_clues: bool = False,
) -> bool:
    textual_present = any(component.kind == "textual" for component in components)
    if not textual_present:
        return False
    described_figures = {component.record_id for component in components if component.kind == "visual"}
    expected_figures = {str(figure["figure_id"]) for figure in paper.get("figures", [])}
    if allow_missing_figure_clues:
        return bool(expected_figures & described_figures)
    return bool(expected_figures) and expected_figures <= described_figures


class QueryProgress:
    def __init__(self, *, total: int, description: str) -> None:
        self.total = total
        self.description = description
        self.count = 0
        self.next_report = 1
        self._bar: Any = None
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        self._bar = tqdm(total=total, desc=description, unit="query")

    def update(self, *, scanned: int, skipped: int) -> None:
        self.count += 1
        if self._bar is not None:
            self._bar.update(1)
            self.set_status(scanned=scanned, skipped=skipped)
        print(f"{self.description}: {self.count}/{self.total} queries; scanned={scanned}; skipped={skipped}", flush=True)

    def set_status(self, *, scanned: int, skipped: int) -> None:
        if self._bar is not None:
            self._bar.set_postfix({"scanned": scanned, "skipped": skipped}, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def select_components(
    components: list[MemoryComponent],
    *,
    mode: QueryMode,
) -> list[MemoryComponent]:
    visual = [component for component in components if component.kind == "visual"]
    textual = [component for component in components if component.kind == "textual"]
    if mode == "visual-only":
        return visual
    return textual + visual


def load_query_generator(config: QueryGenerationConfig) -> InferenceBackend | None:
    if config.model_provider == "null":
        return None
    print(f"Loading query backend: {config.model_provider}")
    print(f"Loading query model: {config.model}")
    runtime = config.runtime or ({} if config.model_provider == "vllm" else {"device_map": config.device_map, "dtype": config.dtype,
                                 "attn_implementation": config.attn_implementation})
    return load_backend(config.model_provider, config.model, runtime=runtime)


def load_query_judge(config: QueryGenerationConfig) -> InferenceBackend | None:
    if not config.judge_queries:
        return None
    print(f"Loading query judgement backend: {config.judgement_model_provider}")
    print(f"Loading query judgement model: {config.judgement_model}")
    runtime = config.judgement_runtime or ({} if config.judgement_model_provider == "vllm" else {"device_map": config.judgement_device_map,
                                           "dtype": config.judgement_dtype,
                                           "attn_implementation": config.judgement_attn_implementation})
    return load_backend(config.judgement_model_provider, config.judgement_model, runtime=runtime)


def generate_query_text(
    selected: list[MemoryComponent],
    config: QueryGenerationConfig,
    prompt_template: str,
    loaded_generator: InferenceBackend | None,
) -> str:
    if loaded_generator is None:
        return _render_null_query(selected)
    prompt = format_query_prompt(prompt_template, selected)
    result = loaded_generator.generate([GenerationRequest("query", prompt, max_tokens=config.max_tokens,
                                                          temperature=config.temperature, thinking=config.thinking)])[0]
    if result.error:
        raise result.error
    return result.text.strip()


def judge_query(
    mode: QueryMode,
    paper: dict[str, Any],
    selected: list[MemoryComponent],
    query: str,
    config: QueryGenerationConfig,
    loaded_judge: InferenceBackend | None,
) -> dict[str, Any] | None:
    if loaded_judge is None:
        return None
    prompt_template = config.judgement_prompt.read_text(encoding="utf-8")
    prompt = format_judgement_prompt(prompt_template, mode, paper, selected, query)
    image_paths = paper_image_paths(paper)
    result = loaded_judge.generate([GenerationRequest("judgement", prompt, images=tuple(image_paths),
                                                      max_tokens=config.judgement_max_tokens,
                                                      temperature=config.judgement_temperature)])[0]
    if result.error:
        return {"error": str(result.error), "model_provider": config.judgement_model_provider,
                "model": config.judgement_model, "prompt_id": config.judgement_prompt_id,
                "prompt_version": config.judgement_prompt_version}
    raw_output = result.text.strip()
    return parse_judgement_output(raw_output, config)


def parse_judgement_output(raw_output: str, config: QueryGenerationConfig) -> dict[str, Any]:
    parsed = parse_json_object(raw_output)
    if parsed is None:
        return {
            "parse_error": True,
            "raw_output": raw_output,
            "model_provider": config.judgement_model_provider,
            "model": config.judgement_model,
            "prompt_id": config.judgement_prompt_id,
            "prompt_version": config.judgement_prompt_version,
        }
    return {
        **parsed,
        "model_provider": config.judgement_model_provider,
        "model": config.judgement_model,
        "prompt_id": config.judgement_prompt_id,
        "prompt_version": config.judgement_prompt_version,
    }


def format_judgement_prompt(
    prompt_template: str,
    mode: QueryMode,
    paper: dict[str, Any],
    selected: list[MemoryComponent],
    query: str,
) -> str:
    selected_cues = "\n".join(f"- {component.kind}: {component.text}" for component in selected)
    captions = "\n".join(str(figure.get("caption", "")).strip() for figure in paper.get("figures", []) if figure.get("caption"))
    paper_text = read_preprocessed_markdown(paper)
    figures_text = format_figures_for_judgement(paper)
    replacements = {
        "{visual_only | visual_text}": "visual_only" if mode == "visual-only" else "visual_text",
        "{selected_cues}": selected_cues,
        "{paper_text}": paper_text,
        "{figures}": figures_text,
        "{query}": query,
        "{title}": str(paper.get("title") or ""),
        "{authors}": format_authors(paper.get("authors")),
        "{doi_arxiv}": format_identifiers(paper),
        "{caption}": captions,
    }
    prompt = prompt_template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    if "{paper_text}" not in prompt_template:
        prompt = f"{prompt}\n\nFULL PAPER TEXT PROVIDED TO JUDGE GROUNDEDNESS:\n{paper_text}"
    if "{figures}" not in prompt_template:
        prompt = f"{prompt}\n\nSOURCE FIGURES:\n{figures_text}"
    return prompt


def paper_image_paths(paper: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for figure in paper.get("figures", []):
        image_path = figure.get("image_path")
        if not image_path:
            continue
        path = Path(str(image_path))
        if path.exists():
            paths.append(path)
    return paths


def format_figures_for_judgement(paper: dict[str, Any]) -> str:
    rows = []
    for index, figure in enumerate(paper.get("figures", []), start=1):
        figure_id = str(figure.get("figure_id") or f"figure-{index}")
        filename = str(figure.get("filename") or Path(str(figure.get("image_path") or figure_id)).name)
        caption = str(figure.get("caption") or "").strip()
        image_path = str(figure.get("image_path") or "").strip()
        parts = [f"{index}. id={figure_id}", f"filename={filename}"]
        if caption:
            parts.append(f"caption={caption}")
        if image_path:
            parts.append(f"attached_image_path={image_path}")
        rows.append("; ".join(parts))
    if not rows:
        return "No extracted figure images were found."
    return "\n".join(rows)


def format_authors(authors: Any) -> str:
    if isinstance(authors, list):
        formatted = []
        for author in authors:
            if isinstance(author, dict):
                formatted.append(str(author.get("name") or " ".join(str(value) for value in author.values() if value)))
            else:
                formatted.append(str(author))
        return ", ".join(value for value in formatted if value)
    return "" if authors is None else str(authors)


def format_identifiers(paper: dict[str, Any]) -> str:
    values = []
    for key in ("doi", "arxiv", "arxiv_id", "acl_id", "anthology_id", "paper_id"):
        value = paper.get(key)
        if value:
            values.append(f"{key}: {value}")
    return "; ".join(values)


def format_query_prompt(prompt_template: str, selected: list[MemoryComponent]) -> str:
    selected_cues = _format_selected_cues_for_query(selected)
    return prompt_template.replace("{selected_cues}", selected_cues)


def _render_null_query(selected: list[MemoryComponent]) -> str:
    memories = " ".join(_display_clue_text(component.text).rstrip(".") for component in selected)
    return f"I'm trying to find a paper I read before. I remember that {memories}."


def _format_selected_cues_for_query(selected: list[MemoryComponent]) -> str:
    lines: list[str] = []
    visual_lines_by_figure: dict[str, list[str]] = {}
    figure_order: list[str] = []

    for component in selected:
        if component.kind == "visual":
            figure_name, clue_text = _split_visual_clue(component.text)
            if figure_name not in visual_lines_by_figure:
                figure_order.append(figure_name)
                visual_lines_by_figure[figure_name] = []
            visual_lines_by_figure[figure_name].append(f"- {_display_clue_text(clue_text)}")
        else:
            lines.append(f"- {_display_clue_text(component.text)}")

    for figure_name in figure_order:
        if lines:
            lines.append("")
        lines.append(figure_name)
        lines.extend(visual_lines_by_figure[figure_name])

    return "\n".join(lines)


def _split_visual_clue(text: str) -> tuple[str, str]:
    match = re.match(r"^(Figure\s+\d+[a-zA-Z]?|Figure):\s*(.+)$", text.strip())
    if match:
        return match.group(1), match.group(2)
    return "Figure", text


def _display_clue_text(text: str) -> str:
    text = text.strip()
    textual_match = re.match(r"^textual\s+[^:]+:\s*(.+)$", text, re.IGNORECASE)
    if textual_match:
        return textual_match.group(1).strip()
    generic_match = re.match(r"^[A-Za-z][A-Za-z0-9_ -]*:\s*(.+)$", text)
    if generic_match:
        return generic_match.group(1).strip()
    return text


def _config_from_yaml(path: Path, *, query_set: ManagedSet = "train") -> QueryGenerationConfig:
    raw, config_path = load_managed_settings(path)
    config_dir = config_path.parent
    if query_set not in ("train", "test"):
        raise ValueError(f"Query generation is only managed for 'train' and 'test', got {query_set!r}.")

    dataset = section(raw, "dataset")
    visual_query = section(raw, "visual_query")
    prompt = section(visual_query, "prompt")
    model = section(visual_query, "model")
    selection = section(visual_query, "selection")
    output = section(visual_query, "output")
    query_sets = section(visual_query, "query_sets")
    selected_set = section(query_sets, query_set)
    judgement = visual_query.get("judgement") if isinstance(visual_query.get("judgement"), dict) else {}
    modes = tuple(selection.get("modes", ["visual-and-text"]))
    return QueryGenerationConfig(
        dataset=resolve_dataset_path(dataset, "preprocessed", config_dir),
        visual_interpretations=resolve_optional_dataset_path(dataset, "visual_interpretations", config_dir),
        textual_interpretations=resolve_optional_dataset_path(dataset, "textual_interpretations", config_dir),
        split_index=resolve_optional_dataset_path(dataset, "split_index", config_dir),
        split_name=selected_set.get("split_name", query_set),
        clues_dir=resolve_dataset_path(dataset, "clues_dir", config_dir),
        output_dir=resolve_path(output["dir"], config_dir),
        prompt=resolve_path(prompt["template"], config_dir),
        prompt_id=prompt.get("name", "query_generation"),
        prompt_version=prompt.get("version", "v1"),
        model_provider=model.get("provider", "null"),
        model=model.get("name", "null"),
        temperature=float(model.get("temperature", 0.0)),
        max_tokens=int(model.get("max_tokens", 220)),
        device_map=model.get("device_map", "auto"),
        dtype=model.get("dtype", "bfloat16"),
        attn_implementation=model.get("attn_implementation", "sdpa"),
        thinking=bool(model.get("thinking", False)),
        batch_size=int(model.get("batch_size", 1)),
        runtime=(visual_query.get("runtime") or {}).get(model.get("provider", "null"), {}),
        judge_queries=bool(judgement.get("enabled", False)),
        judgement_prompt=resolve_path(judgement.get("template", DEFAULT_JUDGEMENT_PROMPT), config_dir),
        judgement_prompt_id=judgement.get("name", "query_judgement"),
        judgement_prompt_version=judgement.get("version", "v1"),
        judgement_model_provider=judgement.get("provider", "transformers"),
        judgement_model=judgement.get("model", DEFAULT_JUDGEMENT_MODEL),
        judgement_temperature=float(judgement.get("temperature", 0.0)),
        judgement_max_tokens=int(judgement.get("max_tokens", 900)),
        judgement_device_map=judgement.get("device_map", "auto"),
        judgement_dtype=judgement.get("dtype", "bfloat16"),
        judgement_attn_implementation=judgement.get("attn_implementation", "sdpa"),
        judgement_batch_size=int(judgement.get("batch_size", 1)),
        judgement_runtime=(judgement.get("runtime") or {}).get(judgement.get("provider", "transformers"), {}),
        modes=modes,
        seed=int(visual_query.get("seed", 13)),
        visual_component_budget=int(
            selection.get(
                "visual_component_budget",
                selection.get("component_budget", DEFAULT_VISUAL_COMPONENT_BUDGET),
            )
        ),
        textual_component_budget=int(
            selection.get(
                "textual_component_budget",
                selection.get("max_text_components", selection.get("component_budget", DEFAULT_TEXTUAL_COMPONENT_BUDGET)),
            )
        ),
        max_images=int(selection.get("max_images", DEFAULT_MAX_IMAGES)),
        max_examples=selected_set.get("max_examples", selection.get("max_examples")),
        allow_partial_components=bool(selection.get("allow_partial_components", False)),
        allow_missing_figure_clues=bool(selection.get("allow_missing_figure_clues", False)),
        start_index=int(selected_set.get("start_index", selection.get("start_index", 0))),
        end_index=selected_set.get("end_index", selection.get("end_index")),
        resume=bool(selected_set.get("resume", selection.get("resume", False))),
        collection_id=selected_set.get("collection_id", f"{visual_query.get('name', DEFAULT_VISUAL_QUERY_NAME)}_{query_set}"),
        config_path=config_path,
        extra_metadata={
            "managed_settings": {
                "dataset": dataset.get("name"),
                "visual_query": visual_query.get("name"),
                "query_set": query_set,
                "raw_settings": raw,
            }
        },
    )


def _validate_config(config: QueryGenerationConfig) -> None:
    if config.model_provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported model provider {config.model_provider!r}; expected one of {sorted(DEFAULT_MODELS)}")
    if config.model == "default":
        object.__setattr__(config, "model", DEFAULT_MODELS[config.model_provider])
    if config.model_provider == "null" and config.model != "null":
        raise ValueError("The null query-generation provider expects model to be 'null'.")
    if config.model_provider != "null" and config.model == "null":
        raise ValueError("Non-null query-generation providers require a real model name.")
    if config.judgement_model_provider not in {"transformers", "vllm"}:
        raise ValueError("Unsupported query-judgement model provider")
    if not config.judgement_model:
        raise ValueError("judgement_model must be set")
    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if config.judgement_max_tokens < 1:
        raise ValueError("judgement_max_tokens must be at least 1")
    if config.max_examples is not None and config.max_examples < 1:
        raise ValueError("max_examples must be at least 1 when set")
    if config.max_images < 1:
        raise ValueError("max_images must be at least 1")
    validate_index_window(config.start_index, config.end_index)
    if config.split_index is not None:
        if config.split_name is None:
            raise ValueError("split_name is required when split_index is set")
        if not config.split_index.exists():
            raise FileNotFoundError(f"Split index does not exist: {config.split_index}")
    for mode in config.modes:
        if mode not in ("visual-only", "visual-and-text"):
            raise ValueError(f"Unsupported query mode: {mode}")
    if not config.prompt.exists():
        raise FileNotFoundError(f"Prompt template does not exist: {config.prompt}")
    if config.judge_queries and not config.judgement_prompt.exists():
        raise FileNotFoundError(f"Judgement prompt template does not exist: {config.judgement_prompt}")


def _write_collection_metadata(
    collection_dir: Path,
    mode: QueryMode,
    config: QueryGenerationConfig,
    prompt_sha256: str,
    example_count: int,
) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "example_count": example_count,
        "config": _metadata_config(config),
        "prompt_sha256": prompt_sha256,
    }
    (collection_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_root_metadata(root: Path, config: QueryGenerationConfig, prompt_sha256: str) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_id": config.collection_id,
        "modes": list(config.modes),
        "config": _metadata_config(config),
        "prompt_sha256": prompt_sha256,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _metadata_config(config: QueryGenerationConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in (
        "dataset",
        "visual_interpretations",
        "textual_interpretations",
        "split_index",
        "clues_dir",
        "output_dir",
        "prompt",
        "judgement_prompt",
        "config_path",
    ):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data
