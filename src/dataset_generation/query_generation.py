"""Generate tip-of-the-tongue query collections from interpretation sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from dataset_generation.component_parsing import split_component_text
from dataset_generation.document_splits import filter_papers_by_split
from dataset_generation.managed_settings import (
    DEFAULT_SETTINGS_PATH,
    ManagedSet,
    load_managed_settings,
    resolve_dataset_path,
    resolve_optional_dataset_path,
    resolve_path,
    section,
)
from dataset_generation.preprocessed import (
    read_clue_rows,
    read_preprocessed_markdown,
    read_preprocessed_papers,
    textual_clue_path,
    visual_clue_path,
)
from dataset_generation.synthetic import TestCollectionExample, write_test_collection
from dataset_generation.textual_clue_descriptions import (
    generate_with_mlx,
    generate_with_transformers,
    load_mlx_model,
    load_transformers_model,
)


QueryMode = Literal["visual-only", "visual-and-text"]
DEFAULT_PROMPT = Path("prompts/query_generation.v1.txt")
DEFAULT_JUDGEMENT_PROMPT = Path("prompts/query_judgement.v1.txt")
DEFAULT_VISUAL_INTERPRETATIONS = "qwen3_vl_figure_description"
DEFAULT_TEXTUAL_INTERPRETATIONS = "qwen3_textual_clue_description"
DEFAULT_VISUAL_COMPONENT_BUDGET = 3
DEFAULT_TEXTUAL_COMPONENT_BUDGET = 3
DEFAULT_VISUAL_QUERY_NAME = "query_generation"
DEFAULT_MODELS = {
    "null": "null",
    "mlx": "Qwen/Qwen3-1.7B-MLX-8bit",
    "transformers": "Qwen/Qwen3-4B",
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
    modes: tuple[QueryMode, ...] = ("visual-only", "visual-and-text")
    seed: int = 13
    visual_component_budget: int = DEFAULT_VISUAL_COMPONENT_BUDGET
    textual_component_budget: int = DEFAULT_TEXTUAL_COMPONENT_BUDGET
    max_examples: int | None = None
    allow_partial_components: bool = False
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


JudgementGenerator = Callable[[dict[str, Any], list[Path], str, int, float], str]


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
    return parser


def run(args: argparse.Namespace) -> Path:
    config = load_query_generation_config(args)
    return generate_query_collections(config)


def load_query_generation_config(args: argparse.Namespace) -> QueryGenerationConfig:
    config = _config_from_yaml(args.settings, query_set=args.set)
    _validate_config(config)
    return config


def generate_query_collections(config: QueryGenerationConfig) -> Path:
    prompt_text = config.prompt.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    loaded_generator = load_query_generator(config)
    loaded_judge = load_query_judge(config)
    return _generate_query_collections_from_preprocessed(config, prompt_text, prompt_sha256, loaded_generator, loaded_judge)





def _generate_query_collections_from_preprocessed(
    config: QueryGenerationConfig,
    prompt_text: str,
    prompt_sha256: str,
    loaded_generator: tuple[dict[str, Any], Callable[..., str]] | None,
    loaded_judge: tuple[dict[str, Any], JudgementGenerator] | None,
) -> Path:
    papers = read_preprocessed_papers(config.dataset)
    components_by_paper = _components_by_paper(
        config.dataset,
        clues_dir=config.clues_dir,
        visual_clues_path=config.visual_interpretations,
        textual_clues_path=config.textual_interpretations,
    )
    selected_papers = _eligible_papers(papers, config)
    root = config.output_dir / config.collection_id
    root.mkdir(parents=True, exist_ok=True)
    root_query_path = config.clues_dir / "queries.jsonl"
    query_keys = _read_existing_query_keys(root_query_path)
    for mode in config.modes:
        collection_dir = root / mode.replace("-", "_")
        collection_dir.mkdir(parents=True, exist_ok=True)
        collection_query_path = collection_dir / "queries.jsonl"
        if not config.resume:
            collection_query_path.unlink(missing_ok=True)
        existing = _read_existing_query_state(collection_query_path, mode) if config.resume else None
        collection_query_keys = _read_existing_query_keys(collection_query_path)
        examples = _generate_mode_examples_from_papers(
            mode,
            selected_papers,
            components_by_paper,
            config,
            prompt_text,
            loaded_generator,
            loaded_judge,
            existing=existing,
            query_keys=query_keys,
            root_query_path=root_query_path,
            collection_query_keys=collection_query_keys,
            collection_query_path=collection_query_path,
        )
        collection = (existing.examples if existing is not None else []) + examples
        write_test_collection(collection, collection_dir)
        _write_collection_metadata(collection_dir, mode, config, prompt_sha256, len(collection))
    _write_root_metadata(root, config, prompt_sha256)
    return root


def _generate_mode_examples_from_papers(
    mode: QueryMode,
    papers: list[tuple[int, dict[str, Any]]],
    components_by_paper: dict[str, list[MemoryComponent]],
    config: QueryGenerationConfig,
    prompt_template: str,
    loaded_generator: tuple[dict[str, Any], Callable[..., str]] | None,
    loaded_judge: tuple[dict[str, Any], JudgementGenerator] | None,
    *,
    existing: ExistingQueryState | None = None,
    query_keys: set[str] | None = None,
    root_query_path: Path | None = None,
    collection_query_keys: set[str] | None = None,
    collection_query_path: Path | None = None,
) -> list[TestCollectionExample]:
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
        if not _has_complete_clue_coverage(paper, components):
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
        query = generate_query_text(selected, config, prompt_template, loaded_generator)
        visual_count = sum(component.kind == "visual" for component in selected)
        text_count = sum(component.kind == "textual" for component in selected)
        judgement = judge_query(mode, paper, selected, query, config, loaded_judge)
        metadata = {
            "mode": mode,
            "method": "preprocessed_acl_paper_query_generation",
            "model_provider": config.model_provider,
            "model": config.model,
            "seed": config.seed,
            "paper_id": paper_id,
            "split": config.split_name,
            "split_index": str(config.split_index) if config.split_index is not None else None,
            "split_paper_index": paper_index if config.split_index is not None else None,
            "selected_component_count": len(selected),
            "selected_visual_count": visual_count,
            "selected_text_count": text_count,
            "component_selection": "all_available",
            "visual_component_budget": config.visual_component_budget,
            "textual_component_budget": config.textual_component_budget,
            "prompt_id": config.prompt_id,
            "prompt_version": config.prompt_version,
            "selected_components": [asdict(component) for component in selected],
        }
        if judgement is not None:
            metadata["query_judgement"] = judgement
        example = TestCollectionExample(
            query_id=query_id,
            query=query,
            relevant_ids=[paper_id],
            metadata=metadata,
        )
        examples.append(example)
        if collection_query_path is not None:
            _append_query(example, collection_query_path, collection_query_keys)
        if root_query_path is not None:
            _append_query(example, root_query_path, query_keys)
        progress.update(scanned=scanned, skipped=empty + underfilled + incomplete_clues + resumed)
        if target_new_examples is not None and len(examples) >= target_new_examples:
            break
    progress.close()
    final_count = existing_count + len(examples)
    if config.max_examples is not None and final_count < config.max_examples:
        raise RuntimeError(
            f"Only have {final_count} {mode} queries, but {config.max_examples} were requested. "
            f"Generated {len(examples)} new queries and resumed {existing_count} existing queries. "
            f"Scanned {scanned} papers; {incomplete_clues} were missing one or more image/textual clue files, "
            f"{underfilled} were missing required modalities, "
            f"{empty} had no usable {mode} components, and {resumed} were already generated. "
            f"Regenerate clues or pass --allow-partial-components."
        )
    return examples


def _read_existing_query_state(path: Path, mode: QueryMode) -> ExistingQueryState:
    examples: list[TestCollectionExample] = []
    paper_ids: set[str] = set()
    query_ids: set[str] = set()
    if not path.exists():
        return ExistingQueryState(examples=examples, paper_ids=paper_ids, query_ids=query_ids)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid query JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Query row must be a mapping at {path}:{line_number}")
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
    if not path.exists():
        return keys
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid query JSONL at {path}:{line_number}") from exc
        if isinstance(row, dict):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    if config.split_index is not None and config.split_name:
        prefix = f"{config.split_name}_{prefix}"
    return f"{prefix}_q{paper_index:05d}"


def _components_by_paper(
    dataset: Path,
    *,
    clues_dir: Path,
    visual_clues_path: Path | None,
    textual_clues_path: Path | None,
) -> dict[str, list[MemoryComponent]]:
    papers = read_preprocessed_papers(dataset)
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
            for paper_id in sorted(paper_ids)
            for clue in read_clue_rows(textual_clue_path(clues_dir, paper_id))
        ]
    if visual_clues_path is not None and visual_clues_path.is_file():
        visual_rows = read_clue_rows(visual_clues_path)
    else:
        visual_rows = [
            clue
            for paper_id, figure_ids in sorted(figure_ids_by_paper.items())
            for figure_id in sorted(figure_ids)
            for clue in read_clue_rows(visual_clue_path(clues_dir, paper_id, figure_id))
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


def _has_complete_clue_coverage(paper: dict[str, Any], components: list[MemoryComponent]) -> bool:
    textual_present = any(component.kind == "textual" for component in components)
    if not textual_present:
        return False
    described_figures = {component.record_id for component in components if component.kind == "visual"}
    expected_figures = {str(figure["figure_id"]) for figure in paper.get("figures", [])}
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
            return
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


def load_query_generator(config: QueryGenerationConfig) -> tuple[dict[str, Any], Callable[..., str]] | None:
    if config.model_provider == "null":
        return None
    print(f"Loading query backend: {config.model_provider}")
    print(f"Loading query model: {config.model}")
    if config.model_provider == "mlx":
        return load_mlx_model(config.model), generate_with_mlx
    if config.model_provider == "transformers":
        return (
            load_transformers_model(
                model_name=config.model,
                device_map=config.device_map,
                dtype=config.dtype,
                attn_implementation=config.attn_implementation,
            ),
            generate_with_transformers,
        )
    raise ValueError(f"Unsupported model provider: {config.model_provider}")


def load_query_judge(config: QueryGenerationConfig) -> tuple[dict[str, Any], JudgementGenerator] | None:
    if not config.judge_queries:
        return None
    print(f"Loading query judgement backend: {config.judgement_model_provider}")
    print(f"Loading query judgement model: {config.judgement_model}")
    if config.judgement_model_provider != "transformers":
        raise ValueError(f"Unsupported judgement model provider: {config.judgement_model_provider}")
    return (
        load_multimodal_transformers_model(
            model_name=config.judgement_model,
            device_map=config.judgement_device_map,
            dtype=config.judgement_dtype,
            attn_implementation=config.judgement_attn_implementation,
        ),
        generate_multimodal_judgement_with_transformers,
    )


def load_multimodal_transformers_model(
    model_name: str,
    device_map: str,
    dtype: str,
    attn_implementation: str | None,
) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

    kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
    except (OSError, ValueError):
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return {"model": model, "processor": processor}


def generate_multimodal_judgement_with_transformers(
    loaded: dict[str, Any],
    image_paths: list[Path],
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from PIL import Image

    model = loaded["model"]
    processor = loaded["processor"]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False
    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def generate_query_text(
    selected: list[MemoryComponent],
    config: QueryGenerationConfig,
    prompt_template: str,
    loaded_generator: tuple[dict[str, Any], Callable[..., str]] | None,
) -> str:
    if loaded_generator is None:
        return _render_null_query(selected)
    loaded, generate = loaded_generator
    prompt = format_query_prompt(prompt_template, selected)
    return generate(
        loaded=loaded,
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        thinking=config.thinking,
    ).strip()


def judge_query(
    mode: QueryMode,
    paper: dict[str, Any],
    selected: list[MemoryComponent],
    query: str,
    config: QueryGenerationConfig,
    loaded_judge: tuple[dict[str, Any], JudgementGenerator] | None,
) -> dict[str, Any] | None:
    if loaded_judge is None:
        return None
    prompt_template = config.judgement_prompt.read_text(encoding="utf-8")
    prompt = format_judgement_prompt(prompt_template, mode, paper, selected, query)
    image_paths = paper_image_paths(paper)
    loaded, generate = loaded_judge
    raw_output = generate(
        loaded=loaded,
        image_paths=image_paths,
        prompt=prompt,
        max_tokens=config.judgement_max_tokens,
        temperature=config.judgement_temperature,
    ).strip()
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


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def format_query_prompt(prompt_template: str, selected: list[MemoryComponent]) -> str:
    selected_cues = "\n".join(f"- {component.text}" for component in selected)
    return prompt_template.replace("{selected_cues}", selected_cues)


def _render_null_query(selected: list[MemoryComponent]) -> str:
    memories = " ".join(component.text.rstrip(".") for component in selected)
    return f"I'm trying to find a paper I read before. I remember that {memories}."


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
    modes = tuple(selection.get("modes", ["visual-only", "visual-and-text"]))
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
        max_examples=selected_set.get("max_examples", selection.get("max_examples")),
        allow_partial_components=bool(selection.get("allow_partial_components", False)),
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
    if config.judgement_model_provider != "transformers":
        raise ValueError("The query-judgement model provider must be 'transformers'.")
    if not config.judgement_model:
        raise ValueError("judgement_model must be set")
    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if config.judgement_max_tokens < 1:
        raise ValueError("judgement_max_tokens must be at least 1")
    if config.max_examples is not None and config.max_examples < 1:
        raise ValueError("max_examples must be at least 1 when set")
    if config.start_index < 0:
        raise ValueError("start_index must be non-negative")
    if config.end_index is not None and config.end_index < config.start_index:
        raise ValueError("end_index must be greater than or equal to start_index")
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
