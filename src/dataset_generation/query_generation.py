"""Generate tip-of-the-tongue query collections from interpretation sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from dataset_generation.preprocessed import (
    DEFAULT_CLUES_DIR,
    DEFAULT_DATA_DIR,
    read_clue_rows,
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
DEFAULT_VISUAL_INTERPRETATIONS = "qwen3_vl_figure_description"
DEFAULT_TEXTUAL_INTERPRETATIONS = "qwen3_textual_clue_description"
DEFAULT_VISUAL_COMPONENT_BUDGET = 3
DEFAULT_TEXTUAL_COMPONENT_BUDGET = 3
DEFAULT_RUN_ID = "query_generation_default"
DEFAULT_MODELS = {
    "null": "null",
    "mlx": "Qwen/Qwen3-1.7B-MLX-8bit",
    "transformers": "Qwen/Qwen3-4B",
}


@dataclass(frozen=True)
class QueryGenerationConfig:
    dataset: Path
    visual_interpretations: Path | None
    textual_interpretations: Path | None
    clues_dir: Path
    output_dir: Path
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
    modes: tuple[QueryMode, ...] = ("visual-only", "visual-and-text")
    seed: int = 13
    visual_component_budget: int = DEFAULT_VISUAL_COMPONENT_BUDGET
    textual_component_budget: int = DEFAULT_TEXTUAL_COMPONENT_BUDGET
    max_examples: int | None = None
    allow_partial_components: bool = False
    start_index: int = 0
    end_index: int | None = None
    resume: bool = False
    collection_id: str = DEFAULT_RUN_ID
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
    parser.add_argument("--config", type=Path, default=None, help="YAML config file for query generation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--clues-dir", type=Path, default=None)
    parser.add_argument("--visual-interpretations", type=Path, default=None)
    parser.add_argument("--textual-interpretations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--collection-id", default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--mode", choices=["visual-only", "visual-and-text"], action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visual-component-budget", type=int, default=None)
    parser.add_argument("--textual-component-budget", type=int, default=None)
    parser.add_argument("--component-budget", type=int, default=None, help="Legacy alias: sets both visual/textual budgets.")
    parser.add_argument("--max-text-components", type=int, default=None, help="Legacy alias for --textual-component-budget.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Alias for --max-examples.")
    parser.add_argument("--allow-partial-components", action="store_true")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", default="train", help="Dataset split used when --dataset is omitted.")
    return parser


def run(args: argparse.Namespace) -> Path:
    config = load_query_generation_config(args)
    return generate_query_collections(config)


def load_query_generation_config(args: argparse.Namespace) -> QueryGenerationConfig:
    base = _config_from_yaml(args.config) if args.config else _default_config(args.data_dir, args.split)
    max_examples = args.max_examples if args.max_examples is not None else args.limit
    overrides = {
        "dataset": args.dataset,
        "clues_dir": args.clues_dir,
        "visual_interpretations": args.visual_interpretations,
        "textual_interpretations": args.textual_interpretations,
        "output_dir": args.output_dir,
        "collection_id": args.collection_id,
        "prompt": args.prompt,
        "prompt_id": args.prompt_id,
        "prompt_version": args.prompt_version,
        "model_provider": args.model_provider,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "thinking": True if args.thinking else None,
        "modes": tuple(args.mode) if args.mode else None,
        "seed": args.seed,
        "visual_component_budget": args.visual_component_budget,
        "textual_component_budget": args.textual_component_budget,
        "max_examples": max_examples,
        "allow_partial_components": True if args.allow_partial_components else None,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "resume": True if args.resume else None,
    }
    data = asdict(base)
    if args.component_budget is not None:
        data["visual_component_budget"] = args.component_budget
        data["textual_component_budget"] = args.component_budget
    if args.max_text_components is not None:
        data["textual_component_budget"] = args.max_text_components
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    config = QueryGenerationConfig(**data)
    _validate_config(config)
    return config


def generate_query_collections(config: QueryGenerationConfig) -> Path:
    prompt_text = config.prompt.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    loaded_generator = load_query_generator(config)
    return _generate_query_collections_from_preprocessed(config, prompt_text, prompt_sha256, loaded_generator)





def _generate_query_collections_from_preprocessed(
    config: QueryGenerationConfig,
    prompt_text: str,
    prompt_sha256: str,
    loaded_generator: tuple[dict[str, Any], Callable[..., str]] | None,
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
    query_keys = _read_existing_query_keys(root_query_path) if config.resume else set()
    for mode in config.modes:
        collection_dir = root / mode.replace("-", "_")
        existing = _read_existing_query_state(collection_dir / "queries.jsonl", mode) if config.resume else None
        examples = _generate_mode_examples_from_papers(
            mode,
            selected_papers,
            components_by_paper,
            config,
            prompt_text,
            loaded_generator,
            existing=existing,
            query_keys=query_keys,
            root_query_path=root_query_path,
        )
        collection = (existing.examples if existing is not None else []) + examples
        write_test_collection(collection, collection_dir)
        _write_collection_metadata(collection_dir, mode, config, prompt_sha256, len(collection))
    _write_root_metadata(root, config, prompt_sha256)
    return root


def _generate_mode_examples_from_papers(
    mode: QueryMode,
    papers: list[dict[str, Any]],
    components_by_paper: dict[str, list[MemoryComponent]],
    config: QueryGenerationConfig,
    prompt_template: str,
    loaded_generator: tuple[dict[str, Any], Callable[..., str]] | None,
    *,
    existing: ExistingQueryState | None = None,
    query_keys: set[str] | None = None,
    root_query_path: Path | None = None,
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
    resumed = 0
    for paper_index, paper in enumerate(papers):
        if target_new_examples == 0:
            break
        scanned += 1
        paper_id = str(paper["paper_id"])
        query_id = f"{mode.replace('-', '_')}_q{paper_index:05d}"
        if existing is not None and (paper_id in existing.paper_ids or query_id in existing.query_ids):
            resumed += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled + resumed)
            continue
        components = components_by_paper.get(paper_id, [])
        selected = select_components(
            components,
            mode=mode,
            seed=config.seed,
            source_record_id=paper_id,
            visual_component_budget=config.visual_component_budget,
            textual_component_budget=config.textual_component_budget,
        )
        if not selected:
            empty += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled)
            continue
        expected_count = _expected_component_count(mode, config)
        if not config.allow_partial_components and len(selected) < expected_count:
            underfilled += 1
            progress.set_status(scanned=scanned, skipped=empty + underfilled)
            continue
        query = generate_query_text(selected, config, prompt_template, loaded_generator)
        visual_count = sum(component.kind == "visual" for component in selected)
        text_count = sum(component.kind == "textual" for component in selected)
        example = TestCollectionExample(
            query_id=query_id,
            query=query,
            relevant_ids=[paper_id],
            metadata={
                "mode": mode,
                "method": "preprocessed_acl_paper_query_generation",
                "model_provider": config.model_provider,
                "model": config.model,
                "seed": config.seed,
                "paper_id": paper_id,
                "selected_component_count": len(selected),
                "selected_visual_count": visual_count,
                "selected_text_count": text_count,
                "visual_component_budget": config.visual_component_budget,
                "textual_component_budget": config.textual_component_budget,
                "prompt_id": config.prompt_id,
                "prompt_version": config.prompt_version,
                "selected_components": [asdict(component) for component in selected],
            },
        )
        examples.append(example)
        if root_query_path is not None:
            _append_query(example, root_query_path, query_keys)
        progress.update(scanned=scanned, skipped=empty + underfilled + resumed)
        if target_new_examples is not None and len(examples) >= target_new_examples:
            break
    progress.close()
    final_count = existing_count + len(examples)
    if config.max_examples is not None and final_count < config.max_examples:
        raise RuntimeError(
            f"Only have {final_count} {mode} queries, but {config.max_examples} were requested. "
            f"Generated {len(examples)} new queries and resumed {existing_count} existing queries. "
            f"Scanned {scanned} papers; {underfilled} had fewer than {_expected_component_count(mode, config)} "
            f"required components, {empty} had no usable {mode} components, and {resumed} were already generated. "
            f"Regenerate clues, lower budgets, or pass --allow-partial-components."
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
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


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


def _eligible_papers(papers: list[dict[str, Any]], config: QueryGenerationConfig) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index, paper in enumerate(papers):
        if index < config.start_index:
            continue
        if config.end_index is not None and index >= config.end_index:
            break
        selected.append(paper)
    return selected


def _components_by_paper(
    dataset: Path,
    *,
    clues_dir: Path,
    visual_clues_path: Path | None,
    textual_clues_path: Path | None,
) -> dict[str, list[MemoryComponent]]:
    papers = read_preprocessed_papers(dataset)
    paper_ids = set()
    figure_to_paper = {}
    figure_to_name = {}
    for paper in papers:
        paper_ids.add(str(paper["paper_id"]))
        for index, figure in enumerate(paper.get("figures", [])):
            fid = str(figure["figure_id"])
            figure_to_paper[fid] = str(paper["paper_id"])
            filename = str(figure.get("filename", ""))
            match = re.search(r"Figure\s*(\d+[a-zA-Z]?)", filename, re.IGNORECASE)
            if match:
                figure_to_name[fid] = f"Figure {match.group(1)}"
            else:
                figure_to_name[fid] = f"Figure {index + 1}"
    
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
            for figure_id, paper_id in sorted(figure_to_paper.items())
            for clue in read_clue_rows(visual_clue_path(clues_dir, paper_id, figure_id))
        ]
    for clue in textual_rows:
        if clue.get("kind") != "textual":
            continue
        paper_id = str(clue.get("paper_id", ""))
        if paper_id in paper_ids:
            text = str(clue.get("output", "")).strip()
            for component_text in _split_component_text(text, "textual"):
                components.setdefault(paper_id, []).append(
                    MemoryComponent(record_id=paper_id, kind="textual", text=component_text)
                )
    for clue in visual_rows:
        if clue.get("kind") != "visual":
            continue
        figure_id = str(clue.get("figure_id", ""))
        paper_id = str(clue.get("paper_id", ""))
        if paper_id in paper_ids and figure_id in figure_to_paper:
            text = str(clue.get("output", "")).strip()
            figure_name = figure_to_name.get(figure_id, "Figure")
            for component_text in _split_component_text(text, "visual"):
                components.setdefault(paper_id, []).append(
                    MemoryComponent(record_id=figure_id, kind="visual", text=f"{figure_name}: {component_text}")
                )
    return components


def _expected_component_count(mode: QueryMode, config: QueryGenerationConfig) -> int:
    if mode == "visual-only":
        return config.visual_component_budget
    return config.visual_component_budget + config.textual_component_budget


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
    seed: int,
    source_record_id: str,
    visual_component_budget: int = DEFAULT_VISUAL_COMPONENT_BUDGET,
    textual_component_budget: int = DEFAULT_TEXTUAL_COMPONENT_BUDGET,
) -> list[MemoryComponent]:
    rng = random.Random(f"{seed}:{source_record_id}:{mode}")
    visual = [component for component in components if component.kind == "visual"]
    textual = [component for component in components if component.kind == "textual"]
    if mode == "visual-only":
        selected = _sample(rng, visual, visual_component_budget)
    else:
        selected_text = _sample(rng, textual, textual_component_budget)
        selected_visual = _sample(rng, visual, visual_component_budget)
        selected = selected_text + selected_visual
    rng.shuffle(selected)
    return selected


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


def format_query_prompt(prompt_template: str, selected: list[MemoryComponent]) -> str:
    selected_cues = "\n".join(f"- {component.text}" for component in selected)
    return prompt_template.replace("{selected_cues}", selected_cues)





def _render_null_query(selected: list[MemoryComponent]) -> str:
    memories = " ".join(component.text.rstrip(".") for component in selected)
    return f"I'm trying to find a paper I read before. I remember that {memories}."


def _split_component_text(text: str, kind: Literal["visual", "textual"]) -> list[str]:
    decoded = _decode_jsonish_components(text)
    if decoded is None:
        return [text]
    if kind == "textual":
        components = [_textual_component(value) for value in decoded]
    else:
        components = []
        for value in decoded:
            components.extend(_visual_components(value))
    cleaned = [component.strip() for component in components if component and component.strip()]
    return cleaned


def _repair_json_dict(text: str) -> dict[str, Any] | None:
    closings = ["", '"', '"]}', ']}', '}', '"}']
    
    stripped = re.sub(r",\s*$", "", text.strip())
    for closing in closings:
        try:
            val = json.loads(stripped + closing)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass
            
    while "," in text:
        text = text[:text.rfind(",")]
        stripped = text.strip()
        for closing in closings:
            try:
                val = json.loads(stripped + closing)
                if isinstance(val, dict):
                    return val
            except json.JSONDecodeError:
                pass
    return None

def _decode_jsonish_components(text: str) -> list[Any] | None:
    text = text.strip()
    first_json = min((index for index in (text.find("["), text.find("{")) if index != -1), default=-1)
    if first_json > 0:
        text = text[first_json:]
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
        
    decoder = json.JSONDecoder()
    index = 0
    values: list[Any] = []
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] in ","):
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            remaining = text[index:].strip()
            dict_start = remaining.find("{")
            if dict_start != -1:
                repaired = _repair_json_dict(remaining[dict_start:])
                if repaired:
                    values.append(repaired)
            break
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
        index = end
    return values or None


def _textual_component(value: Any) -> str | None:
    if isinstance(value, dict):
        cue = value.get("cue")
        return str(cue) if cue is not None else None
    return str(value)


def _visual_components(value: Any) -> list[str]:
    if isinstance(value, dict):
        components: list[str] = []
        for field_value in value.values():
            if isinstance(field_value, list):
                components.extend(str(item) for item in field_value)
            elif field_value is not None:
                components.append(str(field_value))
        return components
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _sample(rng: random.Random, components: list[MemoryComponent], count: int) -> list[MemoryComponent]:
    if count <= 0:
        return []
    if len(components) <= count:
        copied = list(components)
        rng.shuffle(copied)
        return copied
    return rng.sample(components, count)


def _config_from_yaml(path: Path) -> QueryGenerationConfig:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config_dir = config_path.parent
    run = _section(raw, "run")
    inputs = _section(raw, "inputs")
    prompt = _section(raw, "prompt")
    model = _section(raw, "model")
    selection = _section(raw, "selection")
    output = _section(raw, "output")
    modes = tuple(selection.get("modes", ["visual-only", "visual-and-text"]))
    return QueryGenerationConfig(
        dataset=_resolve_path(inputs["dataset"], config_dir),
        visual_interpretations=_resolve_optional_path(inputs.get("visual_interpretations"), config_dir),
        textual_interpretations=_resolve_optional_path(inputs.get("textual_interpretations"), config_dir),
        clues_dir=_resolve_path(inputs.get("clues_dir", DEFAULT_CLUES_DIR), config_dir),
        output_dir=_resolve_path(output["dir"], config_dir),
        prompt=_resolve_path(prompt["template"], config_dir),
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
        modes=modes,
        seed=int(run.get("seed", 13)),
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
        max_examples=selection.get("max_examples"),
        allow_partial_components=bool(selection.get("allow_partial_components", False)),
        start_index=int(selection.get("start_index", 0)),
        end_index=selection.get("end_index"),
        resume=bool(selection.get("resume", False)),
        collection_id=run.get("name", DEFAULT_RUN_ID),
        config_path=config_path,
        extra_metadata={"raw_config": raw},
    )


def _default_config(data_dir: Path, split: str) -> QueryGenerationConfig:
    return QueryGenerationConfig(
        dataset=Path(data_dir) / "preprocessed",
        visual_interpretations=None,
        textual_interpretations=None,
        clues_dir=Path(data_dir) / "clues",
        output_dir=Path(data_dir) / "query_collections",
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _resolve_optional_path(path: str | Path | None, base_dir: Path) -> Path | None:
    return None if path is None else _resolve_path(path, base_dir)


def _validate_config(config: QueryGenerationConfig) -> None:
    if config.model_provider not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported model provider {config.model_provider!r}; expected one of {sorted(DEFAULT_MODELS)}")
    if config.model == "default":
        object.__setattr__(config, "model", DEFAULT_MODELS[config.model_provider])
    if config.model_provider == "null" and config.model != "null":
        raise ValueError("The null query-generation provider expects model to be 'null'.")
    if config.model_provider != "null" and config.model == "null":
        raise ValueError("Non-null query-generation providers require a real model name.")
    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if config.visual_component_budget < 1:
        raise ValueError("visual_component_budget must be at least 1")
    if config.textual_component_budget < 0:
        raise ValueError("textual_component_budget must be non-negative")
    if config.max_examples is not None and config.max_examples < 1:
        raise ValueError("max_examples must be at least 1 when set")
    if config.start_index < 0:
        raise ValueError("start_index must be non-negative")
    if config.end_index is not None and config.end_index < config.start_index:
        raise ValueError("end_index must be greater than or equal to start_index")
    for mode in config.modes:
        if mode not in ("visual-only", "visual-and-text"):
            raise ValueError(f"Unsupported query mode: {mode}")
    if not config.prompt.exists():
        raise FileNotFoundError(f"Prompt template does not exist: {config.prompt}")


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
        "clues_dir",
        "output_dir",
        "prompt",
        "config_path",
    ):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data
