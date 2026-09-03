"""Typed configuration loading for managed dataset generation runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dataset_generation.sources import (
    ACL_ANTHOLOGY_MD_CONFIG,
    ACL_ANTHOLOGY_MD_DATASET,
    ACL_FIG_DATASET,
    DEFAULT_DATA_DIR,
)
from dataset_generation.storage import SUPPORTED_FORMATS


@dataclass(frozen=True)
class RunSection:
    name: str
    seed: int = 13


@dataclass(frozen=True)
class DatasetSection:
    figure_dataset: str = ACL_FIG_DATASET
    figure_split: str = "train"
    paper_dataset: str = ACL_ANTHOLOGY_MD_DATASET
    paper_config: str = ACL_ANTHOLOGY_MD_CONFIG
    paper_split: str = "train"
    data_dir: str = str(DEFAULT_DATA_DIR)
    prefer_local_sources: bool = True
    streaming: bool = False
    limit: int | None = None
    paper_limit: int | None = None


@dataclass(frozen=True)
class ParserSection:
    name: str = "acl_fig_markdown"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSection:
    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    version: str = "v1"


@dataclass(frozen=True)
class ModelSection:
    provider: str = "null"
    name: str = "null"
    temperature: float = 0.0
    max_tokens: int | None = None
    batch_size: int = 1
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputSection:
    dir: str
    format: str = "parquet"


@dataclass(frozen=True)
class ManagedRunConfig:
    run: RunSection
    dataset: DatasetSection
    parser: ParserSection
    prompt: PromptSection | None
    model: ModelSection
    output: OutputSection
    config_path: str

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(frozen=True)
class LoadedPrompt:
    name: str
    version: str
    path: str
    text: str
    sha256: str
    variables: list[str]


def load_run_config(path: str | Path) -> ManagedRunConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    config_dir = config_path.parent
    run = _section(raw, "run")
    output = _section(raw, "output")

    dataset = DatasetSection(**_section(raw, "dataset", required=False))
    parser = ParserSection(**_section(raw, "parser", required=False))
    prompt_raw = raw.get("prompt")
    prompt = PromptSection(**prompt_raw) if prompt_raw is not None else None
    model = ModelSection(**_section(raw, "model", required=False))

    config = ManagedRunConfig(
        run=RunSection(**run),
        dataset=_resolve_dataset_paths(dataset, config_dir),
        parser=parser,
        prompt=_resolve_prompt_path(prompt, config_dir) if prompt else None,
        model=model,
        output=_resolve_output_path(OutputSection(**output), config_dir),
        config_path=str(config_path),
    )
    _validate_config(config)
    return config


def load_prompt(prompt: PromptSection | None) -> LoadedPrompt | None:
    if prompt is None:
        return None
    path = Path(prompt.template)
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LoadedPrompt(
        name=prompt.name,
        version=prompt.version,
        path=str(path),
        text=text,
        sha256=digest,
        variables=prompt.variables,
    )


def _section(raw: dict[str, Any], name: str, required: bool = True) -> dict[str, Any]:
    value = raw.get(name)
    if value is None:
        if required:
            raise ValueError(f"Config is missing required section: {name}")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _resolve_dataset_paths(dataset: DatasetSection, config_dir: Path) -> DatasetSection:
    return DatasetSection(
        figure_dataset=dataset.figure_dataset,
        figure_split=dataset.figure_split,
        paper_dataset=dataset.paper_dataset,
        paper_config=dataset.paper_config,
        paper_split=dataset.paper_split,
        data_dir=str(_resolve_path(dataset.data_dir, config_dir)),
        prefer_local_sources=dataset.prefer_local_sources,
        streaming=dataset.streaming,
        limit=dataset.limit,
        paper_limit=dataset.paper_limit,
    )


def _resolve_prompt_path(prompt: PromptSection, config_dir: Path) -> PromptSection:
    return PromptSection(
        name=prompt.name,
        template=str(_resolve_path(prompt.template, config_dir)),
        variables=prompt.variables,
        version=prompt.version,
    )


def _resolve_output_path(output: OutputSection, config_dir: Path) -> OutputSection:
    return OutputSection(dir=str(_resolve_path(output.dir, config_dir)), format=output.format)


def _resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _validate_config(config: ManagedRunConfig) -> None:
    if not config.run.name:
        raise ValueError("run.name must not be empty")
    if config.dataset.limit is not None and config.dataset.limit < 0:
        raise ValueError("dataset.limit must be non-negative")
    if config.dataset.paper_limit is not None and config.dataset.paper_limit < 0:
        raise ValueError("dataset.paper_limit must be non-negative")
    if config.model.batch_size < 1:
        raise ValueError("model.batch_size must be at least 1")
    if config.model.max_tokens is not None and config.model.max_tokens < 1:
        raise ValueError("model.max_tokens must be at least 1 when set")
    if config.output.format.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported output format {config.output.format!r}; expected one of {sorted(SUPPORTED_FORMATS)}"
        )
    if config.prompt is not None and not Path(config.prompt.template).exists():
        raise FileNotFoundError(f"Prompt template does not exist: {config.prompt.template}")
