"""Managed dataset generation orchestration."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from dataset_generation.config import LoadedPrompt, ManagedRunConfig, load_prompt, load_run_config
from dataset_generation.model_adapters import get_model_adapter
from dataset_generation.parsers import get_parser
from dataset_generation.sources import ACL_FIG_DATASET, materialize_acl_fig_images
from dataset_generation.storage import write_dataset_artifact


def run_from_config(config_path: str | Path, command: list[str] | None = None) -> Path:
    config = load_run_config(config_path)
    return run_managed_dataset(config, command=command)


def run_managed_dataset(config: ManagedRunConfig, command: list[str] | None = None) -> Path:
    prompt = load_prompt(config.prompt)
    model_adapter = get_model_adapter(config.model.provider)()
    model_adapter.validate(config, prompt)

    parser = get_parser(config.parser.name)()
    result = parser.parse(config)

    metadata = {
        **result.metadata,
        "managed_run": {
            "run": asdict(config.run),
            "dataset": asdict(config.dataset),
            "parser": asdict(config.parser),
            "prompt": _prompt_metadata(prompt),
            "model": asdict(config.model),
            "output": asdict(config.output),
            "config_path": config.config_path,
            "command": command or [],
            "git_commit": _git_commit(Path(config.config_path).parent),
        },
    }
    if config.parser.name == "acl_fig_markdown" and config.dataset.figure_dataset == ACL_FIG_DATASET:
        image_stats = materialize_acl_fig_images(
            result.records,
            config.output.dir,
            split=config.dataset.figure_split,
            data_dir=config.dataset.data_dir,
            prefer_local=config.dataset.prefer_local_sources,
        )
        result.stats["images_written"] = image_stats["images_written"]
        result.stats["missing_images"] = image_stats["missing_images"]
        metadata["image_artifacts"] = image_stats
        metadata.setdefault("layout", {})["images"] = "images/<record_id>.<ext>"
    output_path = write_dataset_artifact(
        result.records,
        config.output.dir,
        metadata,
        result.stats,
        config.output.format,
    )
    _write_resolved_config(output_path, config)
    if prompt is not None:
        (output_path / "prompt.txt").write_text(prompt.text, encoding="utf-8")
    return output_path


def _prompt_metadata(prompt: LoadedPrompt | None) -> dict[str, Any] | None:
    if prompt is None:
        return None
    return {
        "name": prompt.name,
        "version": prompt.version,
        "path": prompt.path,
        "sha256": prompt.sha256,
        "variables": prompt.variables,
    }


def _write_resolved_config(output_path: Path, config: ManagedRunConfig) -> None:
    serializable = config.to_metadata()
    (output_path / "resolved_config.yaml").write_text(
        yaml.safe_dump(serializable, sort_keys=False),
        encoding="utf-8",
    )
    (output_path / "resolved_config.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _git_commit(start_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
