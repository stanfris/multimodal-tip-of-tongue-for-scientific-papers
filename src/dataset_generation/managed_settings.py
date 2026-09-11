"""Shared loader for project-managed dataset generation settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml


ManagedSet = Literal["train", "test"]
DEFAULT_SETTINGS_PATH = Path("configs/settings.yaml")


def load_managed_settings(path: Path) -> tuple[dict[str, Any], Path]:
    settings_path = path.expanduser().resolve()
    raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Managed settings must be a mapping: {settings_path}")
    return raw, settings_path


def section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Managed settings section {name!r} must be a mapping")
    return value


def managed_set(value: str) -> ManagedSet:
    if value not in ("train", "test"):
        raise ValueError(f"Managed set must be 'train' or 'test', got {value!r}.")
    return value


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def resolve_optional_path(path: str | Path | None, base_dir: Path) -> Path | None:
    return None if path is None else resolve_path(path, base_dir)


def resolve_dataset_path(dataset: dict[str, Any], key: str, config_dir: Path) -> Path:
    value = dataset.get(key)
    if value is None:
        raise ValueError(f"dataset.{key} must be set in managed settings")
    return resolve_path_under_root(value, dataset.get("root"), config_dir)


def resolve_optional_dataset_path(dataset: dict[str, Any], key: str, config_dir: Path) -> Path | None:
    value = dataset.get(key)
    if value is None:
        return None
    return resolve_path_under_root(value, dataset.get("root"), config_dir)


def resolve_path_under_root(path: str | Path, root: str | Path | None, config_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if root is None:
        return (config_dir / candidate).resolve()
    resolved_root = resolve_path(root, config_dir)
    return (resolved_root / candidate).resolve()
