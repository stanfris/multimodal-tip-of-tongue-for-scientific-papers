"""Base interfaces for optional model-backed generation steps."""

from __future__ import annotations

from typing import Protocol

from dataset_generation.config import LoadedPrompt, ManagedRunConfig


class ModelAdapter(Protocol):
    provider: str

    def validate(self, config: ManagedRunConfig, prompt: LoadedPrompt | None) -> None:
        """Validate model-specific configuration before the parser runs."""
