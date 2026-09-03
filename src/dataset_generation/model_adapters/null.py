"""Deterministic no-op model adapter used by config-driven dataset builds."""

from __future__ import annotations

from dataset_generation.config import LoadedPrompt, ManagedRunConfig


class NullModelAdapter:
    provider = "null"

    def validate(self, config: ManagedRunConfig, prompt: LoadedPrompt | None) -> None:
        if config.model.name != "null":
            raise ValueError("The null model provider expects model.name to be 'null'")
