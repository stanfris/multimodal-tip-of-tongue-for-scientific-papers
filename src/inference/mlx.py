"""Existing Apple Silicon generation exposed through the shared contract."""
from __future__ import annotations

from typing import Any

from inference.base import GenerationRequest, GenerationResult


class MLXBackend:
    def __init__(self, model: str, runtime: dict[str, Any], image: dict[str, Any]) -> None:
        self.visual = "vl" in model.lower()
        try:
            if self.visual:
                from clues.vl_figure_descriptions import load_mlx_model
            else:
                from clues.textual_clue_descriptions import load_mlx_model
            self.loaded = load_mlx_model(model)
        except ImportError as exc:
            raise RuntimeError("MLX backend requested but its mlx-lm/mlx-vlm dependency is unavailable.") from exc

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if self.visual:
            from clues.vl_figure_descriptions import generate_with_mlx
        else:
            from clues.textual_clue_descriptions import generate_with_mlx
        results = []
        for request in requests:
            try:
                if self.visual:
                    if len(request.images) != 1:
                        raise ValueError("The existing MLX visual backend requires exactly one image")
                    value = generate_with_mlx(self.loaded, request.images[0], request.prompt,
                                              request.max_tokens, request.temperature)
                else:
                    if request.images:
                        raise ValueError("The existing MLX text backend does not accept images")
                    value = generate_with_mlx(self.loaded, request.prompt, request.max_tokens,
                                              request.temperature, thinking=request.thinking)
                results.append(GenerationResult(request.request_id, value))
            except Exception as exc:
                results.append(GenerationResult(request.request_id, error=exc))
        return results
