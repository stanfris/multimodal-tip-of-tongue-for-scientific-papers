"""Request contract shared by all offline generation stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    images: tuple[Path, ...] = ()
    max_tokens: int = 256
    temperature: float = 0.0
    thinking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str = ""
    generated_tokens: int | None = None
    error: Exception | None = None


class InferenceBackend(Protocol):
    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...


def load_backend(provider: str, model: str, *, runtime: dict[str, Any] | None = None,
                 image: dict[str, Any] | None = None) -> InferenceBackend:
    """Load only the selected provider's optional dependencies."""
    if provider == "transformers":
        from inference.transformers import TransformersBackend
        return TransformersBackend(model, runtime or {}, image or {})
    if provider == "mlx":
        from inference.mlx import MLXBackend
        return MLXBackend(model, runtime or {}, image or {})
    if provider == "vllm":
        from inference.vllm import VLLMBackend
        return VLLMBackend(model, runtime or {}, image or {})
    raise ValueError(f"Unsupported inference provider: {provider}")


def ordered_results(requests: list[GenerationRequest], results: list[GenerationResult]) -> list[GenerationResult]:
    """Validate IDs and restore input order even if an engine completes out of order."""
    ids = [request.request_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("Generation request IDs must be unique within a batch")
    by_id = {result.request_id: result for result in results}
    if len(by_id) != len(results) or set(by_id) != set(ids):
        raise ValueError("Backend returned missing, duplicate, or unknown request IDs")
    return [by_id[request_id] for request_id in ids]
