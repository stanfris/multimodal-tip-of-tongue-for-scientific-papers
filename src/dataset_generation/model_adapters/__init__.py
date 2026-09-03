"""Model adapter registry for managed dataset generation runs."""

from dataset_generation.model_adapters.base import ModelAdapter
from dataset_generation.model_adapters.null import NullModelAdapter


MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {
    NullModelAdapter.provider: NullModelAdapter,
}


def get_model_adapter(provider: str) -> type[ModelAdapter]:
    try:
        return MODEL_ADAPTERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown model provider {provider!r}; expected one of {sorted(MODEL_ADAPTERS)}") from exc


__all__ = ["ModelAdapter", "get_model_adapter"]
