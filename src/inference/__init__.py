"""Shared, lazily loaded generation backends."""
from inference.base import GenerationRequest, GenerationResult, InferenceBackend, load_backend

__all__ = ["GenerationRequest", "GenerationResult", "InferenceBackend", "load_backend"]
