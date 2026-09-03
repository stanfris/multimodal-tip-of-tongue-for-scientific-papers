"""Base interfaces for dataset parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dataset_generation.config import ManagedRunConfig


@dataclass(frozen=True)
class ParserResult:
    records: list[dict[str, Any]]
    stats: dict[str, Any]
    metadata: dict[str, Any]


class DatasetParser(Protocol):
    name: str

    def parse(self, config: ManagedRunConfig) -> ParserResult:
        """Parse/build records for a managed run."""
