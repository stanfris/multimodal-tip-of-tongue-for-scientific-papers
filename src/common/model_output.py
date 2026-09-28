"""Helpers for parsing model-generated text."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, allowing surrounding text around the object."""
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
