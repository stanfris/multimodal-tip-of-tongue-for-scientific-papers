"""Parse model-generated clue text into reusable query components."""

from __future__ import annotations

import json
import re
from typing import Any, Literal


ClueKind = Literal["visual", "textual"]


def split_component_text(text: str, kind: ClueKind) -> list[str]:
    decoded = _decode_jsonish_components(text)
    if decoded is None:
        return [text]
    if kind == "textual":
        components = [_textual_component(value) for value in decoded]
    else:
        components = []
        for value in decoded:
            components.extend(_visual_components(value))
    return [component.strip() for component in components if component and component.strip()]


def _repair_json_dict(text: str) -> dict[str, Any] | None:
    closings = ["", '"', '"]}', ']}', '}', '"}']

    stripped = re.sub(r",\s*$", "", text.strip())
    for closing in closings:
        try:
            value = json.loads(stripped + closing)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    while "," in text:
        text = text[: text.rfind(",")]
        stripped = text.strip()
        for closing in closings:
            try:
                value = json.loads(stripped + closing)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _decode_jsonish_components(text: str) -> list[Any] | None:
    text = text.strip()
    first_json = min((index for index in (text.find("["), text.find("{")) if index != -1), default=-1)
    if first_json > 0:
        text = text[first_json:]
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]

    decoder = json.JSONDecoder()
    index = 0
    values: list[Any] = []
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] in ","):
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            remaining = text[index:].strip()
            dict_start = remaining.find("{")
            if dict_start != -1:
                repaired = _repair_json_dict(remaining[dict_start:])
                if repaired:
                    values.append(repaired)
            break
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
        index = end
    return values or None


def _textual_component(value: Any) -> str | None:
    if isinstance(value, dict):
        cue = value.get("cue")
        if cue is None:
            return None
        category = value.get("category")
        if category is None:
            return str(cue)
        return f"textual {category}: {cue}"
    return str(value)


def _visual_components(value: Any) -> list[str]:
    if isinstance(value, dict):
        components: list[str] = []
        for field_name, field_value in _ordered_visual_fields(value):
            if isinstance(field_value, list):
                components.extend(f"{field_name}: {item}" for item in field_value)
            elif field_value is not None:
                components.append(f"{field_name}: {field_value}")
        return components
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _ordered_visual_fields(value: dict[str, Any]) -> list[tuple[str, Any]]:
    preferred = [
        "figure_type",
        "layout",
        "visual_elements",
        "colors_and_styles",
        "relationships",
        "distinctive_visual_cues",
    ]
    ordered = [(field_name, value[field_name]) for field_name in preferred if field_name in value]
    ordered.extend((field_name, field_value) for field_name, field_value in value.items() if field_name not in preferred)
    return ordered
