from __future__ import annotations
import json
import re
from typing import Any

def _repair_json_dict(text: str) -> dict[str, Any] | None:
    closings = ["", '"', '"]}', ']}', '}', '"}']
    stripped = re.sub(r",\s*$", "", text.strip())
    for closing in closings:
        try:
            value = json.loads(stripped + closing)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    while "," in text:
        text = text[: text.rfind(",")]
        stripped = text.strip()
        for closing in closings:
            try:
                value = json.loads(stripped + closing)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    return None

def _decode_jsonish_components(text: str) -> list[Any] | None:
    text = text.strip()
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

text = """Figure 2: {
  "figure_type": [],
  "layout": [],
  "visual_elements": [],
  "colors_and_styles": [],
  "relationships": [],
  "distinctive_visual_cues": []
}"""

print(_decode_jsonish_components(text))
