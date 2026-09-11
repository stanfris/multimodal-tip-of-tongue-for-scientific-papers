from __future__ import annotations

from dataset_generation.model_output import parse_json_object


def test_parse_json_object_accepts_surrounding_text() -> None:
    assert parse_json_object('Here is the object: {"query": "find this paper"}') == {
        "query": "find this paper"
    }


def test_parse_json_object_rejects_non_object_json() -> None:
    assert parse_json_object('["not", "a", "mapping"]') is None
