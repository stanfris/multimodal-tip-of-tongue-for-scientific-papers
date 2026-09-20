from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import dataset_generation.query_generation as query_generation
from dataset_generation.synthetic import TestCollectionExample as QueryExample
from tests.test_managed_query_settings import write_settings


def test_test_set_accepts_three_query_limit(tmp_path: Path, monkeypatch) -> None:
    settings = write_settings(tmp_path)
    parser = query_generation.build_parser()
    args = parser.parse_args(["--settings", str(settings), "--set", "test", "--limit", "3"])
    captured = {}

    def fake_generate(config, *, limit):
        captured.update(config=config, limit=limit)
        return tmp_path

    monkeypatch.setattr(query_generation, "generate_query_collections", fake_generate)
    query_generation.run(args)

    assert captured["config"].split_name == "test"
    assert captured["limit"] == 3


def test_limit_applies_to_each_mode_and_visual_text_runs_first(tmp_path: Path, monkeypatch) -> None:
    config = replace(
        query_generation.QueryGenerationConfig(
            dataset=tmp_path,
            visual_interpretations=None,
            textual_interpretations=None,
            clues_dir=tmp_path,
            output_dir=tmp_path / "output",
        ),
        modes=("visual-only", "visual-and-text"),
    )
    calls = []

    monkeypatch.setattr(query_generation, "read_preprocessed_papers", lambda dataset: [])
    monkeypatch.setattr(query_generation, "_components_by_paper", lambda *args, **kwargs: {})
    monkeypatch.setattr(query_generation, "_eligible_papers", lambda papers, config: [])
    monkeypatch.setattr(query_generation, "_read_existing_query_keys", lambda path: set())
    monkeypatch.setattr(query_generation, "write_test_collection", lambda collection, path: None)
    monkeypatch.setattr(query_generation, "_write_collection_metadata", lambda *args: None)
    monkeypatch.setattr(query_generation, "_write_root_metadata", lambda *args: None)

    def fake_mode(mode, papers, components, mode_config, *args, **kwargs):
        existing = kwargs.get("existing")
        count = mode_config.max_examples - (len(existing.examples) if existing is not None else 0)
        calls.append((mode, mode_config.max_examples, count))
        return [QueryExample(f"{mode}-{index}", "query", []) for index in range(count)]

    monkeypatch.setattr(query_generation, "_generate_mode_examples_from_papers", fake_mode)
    query_generation._generate_query_collections_from_preprocessed(config, "prompt", "hash", None, None, limit=3)

    assert calls == [("visual-and-text", 3, 3), ("visual-only", 3, 3)]

    calls.clear()
    existing_queries = [QueryExample(f"existing-{index}", "query", []) for index in range(3)]

    def fake_existing(path, mode):
        examples = existing_queries if mode == "visual-and-text" else []
        return query_generation.ExistingQueryState(examples, set(), set())

    monkeypatch.setattr(query_generation, "_read_existing_query_state", fake_existing)
    query_generation._generate_query_collections_from_preprocessed(
        replace(config, resume=True), "prompt", "hash", None, None, limit=3
    )

    assert calls == [("visual-and-text", 3, 0), ("visual-only", 3, 3)]
