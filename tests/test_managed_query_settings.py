from __future__ import annotations

from pathlib import Path

import pytest

from dataset_generation.query_generation import (
    QueryGenerationConfig,
    _eligible_papers,
    _query_id,
    build_parser,
    load_query_generation_config,
)
from dataset_generation.textual_clue_descriptions import (
    build_parser as build_textual_parser,
    load_managed_textual_description_args,
)
from dataset_generation.vl_figure_descriptions import (
    build_parser as build_visual_parser,
    load_managed_visual_description_args,
)
from scripts.judge_queries import build_parser as build_judgement_parser, load_managed_judgement_args


def write_settings(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    (data_root / "preprocessed").mkdir(parents=True)
    (data_root / "clues").mkdir()
    (data_root / "splits").mkdir()
    (data_root / "splits" / "document_split.json").write_text(
        '{"train": [], "test": []}',
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "visual_interpretation.v1.txt").write_text("{image}", encoding="utf-8")
    (prompts / "textual_interpretation.v1.txt").write_text("{paper_text}", encoding="utf-8")
    (prompts / "query_generation.v1.txt").write_text("{selected_cues}", encoding="utf-8")
    (prompts / "query_judgement.v1.txt").write_text("{}", encoding="utf-8")

    settings = tmp_path / "configs" / "settings.yaml"
    settings.parent.mkdir()
    settings.write_text(
        """
dataset:
  name: test_dataset
  root: ../data
  preprocessed: preprocessed
  clues_dir: clues
  split_index: splits/document_split.json
  visual_interpretations:
  textual_interpretations:

visual_descriptions:
  name: qwen3_vl_figure_description
  prompt:
    name: visual_interpretation
    version: v1
    template: ../prompts/visual_interpretation.v1.txt
  model:
    provider: "transformers"
    name: "Qwen/Qwen3-VL-4B-Instruct"
  selection:
    all: true
    start_index: 3
    end_index: 9
    limit: 4
    resume: true
    overwrite: false
  generation:
    max_tokens: 600
    temperature: 0.0
    batch_size: 1
    device_map: "auto"
    dtype: "bfloat16"
    attn_implementation: "sdpa"
    debug: false

textual_descriptions:
  name: qwen3_textual_clue_description
  prompt:
    name: textual_interpretation
    version: v1
    template: ../prompts/textual_interpretation.v1.txt
  model:
    provider: "transformers"
    name: "Qwen/Qwen3-4B"
  selection:
    all: true
    start_index: 5
    end_index: 11
    limit: 6
    resume: true
    overwrite: false
  generation:
    max_tokens: 600
    max_markdown_chars: 12000
    temperature: 0.0
    thinking: false
    batch_size: 8
    device_map: "auto"
    dtype: "bfloat16"
    attn_implementation: "sdpa"

visual_query:
  name: managed_queries
  seed: 17
  prompt:
    name: query_generation
    version: v1
    template: ../prompts/query_generation.v1.txt
  model:
    provider: "null"
    name: "null"
    temperature: 0.0
    max_tokens: 220
    device_map: "auto"
    dtype: "bfloat16"
    attn_implementation: "sdpa"
    thinking: false
  judgement:
    enabled: false
    name: query_judgement
    version: v1
    template: ../prompts/query_judgement.v1.txt
    provider: "transformers"
    model: "google/gemma-3-27b-it"
    temperature: 0.0
    max_tokens: 900
    device_map: "auto"
    dtype: "bfloat16"
    attn_implementation: "sdpa"
    run:
      modes:
        - visual-only
      limit: 7
      overwrite: true
  selection:
    modes:
      - visual-only
    visual_component_budget: 3
    textual_component_budget: 3
    allow_partial_components: false
    start_index: 0
    end_index:
    max_examples:
    resume: true
  output:
    dir: ../data/query_collections
  query_sets:
    train:
      split_name: train
      collection_id: managed_train
      max_examples: 5
    test:
      split_name: test
      collection_id: managed_test
      max_examples: 2
""",
        encoding="utf-8",
    )
    return settings


def test_managed_settings_select_train_and_test_query_sets(tmp_path: Path) -> None:
    settings = write_settings(tmp_path)
    parser = build_parser()

    train = load_query_generation_config(parser.parse_args(["--settings", str(settings)]))
    test = load_query_generation_config(parser.parse_args(["--settings", str(settings), "--set", "test"]))

    assert train.dataset == tmp_path / "data" / "preprocessed"
    assert train.clues_dir == tmp_path / "data" / "clues"
    assert train.split_index == tmp_path / "data" / "splits" / "document_split.json"
    assert train.split_name == "train"
    assert train.collection_id == "managed_train"
    assert train.max_examples == 5
    assert test.split_name == "test"
    assert test.collection_id == "managed_test"
    assert test.max_examples == 2


def test_managed_settings_load_visual_description_stage(tmp_path: Path) -> None:
    settings = write_settings(tmp_path)
    args = build_visual_parser().parse_args(["--settings", str(settings), "--set", "test"])

    managed = load_managed_visual_description_args(args)

    assert managed.dataset == tmp_path / "data" / "preprocessed"
    assert managed.clues_dir == tmp_path / "data" / "clues"
    assert managed.split_index == tmp_path / "data" / "splits" / "document_split.json"
    assert managed.split == "test"
    assert managed.backend == "transformers"
    assert managed.model == "Qwen/Qwen3-VL-4B-Instruct"
    assert managed.prompt == tmp_path / "prompts" / "visual_interpretation.v1.txt"
    assert managed.all is True
    assert managed.limit == 4
    assert managed.start_index == 3
    assert managed.end_index == 9
    assert managed.resume is True
    assert managed.max_tokens == 600
    assert managed.dtype == "bfloat16"


def test_managed_settings_load_textual_description_stage(tmp_path: Path) -> None:
    settings = write_settings(tmp_path)
    args = build_textual_parser().parse_args(["--settings", str(settings)])

    managed = load_managed_textual_description_args(args)

    assert managed.dataset == tmp_path / "data" / "preprocessed"
    assert managed.clues_dir == tmp_path / "data" / "clues"
    assert managed.split_index == tmp_path / "data" / "splits" / "document_split.json"
    assert managed.split == "train"
    assert managed.backend == "transformers"
    assert managed.model == "Qwen/Qwen3-4B"
    assert managed.prompt == tmp_path / "prompts" / "textual_interpretation.v1.txt"
    assert managed.all is True
    assert managed.limit == 6
    assert managed.start_index == 5
    assert managed.end_index == 11
    assert managed.resume is True
    assert managed.batch_size == 8
    assert managed.max_markdown_chars == 12000


def test_managed_settings_load_posthoc_judgement_stage(tmp_path: Path) -> None:
    settings = write_settings(tmp_path)
    args = build_judgement_parser().parse_args(["--settings", str(settings), "--set", "test"])

    managed, config = load_managed_judgement_args(args)

    assert managed.dataset == tmp_path / "data" / "preprocessed"
    assert managed.split_index == tmp_path / "data" / "splits" / "document_split.json"
    assert managed.split == "test"
    assert managed.input_dir == tmp_path / "data" / "query_collections" / "managed_test"
    assert managed.collection_id == "managed_test"
    assert managed.mode == ["visual-only"]
    assert managed.prompt == tmp_path / "prompts" / "query_judgement.v1.txt"
    assert managed.model == "google/gemma-3-27b-it"
    assert managed.limit == 7
    assert managed.overwrite is True
    assert config.judge_queries is True


@pytest.mark.parametrize("flag", ["--dataset", "--model", "--collection-id", "--limit", "--mode"])
def test_query_generation_parser_rejects_setting_overrides(flag: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([flag, "value"])


def test_query_ids_use_original_split_index_after_windowing() -> None:
    config = QueryGenerationConfig(
        dataset=Path("dataset"),
        visual_interpretations=None,
        textual_interpretations=None,
        clues_dir=Path("clues"),
        output_dir=Path("out"),
        split_name="train",
        start_index=3,
        end_index=5,
    )
    papers = [{"paper_id": f"paper-{index}"} for index in range(8)]

    selected = _eligible_papers(papers, config)

    assert [paper["paper_id"] for _, paper in selected] == ["paper-3", "paper-4"]
    assert [_query_id("visual-only", index, config) for index, _ in selected] == [
        "visual_only_q00003",
        "visual_only_q00004",
    ]
