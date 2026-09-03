from __future__ import annotations

import json

import pytest

from dataset_generation.config import load_prompt, load_run_config
from dataset_generation.manager import run_managed_dataset
from dataset_generation.parsers import PARSERS
from dataset_generation.parsers.base import ParserResult
from dataset_generation.storage import read_dataset_artifact


def fixture_records():
    return [
        {
            "record_id": "fig-1",
            "filename": "2020.acl-main.128.pdf-Figure1.png",
            "extracted_paper_id": "2020.acl-main.128",
            "normalized_paper_id": "2020.acl-main.128",
            "resolved_paper_id": "2020.acl-main.128",
            "label": "bar chart",
            "markdown": "# Direct match",
            "match_status": "matched",
            "source_fig_dataset": "citeseerx/ACL-fig",
            "source_fig_split": "train",
            "source_paper_dataset": "KRLabsOrg/acl-anthology-md",
            "source_paper_config": "fulltext",
        }
    ]


class FixtureParser:
    name = "fixture"

    def parse(self, config):
        return ParserResult(
            records=fixture_records(),
            stats={"total_records": 1, "matched_records": 1},
            metadata={"seed": config.run.seed},
        )


def test_load_run_config_resolves_paths(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe {filename}", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run:
  name: fixture_run
dataset:
  data_dir: data
parser:
  name: fixture
prompt:
  name: visual_interpretation
  template: prompt.txt
model:
  provider: "null"
  name: "null"
output:
  dir: out
  format: jsonl
""",
        encoding="utf-8",
    )

    config = load_run_config(config_path)
    loaded_prompt = load_prompt(config.prompt)

    assert config.dataset.data_dir == str(tmp_path / "data")
    assert config.output.dir == str(tmp_path / "out")
    assert loaded_prompt is not None
    assert loaded_prompt.sha256


def test_load_run_config_rejects_missing_prompt(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run:
  name: broken_run
prompt:
  name: visual_interpretation
  template: missing.txt
model:
  provider: "null"
  name: "null"
output:
  dir: out
  format: jsonl
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Prompt template"):
        load_run_config(config_path)


def test_managed_run_writes_provenance(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe {filename}", encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
run:
  name: fixture_run
  seed: 99
parser:
  name: fixture
prompt:
  name: visual_interpretation
  version: v1
  template: prompt.txt
model:
  provider: "null"
  name: "null"
output:
  dir: out
  format: jsonl
""",
        encoding="utf-8",
    )
    PARSERS[FixtureParser.name] = FixtureParser
    try:
        output = run_managed_dataset(load_run_config(config_path), command=["dataset-generation", "run"])
    finally:
        del PARSERS[FixtureParser.name]

    frame = read_dataset_artifact(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

    assert list(frame["record_id"]) == ["fig-1"]
    assert (output / "resolved_config.yaml").exists()
    assert (output / "prompt.txt").read_text(encoding="utf-8") == "Describe {filename}"
    assert metadata["managed_run"]["run"]["name"] == "fixture_run"
    assert metadata["managed_run"]["prompt"]["sha256"]
