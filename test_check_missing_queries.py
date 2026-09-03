import importlib.util
import json
from pathlib import Path


def load_checker():
    script_path = Path(__file__).parent / "scripts" / "check_missing_queries.py"
    spec = importlib.util.spec_from_file_location("check_missing_queries", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_invalid_visual_clue_reasons_flags_underfilled_paper():
    checker = load_checker()
    clue = {"paper_id": "paper-1", "figure_id": "fig-1", "kind": "visual", "output": "one cue"}

    reasons = checker.invalid_visual_clue_reasons(clue, insufficient_paper_ids={"paper-1"})

    assert reasons == ["paper has fewer visual components than required"]


def test_delete_wrong_visual_lines_rewrites_visual_clues_file(tmp_path):
    checker = load_checker()
    valid_visual = {"paper_id": "paper-1", "figure_id": "fig-1", "kind": "visual", "output": "valid"}
    invalid_visual = {"paper_id": "paper-2", "figure_id": "fig-2", "kind": "visual", "output": "underfilled"}
    textual_clue = {"paper_id": "paper-2", "kind": "textual", "output": "kept because this only cleans visual rows"}
    visual_clues_path = tmp_path / "visual_clues.jsonl"
    visual_clues_path.write_text(
        "\n".join(json.dumps(row) for row in [valid_visual, invalid_visual, textual_clue]) + "\n",
        encoding="utf-8",
    )

    report = checker.delete_wrong_visual_lines(
        visual_clues_path,
        insufficient_paper_ids={"paper-2"},
        backup=True,
    )

    assert report["deleted_lines"] == 1
    assert (tmp_path / "visual_clues.jsonl.bak").exists()
    remaining = [json.loads(line) for line in visual_clues_path.read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in remaining] == ["visual", "textual"]
    assert [row["paper_id"] for row in remaining] == ["paper-1", "paper-2"]


def test_delete_wrong_visual_lines_keeps_single_meaningful_visual_component(tmp_path):
    checker = load_checker()
    meaningful_visual = {
        "paper_id": "paper-1",
        "figure_id": "fig-1",
        "kind": "visual",
        "output": json.dumps(
            {
                "figure_type": [],
                "layout": [],
                "visual_elements": ["two photographs side by side"],
                "colors_and_styles": [],
                "relationships": [],
                "distinctive_visual_cues": [],
            }
        ),
    }
    other_visual = {
        "paper_id": "paper-1",
        "figure_id": "fig-2",
        "kind": "visual",
        "output": json.dumps({"visual_elements": ["one", "two", "three"]}),
    }
    visual_clues_path = tmp_path / "visual_clues.jsonl"
    visual_clues_path.write_text(
        "\n".join(json.dumps(row) for row in [meaningful_visual, other_visual]) + "\n",
        encoding="utf-8",
    )

    report = checker.delete_wrong_visual_lines(
        visual_clues_path,
        insufficient_paper_ids=set(),
        backup=True,
    )

    assert report["deleted_lines"] == 0
    remaining = [json.loads(line) for line in visual_clues_path.read_text(encoding="utf-8").splitlines()]
    assert [row["figure_id"] for row in remaining] == ["fig-1", "fig-2"]


def test_invalid_visual_clue_reasons_flags_empty_visual_json_without_underfill():
    checker = load_checker()
    clue = {
        "paper_id": "paper-1",
        "figure_id": "fig-1",
        "kind": "visual",
        "output": json.dumps(
            {
                "figure_type": [],
                "layout": [],
                "visual_elements": [],
                "colors_and_styles": [],
                "relationships": [],
                "distinctive_visual_cues": [],
            }
        ),
    }

    reasons = checker.invalid_visual_clue_reasons(clue, insufficient_paper_ids=set())

    assert reasons == ["visual clue output is empty JSON"]


def test_delete_wrong_visual_lines_rewrites_empty_visual_json_clue(tmp_path):
    checker = load_checker()
    empty_visual = {
        "paper_id": "paper-1",
        "figure_id": "fig-1",
        "kind": "visual",
        "output": json.dumps(
            {
                "figure_type": [],
                "layout": [],
                "visual_elements": [],
                "colors_and_styles": [],
                "relationships": [],
                "distinctive_visual_cues": [],
            }
        ),
    }
    valid_visual = {
        "paper_id": "paper-1",
        "figure_id": "fig-2",
        "kind": "visual",
        "output": json.dumps({"visual_elements": ["line chart"]}),
    }
    visual_clues_path = tmp_path / "visual_clues.jsonl"
    visual_clues_path.write_text(
        "\n".join(json.dumps(row) for row in [empty_visual, valid_visual]) + "\n",
        encoding="utf-8",
    )

    report = checker.delete_wrong_visual_lines(
        visual_clues_path,
        insufficient_paper_ids=set(),
        backup=True,
    )

    assert report["deleted_lines"] == 1
    assert report["deleted"] == [(1, "fig-1", ["visual clue output is empty JSON"])]
    remaining = [json.loads(line) for line in visual_clues_path.read_text(encoding="utf-8").splitlines()]
    assert [row["figure_id"] for row in remaining] == ["fig-2"]


def test_find_invalid_visual_clue_lines_reports_empty_visual_json(tmp_path):
    checker = load_checker()
    visual_clues_path = tmp_path / "visual_clues.jsonl"
    empty_visual = {
        "paper_id": "paper-1",
        "figure_id": "fig-1",
        "kind": "visual",
        "output": json.dumps(
            {
                "figure_type": [],
                "layout": [],
                "visual_elements": [],
                "colors_and_styles": [],
                "relationships": [],
                "distinctive_visual_cues": [],
            }
        ),
    }
    visual_clues_path.write_text(json.dumps(empty_visual) + "\n", encoding="utf-8")

    findings = checker.find_invalid_visual_clue_lines(
        visual_clues_path,
        insufficient_paper_ids=set(),
    )

    assert findings == [
        {
            "path": visual_clues_path,
            "line_number": 1,
            "paper_id": "paper-1",
            "figure_id": "fig-1",
            "reasons": ["visual clue output is empty JSON"],
        }
    ]


def test_checker_parses_prefixed_empty_visual_json_without_counting_prefix_as_component():
    checker = load_checker()
    text = """Figure 2: {
  "figure_type": [],
  "layout": [],
  "visual_elements": [],
  "colors_and_styles": [],
  "relationships": [],
  "distinctive_visual_cues": []
}"""

    assert checker.split_component_text(text, "visual") == []


def test_find_queries_with_empty_visual_inputs_reports_generated_query(tmp_path):
    checker = load_checker()
    query_path = tmp_path / "queries.jsonl"
    empty_visual = """Figure 2: {
  "figure_type": [],
  "layout": [],
  "visual_elements": [],
  "colors_and_styles": [],
  "relationships": [],
  "distinctive_visual_cues": []
}"""
    row = {
        "query_id": "visual_only_q00001",
        "query": "generated from invalid visual clue",
        "relevant_ids": ["paper-1"],
        "metadata": {
            "mode": "visual-only",
            "paper_id": "paper-1",
            "selected_components": [
                {"record_id": "fig-1", "kind": "visual", "text": empty_visual},
                {"record_id": "fig-2", "kind": "visual", "text": "Figure 1: line chart"},
            ],
        },
    }
    query_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    findings = checker.find_queries_with_empty_visual_inputs([query_path])

    assert len(findings) == 1
    assert findings[0]["query_id"] == "visual_only_q00001"
    assert findings[0]["paper_id"] == "paper-1"
    assert findings[0]["bad_components"] == [
        {"component_index": 0, "record_id": "fig-1", "text_first_line": "Figure 2: {"}
    ]


def test_delete_empty_visual_query_lines_rewrites_query_file(tmp_path):
    checker = load_checker()
    query_path = tmp_path / "queries.jsonl"
    empty_visual = """Figure 2: {
  "figure_type": [],
  "layout": [],
  "visual_elements": [],
  "colors_and_styles": [],
  "relationships": [],
  "distinctive_visual_cues": []
}"""
    invalid_query = {
        "query_id": "visual_only_q00001",
        "query": "generated from invalid visual clue",
        "relevant_ids": ["paper-1"],
        "metadata": {
            "mode": "visual-only",
            "paper_id": "paper-1",
            "selected_components": [{"record_id": "fig-1", "kind": "visual", "text": empty_visual}],
        },
    }
    valid_query = {
        "query_id": "visual_only_q00002",
        "query": "generated from valid visual clue",
        "relevant_ids": ["paper-2"],
        "metadata": {
            "mode": "visual-only",
            "paper_id": "paper-2",
            "selected_components": [{"record_id": "fig-2", "kind": "visual", "text": "Figure 1: line chart"}],
        },
    }
    query_path.write_text(
        "\n".join(json.dumps(row) for row in [invalid_query, valid_query]) + "\n",
        encoding="utf-8",
    )

    reports = checker.delete_empty_visual_query_lines([query_path], backup=True)

    assert reports[0]["deleted_lines"] == 1
    assert reports[0]["kept_lines"] == 1
    assert (tmp_path / "queries.jsonl.bak").exists()
    remaining = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
    assert [row["query_id"] for row in remaining] == ["visual_only_q00002"]


def test_checked_query_paths_always_includes_canonical_queries(tmp_path):
    checker = load_checker()
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    canonical_query_path = canonical / "queries.jsonl"
    canonical_query_path.write_text("", encoding="utf-8")
    extra_query_path = tmp_path / "custom" / "queries.jsonl"
    config = type(
        "Config",
        (),
        {
            "dataset": canonical,
            "output_dir": tmp_path / "query_collections",
            "collection_id": "query_generation_default",
        },
    )()

    paths = checker.checked_query_paths(config, [extra_query_path])

    assert paths == [canonical_query_path.resolve(), extra_query_path.resolve()]
