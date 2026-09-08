from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict

import pytest

from dataset_generation.canonical import (
    align_canonical_clues,
    prepare_canonical_papers_from_rows,
    canonical_textual_clues_path,
    canonical_visual_clues_path,
    read_canonical_clues,
    read_canonical_markdown,
    read_canonical_papers,
    write_canonical_clues,
    write_canonical_papers,
)
from dataset_generation.interpretations import InterpretationRecord, write_interpretations, read_interpretations
from dataset_generation.query_generation import (
    MemoryComponent,
    QueryGenerationConfig,
    _split_component_text,
    generate_query_collections,
    select_components,
)
from dataset_generation.preprocessed import append_clue_row, textual_clue_path, visual_clue_path
from dataset_generation.storage import read_dataset_artifact, read_stats, write_dataset_artifact
from dataset_generation.synthetic import (
    SyntheticCollectionConfig,
    TestCollectionExample as QueryExample,
    generate_synthetic_queries,
    write_test_collection,
)
from dataset_generation.textual_clue_descriptions import (
    apply_text_chat_template,
    default_dataset_dir,
    format_prompt,
    generate_with_mlx,
    strip_thinking,
)
from dataset_generation.vl_figure_descriptions import (
    safe_stem,
    samples_from_paths,
)


def _write_raw_jsonl(rows: list[dict], path):
    """Write raw JSONL without canonical validation — for testing legacy input ingestion."""
    from pathlib import Path
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


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
            "image_width": 640,
            "image_height": 480,
            "image_mode": "RGB",
            "source_fig_dataset": "mineru_pdf_extraction",
            "source_fig_split": "train",
            "source_paper_dataset": "acl_subset",
            "source_paper_config": "official_acl_anthology_xml",
            "source_paper_split": "curated",
        },
        {
            "record_id": "fig-2",
            "filename": "w14-33.pdf-Figure1.png",
            "extracted_paper_id": "w14-33",
            "normalized_paper_id": "w14-33",
            "resolved_paper_id": "w14-3300",
            "label": "table",
            "markdown": "# Fallback match",
            "match_status": "matched",
            "image_width": None,
            "image_height": None,
            "image_mode": None,
            "source_fig_dataset": "mineru_pdf_extraction",
            "source_fig_split": "train",
            "source_paper_dataset": "acl_subset",
            "source_paper_config": "official_acl_anthology_xml",
            "source_paper_split": "curated",
        },
    ]


def write_preprocessed_paper(
    root,
    paper_id: str,
    *,
    markdown: str = "# Paper",
    figure_id: str = "figure-0001",
):
    paper_dir = root / paper_id
    image_dir = paper_dir / "mineru" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / f"{figure_id}.png"
    image_path.write_bytes(f"{paper_id}-{figure_id}".encode("utf-8"))
    (paper_dir / "markdown.md").write_text(markdown, encoding="utf-8")
    (paper_dir / "figures.json").write_text("[]", encoding="utf-8")
    (paper_dir / "paper.json").write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "figures": [
                    {
                        "figure_id": figure_id,
                        "filename": f"{paper_id}-Figure1.png",
                        "image_relpath": f"mineru/images/{figure_id}.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paper_dir


def test_write_and_read_jsonl_artifact(tmp_path):
    records = fixture_records()
    stats = {"matched_records": 2}
    metadata = {"source": "fixture"}
    output = write_dataset_artifact(records, tmp_path / "canonical_fixture", metadata, stats, "jsonl")

    frame = read_dataset_artifact(output)
    loaded_stats = read_stats(output)

    assert list(frame["record_id"]) == ["fig-1", "fig-2"]
    assert loaded_stats["matched_records"] == 2
    assert (output / "metadata.json").exists()


def test_synthetic_collection_is_deterministic_and_writable(tmp_path):
    records = fixture_records()
    config = SyntheticCollectionConfig(seed=7, max_examples=1, negative_examples_per_query=1)
    collection = generate_synthetic_queries(records, config)
    repeat = generate_synthetic_queries(records, config)

    assert collection == repeat
    assert collection[0].query_id == "q00000"
    assert collection[0].relevant_ids

    output = write_test_collection(collection, tmp_path / "test_collection")
    lines = (output / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["metadata"]["seed"] == 7


def test_interpretations_are_sidecar_artifacts(tmp_path):
    output = write_interpretations(
        [
            InterpretationRecord(
                record_id="fig-1",
                kind="visual",
                text="A bar chart with labeled axes.",
                model="test-model",
                prompt_id="visual_interpretation",
                prompt_version="v1",
            )
        ],
        tmp_path / "interpretations" / "run-1",
        {"base_dataset": "data/processed/mineru_pdf_extraction/papers"},
    )

    rows = read_interpretations(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert rows[0]["record_id"] == "fig-1"
    assert rows[0]["kind"] == "visual"
    assert metadata["record_count"] == 1


def test_vl_figure_description_helpers_use_local_paths(tmp_path):
    image_path = tmp_path / "figure one!.png"
    image_path.write_bytes(b"not-an-image-but-existing-path")

    samples = samples_from_paths([image_path], limit=1)

    assert safe_stem("figure one!.png", "fallback") == "figure_one"
    assert samples[0].record_id == "local-1"
    assert samples[0].image_path == image_path.resolve()


def test_textual_prompt_preserves_literal_json_braces():
    prompt = format_prompt(
        """
{
  "cue": "...",
  "category": "..."
}

PAPER:
{paper_text}
""",
        "# Paper markdown",
    )

    assert '"cue": "..."' in prompt
    assert "{paper_text}" not in prompt
    assert prompt.endswith("# Paper markdown\n")


def canonical_source_rows(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    image_a.write_bytes(b"same-image")
    image_b.write_bytes(b"same-image")
    image_c.write_bytes(b"different-image")
    return [
        {
            "record_id": "row-1",
            "filename": "paper-1-Figure1.png",
            "extracted_paper_id": "paper-1",
            "normalized_paper_id": "paper-1",
            "resolved_paper_id": "paper-1",
            "label": "chart",
            "markdown": "# Paper 1",
            "match_status": "matched",
            "source_fig_dataset": "test",
            "source_fig_split": "train",
            "source_paper_dataset": "test-md",
            "source_paper_config": "fulltext",
            "source_paper_split": "train",
            "image_path": str(image_a),
        },
        {
            "record_id": "row-2",
            "filename": "paper-1-Figure1-duplicate.png",
            "extracted_paper_id": "paper-1",
            "normalized_paper_id": "paper-1",
            "resolved_paper_id": "paper-1",
            "label": "chart",
            "markdown": "# Paper 1",
            "match_status": "matched",
            "source_fig_dataset": "test",
            "source_fig_split": "train",
            "source_paper_dataset": "test-md",
            "source_paper_config": "fulltext",
            "source_paper_split": "train",
            "image_path": str(image_b),
        },
        {
            "record_id": "row-3",
            "filename": "paper-1-Figure2.png",
            "extracted_paper_id": "paper-1",
            "normalized_paper_id": "paper-1",
            "resolved_paper_id": "paper-1",
            "label": "table",
            "markdown": "# Paper 1",
            "match_status": "matched",
            "source_fig_dataset": "test",
            "source_fig_split": "train",
            "source_paper_dataset": "test-md",
            "source_paper_config": "fulltext",
            "source_paper_split": "train",
            "image_path": str(image_c),
        },
    ]


def write_processed_rows(tmp_path, rows):
    stats = {"matched_records": len(rows), "match_rate": 1.0}
    metadata = {"test": True}
    return write_dataset_artifact(rows, tmp_path / "processed", metadata, stats, "jsonl")


def test_canonical_duplicate_source_rows_with_identical_images_collapse(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))

    papers, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")

    assert len(papers) == 1
    assert list(papers[0].keys())[:3] == ["paper_id", "markdown_relpath", "markdown_sha256"]
    assert "markdown" not in papers[0]
    assert read_canonical_markdown(tmp_path / "canonical", papers[0]) == "# Paper 1"
    assert len(papers[0]["figures"]) == 2
    assert report["duplicate_figure_rows_collapsed"] == 1
    collapsed = [figure for figure in papers[0]["figures"] if len(figure["source_rows"]) == 2][0]
    assert collapsed["legacy_record_ids"] == ["row-1", "row-2"]


def test_canonical_different_figures_from_same_paper_remain_distinct(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))

    papers, _ = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")

    image_hashes = {figure["image_sha256"] for figure in papers[0]["figures"]}
    assert len(image_hashes) == 2


def test_canonical_textual_clues_are_associated_once_at_paper_level(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    textual = write_interpretations(
        [
            InterpretationRecord(
                record_id="row-1",
                kind="textual",
                text='{"cue":"paper topic"}',
                model="m",
                prompt_id="p",
                prompt_version="v1",
                metadata={"resolved_paper_id": "paper-1", "filename": "figure.png", "image_path": "/bad/image.png"},
            ),
            InterpretationRecord(
                record_id="row-2",
                kind="textual",
                text='{"cue":"paper topic"}',
                model="m",
                prompt_id="p",
                prompt_version="v1",
                metadata={"resolved_paper_id": "paper-1"},
            ),
        ],
        tmp_path / "textual",
    )

    papers, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", textual_interpretations=textual)
    clues = read_canonical_clues(canonical_textual_clues_path(tmp_path / "canonical"), kind="textual")

    assert "textual_clues" not in papers[0]
    assert len(clues) == 1
    assert list(clues[0]) == ["paper_id", "kind", "model", "prompt_id", "prompt_version", "output"]
    assert clues[0]["paper_id"] == "paper-1"
    assert clues[0]["output"] == '{"cue":"paper topic"}'
    assert "filename" not in json.dumps(clues[0])
    assert "image_path" not in json.dumps(clues[0])
    assert "record_id" not in clues[0]
    assert "text" not in clues[0]
    assert "provenance" not in clues[0]
    assert report["textual_clues_recovered"] == 1
    assert report["textual_duplicates_collapsed"] == 1


def test_old_textual_clues_migrate_with_distinct_generations_preserved(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    textual = write_interpretations(
        [
            InterpretationRecord("row-1", "textual", "topic", "m", "p", "v1", {"resolved_paper_id": "paper-1"}),
            InterpretationRecord("row-2", "textual", "method", "m", "p", "v1", {"resolved_paper_id": "paper-1"}),
        ],
        tmp_path / "textual",
    )

    _, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", textual_interpretations=textual)
    clues = read_canonical_clues(canonical_textual_clues_path(tmp_path / "canonical"), kind="textual")

    assert len(clues) == 1
    assert clues[0]["paper_id"] == "paper-1"
    assert clues[0]["output"] in {"method", "topic"}
    assert "provenance" not in clues[0]
    assert report["textual_clue_records_matched"] == 2
    assert report["textual_clues_recovered"] == 1
    assert report["textual_duplicate_paper_entries_collapsed"] == 1


def test_deterministic_visual_clue_recovery_uses_record_id(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    visual = write_interpretations(
        [InterpretationRecord("row-3", "visual", '{"figure_type":["table"]}', "m", "p", "v1", {})],
        tmp_path / "visual",
    )

    papers, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", visual_interpretations=visual)
    clues = read_canonical_clues(canonical_visual_clues_path(tmp_path / "canonical"), kind="visual")

    assert len(clues) == 1
    assert list(clues[0]) == ["paper_id", "figure_id", "kind", "model", "prompt_id", "prompt_version", "output"]
    recovered = [figure for figure in papers[0]["figures"] if figure["figure_id"] == clues[0]["figure_id"]]
    assert recovered[0]["legacy_record_ids"] == ["row-3"]
    assert "record_id" not in clues[0]
    assert "text" not in clues[0]
    assert "provenance" not in clues[0]
    assert report["visual_clues_recovered"] == 1


def test_unresolved_visual_clues_remain_missing_without_fuzzy_matching(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    visual = write_interpretations(
        [
            InterpretationRecord(
                "unknown",
                "visual",
                "looks like figure one",
                "m",
                "p",
                "v1",
                {"filename": "paper-1-Figure1.png"},
            )
        ],
        tmp_path / "visual",
    )

    papers, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", visual_interpretations=visual)
    clues = read_canonical_clues(canonical_visual_clues_path(tmp_path / "canonical"), kind="visual")

    assert report["visual_clues_unresolved"] == 1
    assert clues == []
    assert all("visual_clues" not in figure for figure in papers[0]["figures"])


def test_query_generation_is_one_query_per_paper_with_paper_relevance(tmp_path):
    dataset = tmp_path / "processed" / "mineru_pdf_extraction" / "papers"
    clues = tmp_path / "clues"
    write_preprocessed_paper(dataset, "paper-1")
    append_clue_row(
        textual_clue_path(clues, "paper-1"),
        {"paper_id": "paper-1", "kind": "textual", "model": "m", "prompt_id": "p", "prompt_version": "v1", "output": '{"cue":"topic"}'},
    )
    append_clue_row(
        visual_clue_path(clues, "paper-1", "figure-0001"),
        {"paper_id": "paper-1", "figure_id": "figure-0001", "kind": "visual", "model": "m", "prompt_id": "p", "prompt_version": "v1", "output": '{"figure_type":["line chart"]}'},
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{selected_cues}", encoding="utf-8")
    output = generate_query_collections(
        QueryGenerationConfig(
            dataset=dataset,
            visual_interpretations=tmp_path / "unused",
            textual_interpretations=None,
            clues_dir=clues,
            output_dir=tmp_path / "queries",
            prompt=prompt,
            visual_component_budget=1,
            textual_component_budget=1,
            allow_partial_components=False,
        )
    )

    visual_lines = (output / "visual_only" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    vt_lines = (output / "visual_and_text" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(visual_lines) == 1
    assert len(vt_lines) == 1
    assert json.loads(visual_lines[0])["relevant_ids"] == ["paper-1"]
    assert json.loads(vt_lines[0])["relevant_ids"] == ["paper-1"]


def test_query_generation_parses_prefixed_empty_visual_json_without_raw_fallback():
    text = """Figure 2: {
  "figure_type": [],
  "layout": [],
  "visual_elements": [],
  "colors_and_styles": [],
  "relationships": [],
  "distinctive_visual_cues": []
}"""

    assert _split_component_text(text, "visual") == []


def test_query_generation_resume_skips_existing_queries_and_avoids_clue_duplicates(tmp_path):
    dataset = tmp_path / "processed" / "mineru_pdf_extraction" / "papers"
    clues = tmp_path / "clues"
    write_preprocessed_paper(dataset, "paper-1")
    write_preprocessed_paper(dataset, "paper-2")
    for paper_id in ("paper-1", "paper-2"):
        append_clue_row(
            visual_clue_path(clues, paper_id, "figure-0001"),
            {
                "paper_id": paper_id,
                "figure_id": "figure-0001",
                "kind": "visual",
                "model": "m",
                "prompt_id": "p",
                "prompt_version": "v1",
                "output": '{"figure_type":["line chart"]}',
            },
        )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("{selected_cues}", encoding="utf-8")
    existing = QueryExample(
        query_id="visual_only_q00000",
        query="existing query",
        relevant_ids=["paper-1"],
        metadata={"mode": "visual-only", "paper_id": "paper-1"},
    )
    collection_dir = tmp_path / "queries" / "query_generation_default" / "visual_only"
    write_test_collection([existing], collection_dir)
    (clues / "queries.jsonl").write_text(json.dumps(asdict(existing)) + "\n", encoding="utf-8")

    output = generate_query_collections(
        QueryGenerationConfig(
            dataset=dataset,
            visual_interpretations=None,
            textual_interpretations=None,
            clues_dir=clues,
            output_dir=tmp_path / "queries",
            prompt=prompt,
            modes=("visual-only",),
            visual_component_budget=1,
            resume=True,
        )
    )

    collection_rows = [
        json.loads(line)
        for line in (output / "visual_only" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    clue_query_rows = [
        json.loads(line)
        for line in (clues / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["paper_id"] for row in collection_rows] == ["paper-1", "paper-2"]
    assert [row["metadata"]["paper_id"] for row in clue_query_rows] == ["paper-1", "paper-2"]
    assert collection_rows[0]["query"] == "existing query"


def test_canonical_serialization_deserialization_is_stable(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    papers, _ = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")

    first = (tmp_path / "canonical" / "papers.jsonl").read_text(encoding="utf-8")
    assert first.startswith('{"paper_id":')
    loaded = read_canonical_papers(tmp_path / "canonical")
    write_canonical_papers(loaded, tmp_path / "canonical")
    second = (tmp_path / "canonical" / "papers.jsonl").read_text(encoding="utf-8")

    assert loaded == papers
    assert first == second
    assert "markdown" not in loaded[0]
    assert read_canonical_markdown(tmp_path / "canonical", loaded[0]) == "# Paper 1"


def test_align_canonical_clues_rewrites_mixed_inputs_to_canonical_ids(tmp_path):
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    papers, _ = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")
    duplicate_text = {"record_id": "row-1", "kind": "textual", "text": "topic", "model": "m", "prompt_id": "p", "prompt_version": "v1"}
    stale_visual = {
        "record_id": "legacy-unknown",
        "kind": "visual",
        "text": "visual clue",
        "model": "m",
        "prompt_id": "p",
        "prompt_version": "v1",
        "metadata": {"dataset_row": 2},
    }
    unresolved_visual = {
        "record_id": "unknown",
        "kind": "visual",
        "text": "filename-only should not match",
        "model": "m",
        "prompt_id": "p",
        "prompt_version": "v1",
        "metadata": {"filename": "paper-1-Figure2.png"},
    }
    _write_raw_jsonl(
        [
            duplicate_text,
            {**duplicate_text, "provenance": {"legacy_record_id": "row-2"}},
            {"paper_id": "missing", "record_id": "missing", "kind": "textual", "text": "bad", "model": "m", "prompt_id": "p", "prompt_version": "v1"},
        ],
        tmp_path / "textual_mixed.jsonl",
    )
    _write_raw_jsonl([stale_visual, unresolved_visual], tmp_path / "visual_mixed.jsonl")

    report = align_canonical_clues(
        tmp_path / "canonical",
        textual_inputs=[tmp_path / "textual_mixed.jsonl"],
        visual_inputs=[tmp_path / "visual_mixed.jsonl"],
    )
    textual = read_canonical_clues(canonical_textual_clues_path(tmp_path / "canonical"), kind="textual")
    visual = read_canonical_clues(canonical_visual_clues_path(tmp_path / "canonical"), kind="visual")

    assert len(textual) == 1
    assert list(textual[0]) == ["paper_id", "kind", "model", "prompt_id", "prompt_version", "output"]
    assert textual[0]["paper_id"] == "paper-1"
    assert textual[0]["output"] == "topic"
    assert "record_id" not in textual[0]
    assert "text" not in textual[0]
    assert "provenance" not in textual[0]
    assert report["textual_duplicates_collapsed"] == 1
    assert report["textual_clues_unresolved"] == 1
    assert len(visual) == 1
    assert list(visual[0]) == ["paper_id", "figure_id", "kind", "model", "prompt_id", "prompt_version", "output"]
    expected_figure = [figure for figure in papers[0]["figures"] if figure["source_rows"] == [2]][0]
    assert visual[0]["figure_id"] == expected_figure["figure_id"]
    assert visual[0]["paper_id"] == "paper-1"
    assert "record_id" not in visual[0]
    assert "text" not in visual[0]
    assert "provenance" not in visual[0]
    assert report["visual_clues_unresolved"] == 1


def test_textual_mlx_generation_uses_sampler_keyword(monkeypatch):
    calls = {}

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["template_messages"] = messages
            calls["template_kwargs"] = kwargs
            return "templated prompt"

    def fake_generate(model, tokenizer, **kwargs):
        calls["kwargs"] = kwargs
        return "<think>hidden reasoning</think>\ngenerated"

    def fake_make_sampler(**kwargs):
        calls["sampler_kwargs"] = kwargs
        return "sampler"

    mlx_lm.generate = fake_generate
    sample_utils.make_sampler = fake_make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    output = generate_with_mlx(
        {"model": "model", "tokenizer": FakeTokenizer()},
        prompt="Prompt",
        max_tokens=10,
        temperature=0.2,
    )

    assert output == "generated"
    assert calls["template_kwargs"]["enable_thinking"] is False
    assert calls["sampler_kwargs"] == {"temp": 0.2}
    assert calls["kwargs"]["prompt"] == "templated prompt"
    assert calls["kwargs"]["sampler"] == "sampler"
    assert "temp" not in calls["kwargs"]


def test_text_chat_template_falls_back_without_thinking_kwarg():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unsupported keyword")
            return messages[0]["content"]

    assert apply_text_chat_template(FakeTokenizer(), "Prompt") == "Prompt"


def test_strip_thinking_removes_hidden_reasoning():
    assert strip_thinking("<think>hidden\nreasoning</think>\n[]") == "[]"


def test_textual_clue_default_dataset_dir_matches_data_dir_layout(tmp_path):
    assert default_dataset_dir(tmp_path / "data", "train") == tmp_path / "data" / "preprocessed"





def test_query_component_selection_respects_mode_limits():
    components = [
        *(MemoryComponent(record_id=f"fig-{index}", kind="visual", text=f"visual {index}") for index in range(12)),
        *(MemoryComponent(record_id=f"text-{index}", kind="textual", text=f"text {index}") for index in range(6)),
    ]

    visual_only = select_components(components, mode="visual-only", seed=7, source_record_id="fig-1")
    mixed = select_components(components, mode="visual-and-text", seed=7, source_record_id="fig-1")

    assert len(visual_only) == 3
    assert {component.kind for component in visual_only} == {"visual"}
    assert len(mixed) == 6
    assert sum(component.kind == "visual" for component in mixed) == 3
    assert sum(component.kind == "textual" for component in mixed) == 3


def test_visual_clue_recovery_by_source_row(tmp_path):
    """Visual clues recover via source_rows provenance in papers.jsonl."""
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    papers, _ = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")
    # Find figure with source_row 2 (row-3 -> different-image)
    target_figure = [fig for fig in papers[0]["figures"] if 2 in fig["source_rows"]][0]
    # Write a visual clue that references via dataset_row metadata
    visual = write_interpretations(
        [InterpretationRecord("unknown-id", "visual", '{"figure_type":["table"]}', "m", "p", "v1", {"dataset_row": 2})],
        tmp_path / "visual",
    )

    _, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", visual_interpretations=visual)
    clues = read_canonical_clues(canonical_visual_clues_path(tmp_path / "canonical"), kind="visual")

    assert len(clues) == 1
    assert clues[0]["figure_id"] == target_figure["figure_id"]
    assert clues[0]["paper_id"] == "paper-1"
    assert report["visual_clues_recovered"] == 1


def test_visual_clue_recovery_by_image_sha256(tmp_path):
    """Visual clues recover via exact image SHA256 when it maps to exactly one figure."""
    processed = write_processed_rows(tmp_path, canonical_source_rows(tmp_path))
    papers, _ = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical")
    # Use the figure from the 'different-image' bytes (row-3)
    target_figure = [fig for fig in papers[0]["figures"] if 2 in fig["source_rows"]][0]
    image_sha = target_figure["image_sha256"]
    # Write a visual clue with only image_sha256 — no record_id, no dataset_row
    visual = write_interpretations(
        [InterpretationRecord("totally-unknown", "visual", '{"figure_type":["unique"]}', "m", "p", "v1", {"image_sha256": image_sha})],
        tmp_path / "visual",
    )

    _, report = prepare_canonical_papers_from_rows(processed, tmp_path / "canonical", visual_interpretations=visual)
    clues = read_canonical_clues(canonical_visual_clues_path(tmp_path / "canonical"), kind="visual")

    assert len(clues) == 1
    assert clues[0]["figure_id"] == target_figure["figure_id"]
    assert report["visual_clues_recovered"] == 1
