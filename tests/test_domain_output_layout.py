from __future__ import annotations

import json
from pathlib import Path

from preprocessing.preprocessed import append_clue_row, textual_clue_path, visual_clue_path
from queries import judge_queries, query_generation
from queries.synthetic import TestCollectionExample as QueryExample
from tests.test_managed_query_settings import write_settings


def _paper(root: Path, domain: str) -> dict[str, str]:
    paper_id = f"{domain}_paper"
    paper_dir = root / domain / paper_id
    paper_dir.mkdir(parents=True)
    (paper_dir / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    record = {
        "paper_id": paper_id,
        "source_paper_dataset": domain,
        "split": "test",
        "figures": [{"figure_id": f"{paper_id}_figure", "image_relpath": "images/figure.jpg"}],
    }
    (paper_dir / "paper.json").write_text(json.dumps(record), encoding="utf-8")
    (paper_dir / "images").mkdir()
    (paper_dir / "images" / "figure.jpg").write_bytes(b"image")
    return record


def test_query_generation_reads_domain_clues_and_writes_domain_collections(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "preprocessed"
    clues = tmp_path / "clues"
    for domain in ("ACL", "Biology"):
        paper = _paper(dataset, domain)
        paper_id = paper["paper_id"]
        append_clue_row(
            textual_clue_path(clues, paper_id, domain),
            {"paper_id": paper_id, "kind": "textual", "output": "Text clue."},
        )
        append_clue_row(
            visual_clue_path(clues, paper_id, f"{paper_id}_figure", domain),
            {"paper_id": paper_id, "figure_id": f"{paper_id}_figure", "kind": "visual", "output": "Visual clue."},
        )
    config = query_generation.QueryGenerationConfig(
        dataset=dataset,
        visual_interpretations=None,
        textual_interpretations=None,
        clues_dir=clues,
        output_dir=tmp_path / "query_collections",
        modes=("visual-only",),
        collection_id="query_generation_test",
    )
    components = query_generation._components_by_paper(
        dataset, clues_dir=clues, visual_clues_path=None, textual_clues_path=None
    )
    assert all({component.kind for component in rows} == {"textual", "visual"} for rows in components.values())

    remaining = []

    def fake_mode(mode, papers, components_by_paper, mode_config, *args, **kwargs):
        paper_id = papers[0][1]["paper_id"]
        remaining.append(mode_config.max_examples)
        assert {component.kind for component in components_by_paper[paper_id]} == {"textual", "visual"}
        assert kwargs["root_query_path"] == clues / papers[0][1]["source_paper_dataset"] / "queries.jsonl"
        return [QueryExample(f"{paper_id}_query", "query", [paper_id], metadata={"paper_id": paper_id})]

    monkeypatch.setattr(query_generation, "_generate_mode_examples_from_papers", fake_mode)
    query_generation._generate_query_collections_from_preprocessed(config, "prompt", "hash", None, None, limit=2)

    assert remaining == [2, 1]
    for domain in ("ACL", "Biology"):
        path = config.output_dir / domain / config.collection_id / "visual_only" / "queries.jsonl"
        assert path.exists()
        assert json.loads(path.read_text().splitlines()[0])["relevant_ids"] == [f"{domain}_paper"]
    assert not (config.output_dir / config.collection_id).exists()


def test_query_generation_limits_images_before_clue_coverage_and_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "preprocessed"
    clues = tmp_path / "clues"
    paper = _paper(dataset, "ACL")
    paper_id = paper["paper_id"]
    paper_dir = dataset / "ACL" / paper_id
    paper["figures"].append({"figure_id": "second", "image_relpath": "images/second.jpg"})
    (paper_dir / "paper.json").write_text(json.dumps(paper), encoding="utf-8")
    (paper_dir / "images" / "second.jpg").write_bytes(b"image")
    append_clue_row(textual_clue_path(clues, paper_id, "ACL"),
                    {"paper_id": paper_id, "kind": "textual", "output": "Text clue."})
    append_clue_row(visual_clue_path(clues, paper_id, f"{paper_id}_figure", "ACL"),
                    {"paper_id": paper_id, "figure_id": f"{paper_id}_figure",
                     "kind": "visual", "output": "First visual clue."})

    config = query_generation.QueryGenerationConfig(
        dataset=dataset, visual_interpretations=None, textual_interpretations=None,
        clues_dir=clues, output_dir=tmp_path / "queries", modes=("visual-only",),
        max_images=1,
    )
    query_generation._generate_query_collections_from_preprocessed(config, "{selected_cues}", "hash", None, None)

    path = config.output_dir / "ACL" / config.collection_id / "visual_only" / "queries.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    assert row["metadata"]["selected_visual_count"] == 1
    assert row["metadata"]["max_images"] == 1
    assert "First visual clue" in row["query"]


def test_managed_judgement_reads_and_writes_each_domain_collection(tmp_path: Path, monkeypatch) -> None:
    settings = write_settings(tmp_path)
    dataset = tmp_path / "data" / "preprocessed"
    output = tmp_path / "data" / "query_collections"
    for domain in ("ACL", "Biology"):
        paper = _paper(dataset, domain)
        query_dir = output / domain / "managed_test" / "visual_only"
        query_dir.mkdir(parents=True)
        (query_dir / "queries.jsonl").write_text(
            json.dumps({
                "query_id": f"{domain}_query",
                "query": "query",
                "relevant_ids": [paper["paper_id"]],
                "metadata": {"paper_id": paper["paper_id"], "selected_components": []},
            }) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(judge_queries, "load_query_judge", lambda config: None)
    monkeypatch.setattr(judge_queries, "judge_query", lambda *args: {"score": 1})
    args = judge_queries.build_parser().parse_args(["--settings", str(settings), "--set", "test"])
    assert judge_queries.main(args) == 0

    for domain in ("ACL", "Biology"):
        path = output / domain / "managed_test" / "visual_only" / "query_judgements.jsonl"
        assert json.loads(path.read_text().splitlines()[0])["paper_id"] == f"{domain}_paper"


def test_judgement_reads_staged_paper_with_pdf_split_index(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    paper_dir = data_root / "preprocessed" / "Physics" / "Physics_paper"
    (paper_dir / "images").mkdir(parents=True)
    (paper_dir / "markdown.md").write_text("# Staged paper\n", encoding="utf-8")
    (paper_dir / "images" / "figure.png").write_bytes(b"image")
    split_index = data_root / "splits" / "index.json"
    split_index.parent.mkdir(parents=True)
    split_index.write_text(json.dumps({"train": ["Physics/paper.pdf"]}), encoding="utf-8")
    query_dir = data_root / "query_collections" / "Physics" / "query_generation_train" / "visual_only"
    query_dir.mkdir(parents=True)
    (query_dir / "queries.jsonl").write_text(json.dumps({
        "query_id": "q1", "query": "Find the figure", "relevant_ids": ["Physics_paper"],
        "metadata": {"paper_id": "Physics_paper", "selected_components": []},
    }) + "\n", encoding="utf-8")

    seen = []
    monkeypatch.setattr(judge_queries, "load_query_judge", lambda config: None)

    def fake_judge(mode, paper, selected, query, config, backend):
        seen.append((paper["markdown_path"], paper["figures"][0]["image_path"]))
        return {"grounded": True}

    monkeypatch.setattr(judge_queries, "judge_query", fake_judge)
    args = judge_queries.build_parser().parse_args([
        "--data-dir", str(data_root), "--split-index", str(split_index), "--mode", "visual-only",
    ])
    assert judge_queries.main(args) == 0
    assert seen == [(str(paper_dir / "markdown.md"), str((paper_dir / "images" / "figure.png").resolve()))]
    judgement_path = query_dir / "query_judgements.jsonl"
    assert json.loads(judgement_path.read_text(encoding="utf-8").splitlines()[0])["judgement"]["grounded"] is True
