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
