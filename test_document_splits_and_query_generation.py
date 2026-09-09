from __future__ import annotations

import json
from pathlib import Path

from dataset_generation.document_splits import build_split_index, write_split_index
from dataset_generation.preprocessed import paper_clue_dir, read_preprocessed_papers
from dataset_generation.query_generation import QueryGenerationConfig, generate_query_collections


def test_read_preprocessed_papers_accepts_minimal_layout(tmp_path: Path) -> None:
    paper_dir = tmp_path / "preprocessed" / "2023.acl-long.3"
    images_dir = paper_dir / "images"
    images_dir.mkdir(parents=True)
    (paper_dir / "markdown.md").write_text("# paper", encoding="utf-8")
    (paper_dir / "2023.acl-long.3.pdf").write_bytes(b"%PDF")
    (images_dir / "figure-a.jpg").write_bytes(b"jpg")
    (images_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    papers = read_preprocessed_papers(tmp_path / "preprocessed")

    assert len(papers) == 1
    assert papers[0]["paper_id"] == "2023.acl-long.3"
    assert papers[0]["venue"] == "ACL"
    assert papers[0]["year"] == 2023
    assert papers[0]["volume_id"] == "2023.acl-long"
    assert papers[0]["figures"][0]["figure_id"] == "figure-a"
    assert papers[0]["figures"][0]["image_path"].endswith("figure-a.jpg")


def test_build_split_index_is_seeded_and_stratified() -> None:
    papers = [
        {"paper_id": f"acl-{index}", "venue": "ACL", "year": 2025}
        for index in range(10)
    ] + [
        {"paper_id": f"emnlp-{index}", "venue": "EMNLP", "year": 2025}
        for index in range(10)
    ]

    first = build_split_index(papers, test_fraction=0.2, seed=42)
    second = build_split_index(papers, test_fraction=0.2, seed=42)

    assert first.train == second.train
    assert first.test == second.test
    assert first.metadata["counts_by_stratum"]["ACL:2025"]["test"] == 2
    assert first.metadata["counts_by_stratum"]["EMNLP:2025"]["test"] == 2
    assert set(first.train).isdisjoint(first.test)


def test_build_split_index_supports_fixed_train_test_sizes() -> None:
    papers = [
        {"paper_id": f"acl-{index}", "venue": "ACL", "year": 2025}
        for index in range(5000)
    ] + [
        {"paper_id": f"emnlp-{index}", "venue": "EMNLP", "year": 2025}
        for index in range(5000)
    ]

    split = build_split_index(papers, train_size=1000, test_size=200, test_fraction=None, seed=42)

    assert len(split.train) == 1000
    assert len(split.test) == 200
    assert split.metadata["train_size"] == 1000
    assert split.metadata["test_size"] == 200
    assert split.metadata["counts_by_stratum"]["ACL:2025"]["train"] == 500
    assert split.metadata["counts_by_stratum"]["ACL:2025"]["test"] == 100
    assert split.metadata["counts_by_stratum"]["EMNLP:2025"]["train"] == 500
    assert split.metadata["counts_by_stratum"]["EMNLP:2025"]["test"] == 100
    assert set(split.train).isdisjoint(split.test)


def test_query_generation_uses_split_index_for_stable_resume(tmp_path: Path) -> None:
    dataset = tmp_path / "preprocessed" / "papers"
    clues_dir = tmp_path / "clues"
    output_dir = tmp_path / "query_collections"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Use {selected_cues}", encoding="utf-8")

    for index in range(3):
        paper_id = f"paper-{index}"
        paper_dir = dataset / paper_id
        paper_dir.mkdir(parents=True)
        (paper_dir / "markdown.md").write_text(f"# {paper_id}", encoding="utf-8")
        (paper_dir / "paper.json").write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "venue": "ACL",
                    "year": 2025,
                    "figures": [{"figure_id": f"fig-{index}", "image_relpath": f"fig-{index}.png"}],
                }
            ),
            encoding="utf-8",
        )
        clue_dir = paper_clue_dir(clues_dir, paper_id)
        (clue_dir / "base").mkdir(parents=True)
        (clue_dir / "base" / "textual_clues.jsonl").write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "kind": "textual",
                    "output": '[{"cue": "text cue"}]',
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (clue_dir / "images").mkdir(parents=True)
        (clue_dir / "images" / f"fig-{index}.jsonl").write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "figure_id": f"fig-{index}",
                    "kind": "visual",
                    "output": '[{"mark": "visual cue"}]',
                }
            )
            + "\n",
            encoding="utf-8",
        )

    split_path = write_split_index(
        build_split_index(
            [
                {"paper_id": "paper-0", "venue": "ACL", "year": 2025},
                {"paper_id": "paper-1", "venue": "ACL", "year": 2025},
                {"paper_id": "paper-2", "venue": "ACL", "year": 2025},
            ],
            test_fraction=1 / 3,
            seed=42,
        ),
        tmp_path / "split.json",
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["test"] = ["paper-2", "paper-0"]
    split_path.write_text(json.dumps(split), encoding="utf-8")

    base_config = QueryGenerationConfig(
        dataset=tmp_path / "preprocessed",
        visual_interpretations=None,
        textual_interpretations=None,
        split_index=split_path,
        split_name="test",
        clues_dir=clues_dir,
        output_dir=output_dir,
        prompt=prompt,
        model_provider="null",
        model="null",
        modes=("visual-and-text",),
        visual_component_budget=1,
        textual_component_budget=1,
        max_examples=1,
        resume=False,
        collection_id="split_test",
    )

    generate_query_collections(base_config)
    first_rows = _read_jsonl(output_dir / "split_test" / "visual_and_text" / "queries.jsonl")
    assert [row["query_id"] for row in first_rows] == ["test_visual_and_text_q00000"]
    assert [row["metadata"]["paper_id"] for row in first_rows] == ["paper-2"]

    resumed_config = QueryGenerationConfig(
        **{
            **base_config.__dict__,
            "max_examples": 2,
            "resume": True,
        }
    )
    generate_query_collections(resumed_config)
    resumed_rows = _read_jsonl(output_dir / "split_test" / "visual_and_text" / "queries.jsonl")
    root_rows = _read_jsonl(clues_dir / "queries.jsonl")

    assert [row["query_id"] for row in resumed_rows] == [
        "test_visual_and_text_q00000",
        "test_visual_and_text_q00001",
    ]
    assert [row["metadata"]["paper_id"] for row in resumed_rows] == ["paper-2", "paper-0"]
    assert [row["query_id"] for row in root_rows] == [
        "test_visual_and_text_q00000",
        "test_visual_and_text_q00001",
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
