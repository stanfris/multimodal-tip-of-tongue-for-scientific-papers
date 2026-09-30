from __future__ import annotations

from pathlib import Path
import json

import pytest

from clues.interpretations import InterpretationRecord, interpretation_key
from inference import GenerationResult


class FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, requests):
        self.calls.append(requests)
        return [GenerationResult(request.request_id, f"output-{request.request_id}") for request in requests]


def test_visual_stage_batches_pending_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from clues import vl_figure_descriptions as stage

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe the figure")
    samples = [stage.FigureSample(f"figure-{index}", tmp_path / f"{index}.png",
                                  {"paper_id": "paper", "figure_id": f"figure-{index}"})
               for index in range(3)]
    completed = {interpretation_key(InterpretationRecord("figure-1", "visual", "",
                                                           "Qwen/Qwen3-VL-4B-Instruct", "visual_interpretation", "v1"))}
    backend = FakeBackend()
    monkeypatch.setattr(stage, "iter_samples_from_dataset", lambda *args, **kwargs: samples)
    monkeypatch.setattr(stage, "read_completed_interpretation_keys", lambda *args, **kwargs: completed)
    monkeypatch.setattr(stage, "read_preprocessed_papers", lambda *args: [])
    monkeypatch.setattr(stage, "load_backend", lambda *args, **kwargs: backend)
    args = stage.build_parser().parse_args(["--all", "--resume", "--backend", "vllm",
                                            "--batch-size", "3", "--prompt", str(prompt),
                                            "--output-dir", str(tmp_path / "out")])
    stage.run(args)
    assert len(backend.calls) == 1
    assert [request.request_id for request in backend.calls[0]] == ["0:figure-0", "1:figure-2"]
    assert (tmp_path / "out" / "visual_clues.metadata.json").exists()


def test_text_stage_batches_pending_and_handles_one_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from clues import textual_clue_descriptions as stage

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Text: {paper_text}")
    samples = [stage.TextSample(f"paper-{index}", f"markdown-{index}",
                                {"paper_id": f"paper-{index}"}) for index in range(3)]
    completed = {interpretation_key(InterpretationRecord("paper-0", "textual", "",
                                                           "Qwen/Qwen3-4B", "textual_interpretation", "v1"))}

    class FailingBackend(FakeBackend):
        def generate(self, requests):
            self.calls.append(requests)
            return [GenerationResult(request.request_id, error=ValueError("bad")) if "paper-1" in request.request_id
                    else GenerationResult(request.request_id, "good") for request in requests]

    backend = FailingBackend()
    monkeypatch.setattr(stage, "iter_samples_from_dataset", lambda *args, **kwargs: samples)
    monkeypatch.setattr(stage, "read_completed_interpretation_keys", lambda *args, **kwargs: completed)
    monkeypatch.setattr(stage, "read_preprocessed_papers", lambda *args: [])
    monkeypatch.setattr(stage, "load_backend", lambda *args, **kwargs: backend)
    args = stage.build_parser().parse_args(["--all", "--resume", "--backend", "vllm",
                                            "--batch-size", "3", "--prompt", str(prompt),
                                            "--output-dir", str(tmp_path / "out")])
    stage.run(args)
    assert len(backend.calls) == 1 and len(backend.calls[0]) == 2
    assert "Text: markdown-1" in [request.prompt for request in backend.calls[0]]
    assert (tmp_path / "out" / "textual_failures.jsonl").exists()


def test_query_generation_uses_batched_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from queries import query_generation as stage

    backend = FakeBackend()
    component = stage.MemoryComponent("clue", "textual", "a useful clue")
    papers = [(index, {"paper_id": f"paper-{index}"}) for index in range(3)]
    config = stage.QueryGenerationConfig(dataset=tmp_path, visual_interpretations=None,
        textual_interpretations=None, clues_dir=tmp_path, output_dir=tmp_path,
        model_provider="vllm", model="microsoft/phi-4", batch_size=2,
        allow_partial_components=True)
    monkeypatch.setattr(stage, "_has_complete_clue_coverage", lambda *args: True)
    monkeypatch.setattr(stage, "select_components", lambda components, mode: components)
    examples = stage._generate_mode_examples_from_papers(
        "visual-only", papers, {paper["paper_id"]: [component] for _, paper in papers},
        config, "{selected_cues}", backend, None)
    assert len(examples) == 3
    assert [len(call) for call in backend.calls] == [2, 1]
    assert [example.query for example in examples] == [f"output-{example.query_id}" for example in examples]


def test_judgement_uses_same_multimodal_backend(tmp_path: Path) -> None:
    from queries import query_generation as stage
    from PIL import Image

    prompt = tmp_path / "judge.txt"
    prompt.write_text("Judge {query} {figures}")

    class JudgeBackend(FakeBackend):
        def generate(self, requests):
            self.calls.append(requests)
            return [GenerationResult(request.request_id, '{"grounded": true}') for request in requests]

    backend = JudgeBackend()
    markdown = tmp_path / "paper.md"
    markdown.write_text("Paper content")
    Image.new("RGB", (4, 4)).save(tmp_path / "figure.png")
    config = stage.QueryGenerationConfig(dataset=tmp_path, visual_interpretations=None,
        textual_interpretations=None, clues_dir=tmp_path, output_dir=tmp_path,
        judge_queries=True, judgement_model_provider="vllm", judgement_prompt=prompt)
    paper = {"paper_id": "paper", "markdown_path": str(markdown),
             "figures": [{"image_path": str(tmp_path / "figure.png")}]}
    judgement = stage.judge_query("visual-only", paper, [], "query", config, backend)
    assert judgement["grounded"] is True
    assert backend.calls[0][0].images == (tmp_path / "figure.png",)


def test_judgement_stage_batches_and_preserves_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from queries import judge_queries as stage

    markdown = tmp_path / "paper.md"
    markdown.write_text("Paper content")
    papers = [{"paper_id": f"paper-{index}", "markdown_path": str(markdown),
               "figures": [], "split": "train"}
              for index in range(2)]
    query_dir = tmp_path / "queries" / "visual_only"
    query_dir.mkdir(parents=True)
    query_rows = [{"query_id": f"query-{index}", "query": f"find {index}",
                   "relevant_ids": [f"paper-{index}"], "metadata": {"paper_id": f"paper-{index}"}}
                  for index in range(2)]
    (query_dir / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in query_rows) + "\n")
    prompt = tmp_path / "judge.txt"
    prompt.write_text("Judge {query}")

    class JudgeBackend(FakeBackend):
        def generate(self, requests):
            self.calls.append(requests)
            return [GenerationResult(request.request_id, '{"grounded": true}') for request in requests]

    backend = JudgeBackend()
    monkeypatch.setattr(stage, "read_preprocessed_papers", lambda dataset: papers)
    monkeypatch.setattr(stage, "load_query_judge", lambda config: backend)
    args = stage.build_parser().parse_args(["--dataset", str(tmp_path), "--input-dir", str(tmp_path / "queries"),
                                            "--mode", "visual-only", "--prompt", str(prompt),
                                            "--provider", "vllm", "--batch-size", "2"])
    assert stage.main(args) == 0
    assert len(backend.calls) == 1 and [request.request_id for request in backend.calls[0]] == ["query-0", "query-1"]
    results = [json.loads(line) for line in (query_dir / "query_judgements.jsonl").read_text().splitlines()]
    assert [row["query_id"] for row in results] == ["query-0", "query-1"]
