from __future__ import annotations

from pathlib import Path

from dataset_generation.interpretations import InterpretationRecord
from dataset_generation.preprocessed import read_clue_rows
from dataset_generation.textual_clue_descriptions import _append_textual_clue
from dataset_generation.vl_figure_descriptions import _append_visual_clue


def test_visual_clue_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    old_record = InterpretationRecord(
        record_id="fig-1",
        kind="visual",
        text="old",
        model="model",
        prompt_id="visual_interpretation",
        prompt_version="v1",
        metadata={"paper_id": "paper-1", "figure_id": "fig-1"},
    )
    new_record = InterpretationRecord(
        record_id="fig-1",
        kind="visual",
        text="new",
        model="model",
        prompt_id="visual_interpretation",
        prompt_version="v2",
        metadata={"paper_id": "paper-1", "figure_id": "fig-1"},
    )

    clue_path = _append_visual_clue(old_record, tmp_path)
    _append_visual_clue(new_record, tmp_path, overwrite=True)

    assert read_clue_rows(clue_path) == [
        {
            "paper_id": "paper-1",
            "figure_id": "fig-1",
            "kind": "visual",
            "model": "model",
            "prompt_id": "visual_interpretation",
            "prompt_version": "v2",
            "output": "new",
        }
    ]


def test_textual_clue_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    old_record = InterpretationRecord(
        record_id="paper-1",
        kind="textual",
        text="old",
        model="model",
        prompt_id="textual_interpretation",
        prompt_version="v1",
    )
    new_record = InterpretationRecord(
        record_id="paper-1",
        kind="textual",
        text="new",
        model="model",
        prompt_id="textual_interpretation",
        prompt_version="v2",
    )

    clue_path = _append_textual_clue(old_record, tmp_path)
    _append_textual_clue(new_record, tmp_path, overwrite=True)

    assert read_clue_rows(clue_path) == [
        {
            "paper_id": "paper-1",
            "kind": "textual",
            "model": "model",
            "prompt_id": "textual_interpretation",
            "prompt_version": "v2",
            "output": "new",
        }
    ]
