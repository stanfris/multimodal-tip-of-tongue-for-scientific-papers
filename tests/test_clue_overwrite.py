from __future__ import annotations

from pathlib import Path

from clues.interpretations import InterpretationRecord
from preprocessing.preprocessed import read_clue_rows
from clues.textual_clue_descriptions import _append_textual_clue
from clues.vl_figure_descriptions import _append_visual_clue


def test_visual_clue_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    old_record = InterpretationRecord(
        record_id="fig-1",
        kind="visual",
        text="old",
        model="model",
        prompt_id="visual_interpretation",
        prompt_version="v1",
        metadata={"paper_id": "ACL_paper-1", "figure_id": "fig-1", "source_paper_dataset": "ACL"},
    )
    new_record = InterpretationRecord(
        record_id="fig-1",
        kind="visual",
        text="new",
        model="model",
        prompt_id="visual_interpretation",
        prompt_version="v2",
        metadata={"paper_id": "ACL_paper-1", "figure_id": "fig-1", "source_paper_dataset": "ACL"},
    )

    clue_path = _append_visual_clue(old_record, tmp_path)
    _append_visual_clue(new_record, tmp_path, overwrite=True)

    assert clue_path == tmp_path / "ACL" / "ACL_paper-1" / "images" / "fig-1.jsonl"
    assert read_clue_rows(clue_path) == [
        {
            "paper_id": "ACL_paper-1",
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
        record_id="ACL_paper-1",
        kind="textual",
        text="old",
        model="model",
        prompt_id="textual_interpretation",
        prompt_version="v1",
        metadata={"source_paper_dataset": "ACL"},
    )
    new_record = InterpretationRecord(
        record_id="ACL_paper-1",
        kind="textual",
        text="new",
        model="model",
        prompt_id="textual_interpretation",
        prompt_version="v2",
        metadata={"source_paper_dataset": "ACL"},
    )

    clue_path = _append_textual_clue(old_record, tmp_path)
    _append_textual_clue(new_record, tmp_path, overwrite=True)

    assert clue_path == tmp_path / "ACL" / "ACL_paper-1" / "base" / "textual_clues.jsonl"
    assert read_clue_rows(clue_path) == [
        {
            "paper_id": "ACL_paper-1",
            "kind": "textual",
            "model": "model",
            "prompt_id": "textual_interpretation",
            "prompt_version": "v2",
            "output": "new",
        }
    ]
