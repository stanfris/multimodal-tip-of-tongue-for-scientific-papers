from __future__ import annotations

import json

from preprocessing.preprocessed import append_clue_row, textual_clue_path, visual_clue_path
from queries.audit_query_generation import audit, stream_audit
from tests.test_domain_output_layout import _paper
from tests.test_managed_query_settings import write_settings


def test_audit_separates_incomplete_clues_from_eligible_missing_query(tmp_path) -> None:
    settings = write_settings(tmp_path)
    data_root = tmp_path / "data"
    for domain in ("ACL", "Biology"):
        paper = _paper(data_root / "preprocessed", domain)
        paper_id = paper["paper_id"]
        append_clue_row(
            textual_clue_path(data_root / "clues", paper_id, domain),
            {"paper_id": paper_id, "kind": "textual", "output": "Text clue."},
        )
        if domain == "ACL":
            append_clue_row(
                visual_clue_path(data_root / "clues", paper_id, f"{paper_id}_figure", domain),
                {"paper_id": paper_id, "figure_id": f"{paper_id}_figure", "kind": "visual", "output": "Visual clue."},
            )

    report = audit(settings, "test")
    assert report["domains"]["ACL"]["visual-only"]["reasons"]["eligible_but_no_query"] == ["ACL_paper"]
    assert report["domains"]["Biology"]["visual-only"]["reasons"]["missing_usable_figure_clue"] == ["Biology_paper"]
    assert json.loads(json.dumps(report))["papers_selected_for_queries"] == 2


def test_stream_audit_writes_missing_papers_as_it_checks_them(tmp_path, monkeypatch) -> None:
    settings = write_settings(tmp_path)
    split_index = tmp_path / "data" / "splits" / "index.json"
    split_index.parent.mkdir()
    split_index.write_text(json.dumps({"train": ["ACL/paper.pdf", "Biology/paper.pdf"]}), encoding="utf-8")
    settings.write_text(settings.read_text().replace("  clues_dir: clues", "  clues_dir: clues\n  split_index: ../data/splits/index.json"))
    for domain in ("ACL", "Biology"):
        _paper(tmp_path / "data" / "preprocessed", domain)

    output = tmp_path / "audit.jsonl"
    import queries.audit_query_generation as module
    original = module._paper_reason
    checked = 0

    def check_previous_result(paper, config, mode):
        nonlocal checked
        if checked:
            assert len(output.read_text().splitlines()) == 1
        checked += 1
        return original(paper, config, mode)

    monkeypatch.setattr(module, "_paper_reason", check_previous_result)
    stream_audit(settings, "train", output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["reason"] for row in rows] == ["no_usable_textual_clue", "no_usable_textual_clue"]
