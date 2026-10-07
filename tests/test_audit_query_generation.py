from __future__ import annotations

import json

from preprocessing.preprocessed import append_clue_row, textual_clue_path, visual_clue_path
from queries.audit_query_generation import audit
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
