from __future__ import annotations

from scripts.pdf_page_distribution import arxiv_subset_from_manifest_row


def test_arxiv_subset_prefers_selection_domain_over_primary_category() -> None:
    row = {
        "selection_domain": "eess.",
        "primary_category": "physics.ins-det",
        "categories": ["physics.ins-det", "eess.SP"],
    }

    assert arxiv_subset_from_manifest_row(row) == "arxiv_engineering"


def test_arxiv_subset_uses_primary_category_for_legacy_rows() -> None:
    row = {
        "primary_category": "eess.SY",
        "categories": ["physics.app-ph", "eess.SY"],
    }

    assert arxiv_subset_from_manifest_row(row) == "arxiv_engineering"
