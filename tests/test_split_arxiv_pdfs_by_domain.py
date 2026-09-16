from __future__ import annotations

import json

from scripts.split_arxiv_pdfs_by_domain import split_pdfs


def write_manifest(path, rows):  # type: ignore[no-untyped-def]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_split_pdfs_copies_by_selection_domain(tmp_path) -> None:
    manifest = tmp_path / "eligible_records.jsonl"
    pdf_dir = tmp_path / "pdfs"
    output_dir = tmp_path / "split"
    pdf_dir.mkdir()
    (pdf_dir / "physics.pdf").write_bytes(b"%PDF\nphysics")
    (pdf_dir / "engineering.pdf").write_bytes(b"%PDF\nengineering")
    write_manifest(
        manifest,
        [
            {
                "pdf_filename": "physics.pdf",
                "selection_domain": "physics.",
                "primary_category": "physics.optics",
            },
            {
                "pdf_filename": "engineering.pdf",
                "selection_domain": "eess.",
                "primary_category": "physics.ins-det",
                "categories": ["physics.ins-det", "eess.SP"],
            },
        ],
    )

    counts = split_pdfs(
        manifest_path=manifest,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        move=False,
        dry_run=False,
        overwrite=False,
    )

    assert counts["written_physics"] == 1
    assert counts["written_engineering"] == 1
    assert (output_dir / "physics" / "physics.pdf").read_bytes() == b"%PDF\nphysics"
    assert (output_dir / "engineering" / "engineering.pdf").read_bytes() == b"%PDF\nengineering"
    assert (pdf_dir / "physics.pdf").exists()


def test_split_pdfs_reports_missing_sources(tmp_path) -> None:
    manifest = tmp_path / "eligible_records.jsonl"
    pdf_dir = tmp_path / "pdfs"
    output_dir = tmp_path / "split"
    pdf_dir.mkdir()
    write_manifest(manifest, [{"pdf_filename": "missing.pdf", "selection_domain": "eess."}])

    counts = split_pdfs(
        manifest_path=manifest,
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        move=False,
        dry_run=False,
        overwrite=False,
    )

    assert counts == {"missing_source_engineering": 1}
