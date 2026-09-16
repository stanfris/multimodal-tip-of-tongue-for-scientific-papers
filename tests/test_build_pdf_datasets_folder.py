from __future__ import annotations

from scripts.build_pdf_datasets_folder import DatasetSource
from scripts.build_pdf_datasets_folder import materialize_pdf_datasets


def test_materialize_pdf_datasets_copies_five_folders(tmp_path) -> None:
    source_root = tmp_path / "sources"
    output_dir = tmp_path / "pdf_datasets"
    sources = []
    for name in ("ACL", "Physics", "Engineering", "Biology", "Medicine"):
        source = source_root / name
        source.mkdir(parents=True)
        (source / f"{name.lower()}.pdf").write_bytes(f"%PDF\n{name}".encode())
        sources.append(DatasetSource(name, source))

    counts = materialize_pdf_datasets(
        sources=sources,
        output_dir=output_dir,
        move=False,
        overwrite=False,
        clean=False,
        dry_run=False,
    )

    for name in ("ACL", "Physics", "Engineering", "Biology", "Medicine"):
        assert (output_dir / name / f"{name.lower()}.pdf").exists()
        assert counts[f"written_{name}"] == 1


def test_materialize_pdf_datasets_preserves_nested_relative_paths(tmp_path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "paper.pdf").write_bytes(b"%PDF\nnested")

    materialize_pdf_datasets(
        sources=[DatasetSource("ACL", source)],
        output_dir=tmp_path / "out",
        move=False,
        overwrite=False,
        clean=False,
        dry_run=False,
    )

    assert (tmp_path / "out" / "ACL" / "nested" / "paper.pdf").read_bytes() == b"%PDF\nnested"


def test_materialize_pdf_datasets_reports_missing_source(tmp_path) -> None:
    counts = materialize_pdf_datasets(
        sources=[DatasetSource("Medicine", tmp_path / "missing")],
        output_dir=tmp_path / "out",
        move=False,
        overwrite=False,
        clean=False,
        dry_run=False,
    )

    assert counts == {"missing_source_Medicine": 1}
    assert (tmp_path / "out" / "Medicine").is_dir()
