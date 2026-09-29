import json
from pathlib import Path

from preprocessing.compact_preprocessed_collection import compact_paper_dir
from preprocessing.compact_preprocessed_collection import parse_args


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_compact_paper_dir_preserves_metadata_and_uses_relative_source_pdf(tmp_path: Path) -> None:
    paper_dir = tmp_path / "processed" / "ACL" / "ACL_paper"
    image_path = paper_dir / "mineru" / "raw" / "hybrid_auto" / "images" / "fig.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    (paper_dir / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    content_path = paper_dir / "mineru" / "raw" / "hybrid_auto" / "paper_content_list_v2.json"
    write_json(
        content_path,
        [[{"type": "image", "page_idx": 0, "img_path": "images/fig.png"}]],
    )
    figures = [
        {
            "paper_id": "ACL_paper",
            "figure_id": "fig-1",
            "type": "image",
            "page_idx": 0,
            "image_relpath": "mineru/raw/hybrid_auto/images/fig.png",
            "image_path": str(image_path),
        }
    ]
    write_json(
        paper_dir / "paper.json",
        {
            "paper_id": "ACL_paper",
            "source_pdf": str(tmp_path / "stale" / "paper.pdf"),
            "source_pdf_relpath": "ACL/paper.pdf",
            "source_paper_dataset": "ACL",
            "figures": figures,
            "structured_outputs": {"content": "mineru/raw/hybrid_auto/paper_content_list_v2.json"},
            "mineru_raw_relpath": "mineru",
        },
    )
    write_json(paper_dir / "figures.json", figures)
    pdf_dir = tmp_path / "pdf_datasets"
    source_pdf = pdf_dir / "ACL" / "paper.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF-1.7\n")

    result = compact_paper_dir(paper_dir, pdf_dir, dry_run=False)

    assert result["status"] == "processed"
    assert (paper_dir / "paper.pdf").read_bytes() == b"%PDF-1.7\n"
    assert (paper_dir / "images" / "fig.png").read_bytes() == b"image"
    assert not (paper_dir / "mineru").exists()

    updated_paper = json.loads((paper_dir / "paper.json").read_text(encoding="utf-8"))
    updated_figures = json.loads((paper_dir / "figures.json").read_text(encoding="utf-8"))
    assert updated_paper["figures"][0]["image_relpath"] == "images/fig.png"
    assert updated_paper["figures"][0]["image_path"] == "images/fig.png"
    assert updated_paper["source_pdf_copy_relpath"] == "paper.pdf"
    assert "structured_outputs" not in updated_paper
    assert updated_figures == updated_paper["figures"]


def test_compact_parser_derives_pdf_dir_from_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "compact_preprocessed_collection",
            "--preprocessed-dir",
            str(tmp_path / "processed"),
            "--root-dir",
            str(tmp_path),
        ],
    )

    args = parse_args()
    assert args.root_dir == tmp_path
    assert args.pdf_dir is None
