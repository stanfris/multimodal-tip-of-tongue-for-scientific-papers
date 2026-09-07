from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_generation.mineru_extraction import (
    ExtractionError,
    MinerUOptions,
    build_extract_parser,
    is_complete,
    normalize_figures,
    options_from_args,
    validate_and_normalize,
)


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c49444154789c63606060000000040001f61738550000000049454e44ae426082"
        )
    )


def test_normalize_figures_uses_structured_content_and_skips_tables(tmp_path: Path) -> None:
    image = tmp_path / "mineru" / "images" / "fig1.png"
    write_png(image)
    content = [
        {"type": "table", "img_path": "images/table.png", "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/fig1.png",
            "page_idx": 2,
            "bbox": [1, 2, 3, 4],
            "image_caption": ["Figure 1: Result"],
        },
    ]

    figures = normalize_figures(
        content,
        tmp_path / "mineru",
        "paper-1",
        tmp_path / "paper-1.pdf",
        tmp_path,
        tmp_path / "final",
    )

    assert len(figures) == 1
    assert figures[0]["type"] == "image"
    assert figures[0]["page_idx"] == 2
    assert figures[0]["caption"] == ["Figure 1: Result"]
    assert figures[0]["image_relpath"] == "mineru/images/fig1.png"
    assert figures[0]["image_path"] == str((tmp_path / "final" / "mineru/images/fig1.png").resolve())


def test_extract_options_accept_page_range() -> None:
    args = build_extract_parser().parse_args(["--start-page-id", "2", "--end-page-id", "7"])

    options = options_from_args(args)

    assert options.start_page_id == 2
    assert options.end_page_id == 7


def test_validate_and_normalize_writes_paper_and_figures(tmp_path: Path) -> None:
    raw = tmp_path / "work" / "mineru" / "paper"
    raw.mkdir(parents=True)
    (raw / "paper.md").write_text("# Title\n\n" + "body " * 80, encoding="utf-8")
    write_png(raw / "images" / "fig.png")
    (raw / "paper_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "image",
                    "img_path": "images/fig.png",
                    "page_idx": 0,
                    "bbox": [10, 20, 30, 40],
                    "image_caption": ["Figure 1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")

    record = validate_and_normalize(
        raw,
        pdf,
        tmp_path / "work",
        tmp_path / "final",
        MinerUOptions(min_markdown_chars=10),
    )

    assert record["paper_id"] == "paper"
    assert record["markdown_relpath"] == "markdown.md"
    assert record["num_pages"] == 1
    assert len(record["figures"]) == 1
    assert (tmp_path / "work" / "paper.json").exists()
    assert (tmp_path / "work" / "figures.json").exists()


def test_validate_and_normalize_rejects_missing_images(tmp_path: Path) -> None:
    raw = tmp_path / "work" / "mineru"
    raw.mkdir(parents=True)
    (raw / "paper.md").write_text("# Title\n\n" + "body " * 80, encoding="utf-8")
    (raw / "paper_content_list.json").write_text(
        json.dumps([{"type": "image", "img_path": "images/missing.png", "page_idx": 0}]),
        encoding="utf-8",
    )
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")

    with pytest.raises(ExtractionError, match="Referenced image does not exist"):
        validate_and_normalize(raw, pdf, tmp_path / "work", tmp_path / "final", MinerUOptions(min_markdown_chars=10))


def test_validate_and_normalize_rejects_malformed_content_list(tmp_path: Path) -> None:
    raw = tmp_path / "work" / "mineru"
    raw.mkdir(parents=True)
    (raw / "paper.md").write_text("# Title\n\n" + "body " * 80, encoding="utf-8")
    (raw / "paper_content_list.json").write_text("{not-json", encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")

    with pytest.raises(ExtractionError, match="Could not parse MinerU content list JSON"):
        validate_and_normalize(raw, pdf, tmp_path / "work", tmp_path / "final", MinerUOptions(min_markdown_chars=10))


def test_is_complete_revalidates_referenced_outputs(tmp_path: Path) -> None:
    paper_dir = tmp_path / "papers" / "paper"
    write_png(paper_dir / "mineru" / "images" / "fig.png")
    (paper_dir / "markdown.md").write_text("# Title\n", encoding="utf-8")
    (paper_dir / "paper.json").write_text(
        json.dumps(
            {
                "paper_id": "paper",
                "markdown_relpath": "markdown.md",
                "figures": [{"image_relpath": "mineru/images/fig.png"}],
            }
        ),
        encoding="utf-8",
    )
    (paper_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    assert is_complete(tmp_path, "paper")
    (paper_dir / "mineru" / "images" / "fig.png").unlink()
    assert not is_complete(tmp_path, "paper")
