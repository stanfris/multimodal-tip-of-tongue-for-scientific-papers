import json
from pathlib import Path

from preprocessing.reduce_preprocessed_collection import parse_args, reduce_paper_dir, should_report_progress


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_reduce_paper_dir_preserves_complete_document(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper-1"
    paper_dir.mkdir()
    content_path = paper_dir / "content_list_v2.json"
    figures_path = paper_dir / "figures.json"
    markdown_path = paper_dir / "markdown.md"
    paper_path = paper_dir / "paper.json"

    content = [[{"type": "text", "text": f"page {index}"}] for index in range(12)]
    figures = [{"type": "image", "page_idx": 11, "img_path": "images/late.png"}]
    markdown = "# Paper\n\n## References\n\nReference text.\n\n## Appendix\n\nKeep this appendix.\n"
    paper = {
        "paper_id": "paper-1",
        "num_pages": 12,
        "figures": figures,
        "structured_outputs": {"content_v2": content_path.name},
    }

    write_json(content_path, content)
    write_json(figures_path, figures)
    write_json(paper_path, paper)
    markdown_path.write_text(markdown, encoding="utf-8")

    result = reduce_paper_dir(paper_dir, dry_run=False)

    assert result["status"] == "processed"
    assert result["current_num_pages"] == 12
    assert json.loads(content_path.read_text(encoding="utf-8")) == content
    assert json.loads(figures_path.read_text(encoding="utf-8")) == figures
    assert markdown_path.read_text(encoding="utf-8") == markdown

    updated_paper = json.loads(paper_path.read_text(encoding="utf-8"))
    assert updated_paper["num_pages"] == 12
    assert updated_paper["figures"] == figures
    assert updated_paper["preprocessed_analysis"]["content_policy"] == "full_document"
    assert "preprocessed_reduction" not in updated_paper


def test_parser_has_no_page_limit(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["reduce_preprocessed_collection"])

    args = parse_args()

    assert not hasattr(args, "max_pages")


def test_reduce_skips_already_compacted_paper(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper-1"
    (paper_dir / "images").mkdir(parents=True)
    write_json(paper_dir / "paper.json", {"paper_id": "paper-1", "split": "train", "figures": []})
    write_json(paper_dir / "figures.json", [])
    (paper_dir / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.7\n")

    result = reduce_paper_dir(paper_dir, dry_run=False)

    assert result == {
        "status": "skipped",
        "paper_dir": str(paper_dir),
        "reason": "already_compacted",
    }
    assert not (paper_dir / "equations.json").exists()


def test_progress_reports_first_interval_and_last() -> None:
    assert should_report_progress(1, 250, 100)
    assert should_report_progress(100, 250, 100)
    assert should_report_progress(250, 250, 100)
    assert not should_report_progress(99, 250, 100)
    assert not should_report_progress(250, 250, 0)
