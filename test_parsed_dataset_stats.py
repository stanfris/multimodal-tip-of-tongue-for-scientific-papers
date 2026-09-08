from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_generation.parsed_dataset_stats import compute_stats, extract_abstract, run


def test_extract_abstract_stops_at_next_heading() -> None:
    markdown = """# Paper Title

## Abstract

This is the parsed abstract.

It has two paragraphs.

## 1 Introduction

Body text.
"""

    assert extract_abstract(markdown) == "This is the parsed abstract.\n\nIt has two paragraphs."


def test_compute_stats_compares_abstracts_and_visual_counts(tmp_path: Path) -> None:
    metadata_path = tmp_path / "papers.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "anthology_id": "paper-1",
                "title": "A Paper",
                "abstract": "Neural systems compare figures and tables with parsed images.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paper_dir = tmp_path / "preprocessed" / "paper-1"
    images_dir = paper_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "one.jpg").write_bytes(b"fake")
    (images_dir / "two.png").write_bytes(b"fake")
    (paper_dir / "markdown.md").write_text(
        """# A Paper

## Abstract

Neural systems compare figures and tables with parsed images.

## 1 Intro

![](images/one.jpg)
Figure 1: First figure.

<table><tr><td>x</td></tr></table>
Table 1: First table.
""",
        encoding="utf-8",
    )

    report = compute_stats(tmp_path / "preprocessed", metadata_path)

    assert report["paper_count"] == 1
    row = report["papers"][0]
    assert row["abstract_sequence_similarity"] == 1.0
    assert row["abstract_word_jaccard"] == 1.0
    assert row["figure_caption_count"] == 1
    assert row["table_caption_count"] == 1
    assert row["figure_table_caption_count"] == 2
    assert row["markdown_linked_image_count"] == 1
    assert row["markdown_image_count"] == 2
    assert row["image_file_count"] == 2
    assert row["markdown_to_image_file_delta"] == 0
    assert row["markdown_to_image_file_absolute_error"] == 0
    assert row["caption_to_image_file_delta"] == 0
    assert row["caption_to_image_file_absolute_error"] == 0
    assert row["caption_to_markdown_image_delta"] == 0
    assert row["caption_to_markdown_image_absolute_error"] == 0
    assert report["aggregate"]["markdown_to_image_file_delta"] == 0.0
    assert report["aggregate"]["markdown_to_image_file_absolute_error"] == 0.0
    assert report["aggregate"]["caption_to_image_file_delta"] == 0.0
    assert report["aggregate"]["caption_to_image_file_absolute_error"] == 0.0
    assert report["aggregate"]["caption_to_markdown_image_delta"] == 0.0
    assert report["aggregate"]["caption_to_markdown_image_absolute_error"] == 0.0


def test_run_writes_json_and_csv_outputs(tmp_path: Path) -> None:
    metadata_path = tmp_path / "papers.jsonl"
    metadata_path.write_text('{"anthology_id": "paper-1", "abstract": "An abstract."}\n', encoding="utf-8")
    paper_dir = tmp_path / "preprocessed" / "paper-1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "markdown.md").write_text("## Abstract\n\nAn abstract.\n", encoding="utf-8")

    output_path = tmp_path / "stats" / "report.json"
    csv_path = tmp_path / "stats" / "rows.csv"
    report = run(
        argparse.Namespace(
            preprocessed_dir=tmp_path / "preprocessed",
            metadata=metadata_path,
            output=output_path,
            csv_output=csv_path,
        )
    )

    assert report["paper_count"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["paper_count"] == 1
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "paper_count" in csv_text
    assert "abstract_word_jaccard" in csv_text
    assert "missed_visual_items_absolute_distance" in csv_text
    assert "markdown_to_image_file_delta" not in csv_text
    assert "paper_id" not in csv_text
