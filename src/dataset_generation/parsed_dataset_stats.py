"""Statistics for compact parsed ACL paper directories."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dataset_generation.jsonl import read_jsonl_objects


DEFAULT_METADATA_PATH = Path("data") / "acl_subset" / "papers.jsonl"
DEFAULT_PREPROCESSED_DIR = Path("data") / "preprocessed"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

ABSTRACT_HEADING_RE = re.compile(r"^\s*#{1,6}\s+abstract\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
TABLE_TAG_RE = re.compile(r"<table\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")


@dataclass(frozen=True)
class PaperInputs:
    paper_id: str
    paper_dir: Path
    markdown_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute quality statistics for parsed paper datasets.")
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED_DIR,
        help="Directory containing parsed paper directories, or a parent with a papers/ subdirectory.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="ACL subset papers.jsonl metadata with anthology_id and abstract fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. The report is always printed to stdout.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-paper rows.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = compute_stats(args.preprocessed_dir, args.metadata)
    if args.output:
        write_json(args.output, report)
    if args.csv_output:
        write_csv(args.csv_output, aggregate_csv_row(report))
    return report


def compute_stats(preprocessed_dir: str | Path, metadata_path: str | Path) -> dict[str, Any]:
    metadata = read_metadata(metadata_path)
    papers = discover_papers(preprocessed_dir)
    rows = [summarize_paper(paper, metadata.get(paper.paper_id, {})) for paper in papers]
    return {
        "preprocessed_dir": str(Path(preprocessed_dir)),
        "metadata": str(Path(metadata_path)),
        "paper_count": len(rows),
        "aggregate": aggregate_rows(rows),
        "papers": rows,
    }


def read_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata_path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_objects(metadata_path):
        paper_id = str(row.get("anthology_id") or row.get("paper_id") or "").strip()
        if paper_id:
            rows[paper_id] = row
    return rows


def discover_papers(path: str | Path) -> list[PaperInputs]:
    root = paper_root(Path(path))
    if not root.exists():
        raise FileNotFoundError(f"Preprocessed directory does not exist: {root}")
    papers = []
    for paper_dir in sorted(candidate for candidate in root.iterdir() if candidate.is_dir()):
        markdown_path = paper_dir / "markdown.md"
        if markdown_path.exists():
            papers.append(PaperInputs(paper_id=paper_dir.name, paper_dir=paper_dir, markdown_path=markdown_path))
    return papers


def paper_root(path: Path) -> Path:
    if path.exists() and any((child / "markdown.md").exists() for child in path.iterdir() if child.is_dir()):
        return path
    nested = path / "papers"
    return nested if nested.exists() else path


def summarize_paper(paper: PaperInputs, metadata: dict[str, Any]) -> dict[str, Any]:
    markdown = paper.markdown_path.read_text(encoding="utf-8", errors="replace")
    metadata_abstract = str(metadata.get("abstract") or "")
    parsed_abstract = extract_abstract(markdown)
    metadata_tokens = token_set(metadata_abstract)
    parsed_tokens = token_set(parsed_abstract)
    overlap = metadata_tokens & parsed_tokens
    markdown_linked_image_count = len(extract_markdown_images(markdown))
    markdown_table_count = count_markdown_tables(markdown)
    markdown_image_count = markdown_linked_image_count + markdown_table_count
    image_file_count = count_image_files(paper.paper_dir / "images")
    markdown_to_image_file_delta = markdown_image_count - image_file_count
    return {
        "paper_id": paper.paper_id,
        "title": metadata.get("title"),
        "has_metadata": bool(metadata),
        "has_metadata_abstract": bool(metadata_abstract.strip()),
        "has_parsed_abstract": bool(parsed_abstract.strip()),
        "metadata_abstract_chars": len(metadata_abstract),
        "parsed_abstract_chars": len(parsed_abstract),
        "abstract_sequence_similarity": round(sequence_similarity(metadata_abstract, parsed_abstract), 6),
        "abstract_word_jaccard": round(jaccard(metadata_tokens, parsed_tokens), 6),
        "metadata_abstract_word_recall": round(len(overlap) / len(metadata_tokens), 6) if metadata_tokens else None,
        "parsed_abstract_word_precision": round(len(overlap) / len(parsed_tokens), 6) if parsed_tokens else None,
        "markdown_figure_count": markdown_linked_image_count,
        "markdown_table_count": markdown_table_count,
        "markdown_linked_image_count": markdown_linked_image_count,
        "markdown_image_count": markdown_image_count,
        "image_file_count": image_file_count,
        "markdown_to_image_file_delta": markdown_to_image_file_delta,
        "markdown_to_image_file_absolute_error": abs(markdown_to_image_file_delta),
    }


def extract_abstract(markdown: str) -> str:
    match = ABSTRACT_HEADING_RE.search(markdown)
    if not match:
        return ""
    next_heading = NEXT_HEADING_RE.search(markdown, match.end())
    end = next_heading.start() if next_heading else len(markdown)
    return markdown[match.end() : end].strip()


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = text.replace("$", " ")
    text = re.sub(r"[_^{}\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def token_set(text: str) -> set[str]:
    return set(WORD_RE.findall(normalize_text(text)))


def sequence_similarity(left: str, right: str) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized and not right_normalized:
        return 1.0
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def extract_markdown_images(markdown: str) -> list[str]:
    return [*MARKDOWN_IMAGE_RE.findall(markdown), *HTML_IMG_RE.findall(markdown)]


def count_markdown_tables(markdown: str) -> int:
    return len(TABLE_TAG_RE.findall(markdown))


def count_image_files(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_file_total = sum(int(row["image_file_count"]) for row in rows)
    markdown_image_total = sum(int(row["markdown_image_count"]) for row in rows)
    markdown_linked_image_total = sum(int(row["markdown_linked_image_count"]) for row in rows)
    markdown_figure_total = sum(int(row["markdown_figure_count"]) for row in rows)
    markdown_table_total = sum(int(row["markdown_table_count"]) for row in rows)
    mean_markdown_to_image_file_error = mean(row["markdown_to_image_file_delta"] for row in rows)
    mean_markdown_to_image_file_absolute_error = mean(
        row["markdown_to_image_file_absolute_error"] for row in rows
    )
    return {
        "metadata_match_count": sum(1 for row in rows if row["has_metadata"]),
        "metadata_abstract_count": sum(1 for row in rows if row["has_metadata_abstract"]),
        "parsed_abstract_count": sum(1 for row in rows if row["has_parsed_abstract"]),
        "mean_abstract_sequence_similarity": mean(row["abstract_sequence_similarity"] for row in rows),
        "mean_abstract_word_jaccard": mean(row["abstract_word_jaccard"] for row in rows),
        "mean_metadata_abstract_word_recall": mean_optional(row["metadata_abstract_word_recall"] for row in rows),
        "mean_parsed_abstract_word_precision": mean_optional(row["parsed_abstract_word_precision"] for row in rows),
        "markdown_figure_count": markdown_figure_total,
        "markdown_table_count": markdown_table_total,
        "markdown_linked_image_count": markdown_linked_image_total,
        "markdown_image_count": markdown_image_total,
        "image_file_count": image_file_total,
        "markdown_to_image_file_delta": mean_markdown_to_image_file_error,
        "markdown_to_image_file_absolute_error": mean_markdown_to_image_file_absolute_error,
        "total_markdown_to_image_file_delta": markdown_image_total - image_file_total,
        "papers_with_markdown_image_file_mismatch": sum(
            1 for row in rows if row["markdown_image_count"] != row["image_file_count"]
        ),
    }


def aggregate_csv_row(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"]
    return {
        "paper_count": report["paper_count"],
        "metadata_match_count": aggregate["metadata_match_count"],
        "metadata_abstract_count": aggregate["metadata_abstract_count"],
        "parsed_abstract_count": aggregate["parsed_abstract_count"],
        "abstract_sequence_similarity": aggregate["mean_abstract_sequence_similarity"],
        "abstract_word_jaccard": aggregate["mean_abstract_word_jaccard"],
        "metadata_abstract_word_recall": aggregate["mean_metadata_abstract_word_recall"],
        "parsed_abstract_word_precision": aggregate["mean_parsed_abstract_word_precision"],
        "missed_visual_items_absolute_distance": aggregate["markdown_to_image_file_absolute_error"],
        "papers_with_markdown_image_file_mismatch": aggregate["papers_with_markdown_image_file_mismatch"],
    }


def mean(values: Any) -> float | None:
    materialized = list(values)
    return round(statistics.fmean(materialized), 6) if materialized else None


def mean_optional(values: Any) -> float | None:
    materialized = [value for value in values if value is not None]
    return round(statistics.fmean(materialized), 6) if materialized else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
