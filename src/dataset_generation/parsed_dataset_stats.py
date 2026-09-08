"""Statistics for compact parsed ACL paper directories."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_METADATA_PATH = Path("data") / "acl_subset" / "papers.jsonl"
DEFAULT_PREPROCESSED_DIR = Path("data") / "preprocessed"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

ABSTRACT_HEADING_RE = re.compile(r"^\s*#{1,6}\s+abstract\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
CAPTION_RE = re.compile(
    r"(?im)^\s*(?P<kind>fig(?:ure)?|table)\s*\.?\s*(?P<number>[0-9]+[A-Za-z]?)\s*[:.)]"
)
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
        write_csv(args.csv_output, report["papers"])
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
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
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
    caption_counts = count_caption_labels(markdown)
    markdown_image_count = len(extract_markdown_images(markdown))
    image_file_count = count_image_files(paper.paper_dir / "images")
    expected_visual_count = caption_counts["figure"] + caption_counts["table"]
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
        "figure_caption_count": caption_counts["figure"],
        "table_caption_count": caption_counts["table"],
        "figure_table_caption_count": expected_visual_count,
        "markdown_image_count": markdown_image_count,
        "image_file_count": image_file_count,
        "caption_to_image_file_delta": expected_visual_count - image_file_count,
        "caption_to_image_file_absolute_error": abs(expected_visual_count - image_file_count),
        "caption_to_markdown_image_delta": expected_visual_count - markdown_image_count,
        "caption_to_markdown_image_absolute_error": abs(expected_visual_count - markdown_image_count),
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


def count_caption_labels(markdown: str) -> Counter[str]:
    labels: set[tuple[str, str]] = set()
    for match in CAPTION_RE.finditer(markdown):
        kind = "figure" if match.group("kind").casefold().startswith("fig") else "table"
        labels.add((kind, match.group("number").casefold()))
    return Counter(kind for kind, _number in labels)


def extract_markdown_images(markdown: str) -> list[str]:
    return [*MARKDOWN_IMAGE_RE.findall(markdown), *HTML_IMG_RE.findall(markdown)]


def count_image_files(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caption_total = sum(int(row["figure_table_caption_count"]) for row in rows)
    image_file_total = sum(int(row["image_file_count"]) for row in rows)
    markdown_image_total = sum(int(row["markdown_image_count"]) for row in rows)
    figure_total = sum(int(row["figure_caption_count"]) for row in rows)
    table_total = sum(int(row["table_caption_count"]) for row in rows)
    mean_caption_to_image_file_error = mean(row["caption_to_image_file_delta"] for row in rows)
    mean_caption_to_image_file_absolute_error = mean(row["caption_to_image_file_absolute_error"] for row in rows)
    mean_caption_to_markdown_image_error = mean(row["caption_to_markdown_image_delta"] for row in rows)
    mean_caption_to_markdown_image_absolute_error = mean(
        row["caption_to_markdown_image_absolute_error"] for row in rows
    )
    return {
        "metadata_match_count": sum(1 for row in rows if row["has_metadata"]),
        "metadata_abstract_count": sum(1 for row in rows if row["has_metadata_abstract"]),
        "parsed_abstract_count": sum(1 for row in rows if row["has_parsed_abstract"]),
        "mean_abstract_sequence_similarity": mean(row["abstract_sequence_similarity"] for row in rows),
        "mean_abstract_word_jaccard": mean(row["abstract_word_jaccard"] for row in rows),
        "mean_metadata_abstract_word_recall": mean_optional(row["metadata_abstract_word_recall"] for row in rows),
        "mean_parsed_abstract_word_precision": mean_optional(row["parsed_abstract_word_precision"] for row in rows),
        "figure_caption_count": figure_total,
        "table_caption_count": table_total,
        "figure_table_caption_count": caption_total,
        "markdown_image_count": markdown_image_total,
        "image_file_count": image_file_total,
        "caption_to_image_file_delta": mean_caption_to_image_file_error,
        "caption_to_image_file_absolute_error": mean_caption_to_image_file_absolute_error,
        "caption_to_markdown_image_delta": mean_caption_to_markdown_image_error,
        "caption_to_markdown_image_absolute_error": mean_caption_to_markdown_image_absolute_error,
        "total_caption_to_image_file_delta": caption_total - image_file_total,
        "total_caption_to_markdown_image_delta": caption_total - markdown_image_total,
        "papers_with_caption_image_file_mismatch": sum(
            1 for row in rows if row["figure_table_caption_count"] != row["image_file_count"]
        ),
        "papers_with_caption_markdown_image_mismatch": sum(
            1 for row in rows if row["figure_table_caption_count"] != row["markdown_image_count"]
        ),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
