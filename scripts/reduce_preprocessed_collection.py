#!/usr/bin/env python3
"""Trim preprocessed MinerU paper outputs and summarize visual/equation blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REFERENCES_HEADING_RE = re.compile(r"^##\s+(?:\d+(?:\.\d+)*\s+)?references\b", re.IGNORECASE)
LEVEL_TWO_HEADING_RE = re.compile(r"^##\s+")
EQUATION_TYPES = {"equation", "equation_interline", "equation_inline"}
IMAGE_LIKE_TYPES = {"image", "chart", "table"}


def main() -> None:
    args = parse_args()
    root = args.preprocessed_dir
    if not root.exists():
        raise SystemExit(f"Preprocessed directory does not exist: {root}")

    paper_summaries: list[dict[str, Any]] = []
    aggregate_block_types: Counter[str] = Counter()
    aggregate_image_types: Counter[str] = Counter()
    aggregate_image_extensions: Counter[str] = Counter()
    aggregate_equation_types: Counter[str] = Counter()

    for paper_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paper_path = paper_dir / "paper.json"
        markdown_path = paper_dir / "markdown.md"
        figures_path = paper_dir / "figures.json"
        if not paper_path.exists() or not markdown_path.exists():
            continue

        paper = read_json(paper_path)
        paper_id = str(paper.get("paper_id") or paper_dir.name)
        structured_outputs = paper.get("structured_outputs") or {}

        content_summaries: list[dict[str, Any]] = []
        content_results: list[dict[str, Any]] = []

        for name, relpath in sorted(structured_outputs.items()):
            content_path = paper_dir / str(relpath)
            if not content_path.exists():
                content_summaries.append({"name": name, "path": str(relpath), "missing": True})
                continue

            content = read_json(content_path)
            if is_v2_content_list(content):
                trimmed, summary, found_equations = trim_v2_content_list(content, args.max_pages)
            else:
                trimmed, summary, found_equations = trim_v1_content_list(content, args.max_pages)

            if not args.dry_run:
                write_json(content_path, trimmed)

            content_result = {"name": name, "path": str(relpath), **summary}
            content_summaries.append(content_result)
            content_results.append({"summary": content_result, "equations": found_equations})

        markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
        trimmed_markdown, markdown_summary = trim_markdown_after_references(markdown)
        if not args.dry_run and trimmed_markdown != markdown:
            markdown_path.write_text(trimmed_markdown, encoding="utf-8")

        if figures_path.exists():
            figures = read_json(figures_path)
            trimmed_figures = trim_page_indexed_records(figures, args.max_pages)
            if not args.dry_run and trimmed_figures != figures:
                write_json(figures_path, trimmed_figures)
        else:
            figures = []
            trimmed_figures = []

        if not args.dry_run:
            paper["num_pages_original"] = paper.get("num_pages_original", paper.get("num_pages"))
            paper["num_pages"] = min(args.max_pages, int(paper.get("num_pages") or args.max_pages))
            paper["markdown_sha256"] = sha256_text(
                markdown_path.read_text(encoding="utf-8", errors="replace")
            )
            if isinstance(paper.get("figures"), list):
                paper["figures"] = trim_page_indexed_records(paper["figures"], args.max_pages)
            paper["preprocessed_reduction"] = {
                "max_pages": args.max_pages,
                "markdown_trim_rule": "keep References; remove the first level-2 heading after References and everything after it",
                "equations_relpath": "equations.json",
                "report_relpath": "../preprocessed_reduction_report.json",
            }
            write_json(paper_path, paper)
            write_json(paper_dir / "equations.json", primary_equations(content_results))

        primary = primary_summary(content_results)
        equations = primary_equations(content_results)
        paper_block_types = Counter(primary.get("block_type_counts", {}))
        paper_image_types = Counter(primary.get("image_like_type_counts", {}))
        paper_image_extensions = Counter(primary.get("image_extensions", {}))
        paper_equation_types = Counter(primary.get("equation_type_counts", {}))
        paper_summary = {
            "paper_id": paper_id,
            "paper_dir": paper_dir.name,
            "original_num_pages": paper.get("num_pages_original", paper.get("num_pages")),
            "current_num_pages": min(args.max_pages, int(paper.get("num_pages") or args.max_pages)),
            "markdown": markdown_summary,
            "figures_json_count_before": len(figures) if isinstance(figures, list) else None,
            "figures_json_count_after": len(trimmed_figures) if isinstance(trimmed_figures, list) else None,
            "content_lists": content_summaries,
            "block_type_counts": dict(sorted(paper_block_types.items())),
            "image_like_type_counts": dict(sorted(paper_image_types.items())),
            "image_extensions": dict(sorted(paper_image_extensions.items())),
            "equation_type_counts": dict(sorted(paper_equation_types.items())),
            "equation_count": len(equations),
        }
        paper_summaries.append(paper_summary)
        aggregate_block_types.update(paper_block_types)
        aggregate_image_types.update(paper_image_types)
        aggregate_image_extensions.update(paper_image_extensions)
        aggregate_equation_types.update(paper_equation_types)

    report = {
        "preprocessed_dir": str(root),
        "dry_run": args.dry_run,
        "max_pages": args.max_pages,
        "paper_count": len(paper_summaries),
        "aggregate": {
            "block_type_counts": dict(sorted(aggregate_block_types.items())),
            "image_like_type_counts": dict(sorted(aggregate_image_types.items())),
            "image_extensions": dict(sorted(aggregate_image_extensions.items())),
            "equation_type_counts": dict(sorted(aggregate_equation_types.items())),
            "equation_count": sum(summary["equation_count"] for summary in paper_summaries),
        },
        "papers": paper_summaries,
    }

    if args.report:
        if not args.dry_run:
            write_json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root directory containing one subdirectory per paper.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Keep pages with zero-based page indexes below this value.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/preprocessed_reduction_report.json"),
        help="Report JSON path. The report is printed even when this is set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing files.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_v2_content_list(content: Any) -> bool:
    return (
        isinstance(content, list)
        and (not content or isinstance(content[0], list))
    )


def trim_v1_content_list(
    blocks: list[dict[str, Any]], max_pages: int
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    before = len(blocks)
    trimmed = [block for block in blocks if page_idx(block) is None or page_idx(block) < max_pages]
    summary, equations = summarize_blocks(trimmed, source_format="v1")
    summary.update({"format": "v1", "blocks_before": before, "blocks_after": len(trimmed)})
    return trimmed, summary, equations


def trim_v2_content_list(
    pages: list[list[dict[str, Any]]], max_pages: int
) -> tuple[list[list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    before_pages = len(pages)
    before_blocks = sum(len(page) for page in pages)
    trimmed = pages[:max_pages]
    flat_blocks: list[dict[str, Any]] = []
    for index, page in enumerate(trimmed):
        for block in page:
            copied = dict(block)
            copied.setdefault("page_idx", index)
            flat_blocks.append(copied)
    summary, equations = summarize_blocks(flat_blocks, source_format="v2")
    summary.update(
        {
            "format": "v2",
            "pages_before": before_pages,
            "pages_after": len(trimmed),
            "blocks_before": before_blocks,
            "blocks_after": len(flat_blocks),
        }
    )
    return trimmed, summary, equations


def summarize_blocks(blocks: list[dict[str, Any]], source_format: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    block_types: Counter[str] = Counter()
    image_types: Counter[str] = Counter()
    image_extensions: Counter[str] = Counter()
    equation_types: Counter[str] = Counter()
    equations: list[dict[str, Any]] = []

    for block_index, block in enumerate(blocks):
        block_type = str(block.get("type") or "unknown")
        block_types[block_type] += 1
        if block_type in IMAGE_LIKE_TYPES:
            image_types[block_type] += 1
            image_path = image_path_from_block(block)
            if image_path:
                suffix = Path(image_path).suffix.lower() or "<no_extension>"
                image_extensions[suffix] += 1
        for equation in equations_from_block(block):
            equation_types[equation["type"]] += 1
            equations.append(
                {
                    "source_format": source_format,
                    "block_index": block_index,
                    "page_idx": block.get("page_idx"),
                    **equation,
                }
            )

    return (
        {
            "block_type_counts": dict(sorted(block_types.items())),
            "image_like_type_counts": dict(sorted(image_types.items())),
            "image_extensions": dict(sorted(image_extensions.items())),
            "equation_type_counts": dict(sorted(equation_types.items())),
        },
        equations,
    )


def page_idx(record: dict[str, Any]) -> int | None:
    value = record.get("page_idx", record.get("page"))
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def trim_page_indexed_records(records: Any, max_pages: int) -> Any:
    if not isinstance(records, list):
        return records
    return [record for record in records if not isinstance(record, dict) or page_idx(record) is None or page_idx(record) < max_pages]


def trim_markdown_after_references(markdown: str) -> tuple[str, dict[str, Any]]:
    lines = markdown.splitlines(keepends=True)
    references_index: int | None = None
    trim_index: int | None = None

    for index, line in enumerate(lines):
        if REFERENCES_HEADING_RE.match(line.strip()):
            references_index = index
            break

    if references_index is not None:
        for index in range(references_index + 1, len(lines)):
            if LEVEL_TWO_HEADING_RE.match(lines[index].strip()):
                trim_index = index
                break

    if trim_index is None:
        return markdown, {
            "found_references": references_index is not None,
            "trimmed": False,
            "lines_before": len(lines),
            "lines_after": len(lines),
        }

    trimmed = "".join(lines[:trim_index]).rstrip() + "\n"
    return trimmed, {
        "found_references": True,
        "trimmed": len(trimmed) != len(markdown),
        "trim_heading": lines[trim_index].strip(),
        "lines_before": len(lines),
        "lines_after": len(trimmed.splitlines()),
    }


def equations_from_block(block: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    block_type = str(block.get("type") or "")
    if block_type in EQUATION_TYPES:
        content = extract_text(block.get("content")) or str(block.get("text") or "")
        if content.strip():
            found.append({"type": block_type, "content": content.strip()})

    for content_type, content in walk_content_items(block.get("content")):
        if content_type in EQUATION_TYPES:
            text = extract_text(content)
            if text.strip():
                found.append({"type": content_type, "content": text.strip()})
    return found


def walk_content_items(value: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        content_type = value.get("type")
        if isinstance(content_type, str):
            found.append((content_type, value.get("content", value)))
        for nested in value.values():
            found.extend(walk_content_items(nested))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_content_items(item))
    return found


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "latex", "html"):
            if key in value:
                text = extract_text(value[key])
                if text:
                    return text
        return " ".join(filter(None, (extract_text(item) for item in value.values())))
    if isinstance(value, list):
        return " ".join(filter(None, (extract_text(item) for item in value)))
    return ""


def image_path_from_block(block: dict[str, Any]) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("img_path", "image_path", "path"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("image_source", "table_source", "chart_source"):
            nested = content.get(key)
            if isinstance(nested, dict):
                value = image_path_from_block(nested)
                if value:
                    return value
    return None


def primary_summary(content_results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in content_results:
        summary = result["summary"]
        if summary.get("format") == "v2":
            return summary
    if content_results:
        return content_results[0]["summary"]
    return {}


def primary_equations(content_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in content_results:
        if result["summary"].get("format") == "v2":
            return result["equations"]
    if content_results:
        return content_results[0]["equations"]
    return []


if __name__ == "__main__":
    main()
