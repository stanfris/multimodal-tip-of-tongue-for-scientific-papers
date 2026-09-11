"""Canonical paper-level dataset representation and migration helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dataset_generation.jsonl import read_jsonl_objects
from dataset_generation.storage import read_dataset_artifact


DEFAULT_CANONICAL_DIR = Path("data") / "canonical"


@dataclass(frozen=True)
class CanonicalClue:
    output: str
    kind: str
    model: str
    prompt_id: str
    prompt_version: str


@dataclass(frozen=True)
class CanonicalFigure:
    figure_id: str
    filename: str
    image_relpath: str
    image_sha256: str
    source_rows: list[int] = field(default_factory=list)
    legacy_record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalPaper:
    paper_id: str
    markdown_relpath: str
    markdown_sha256: str
    source: dict[str, Any]
    figures: list[CanonicalFigure] = field(default_factory=list)
    queries: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"visual_only": [], "visual_and_text": []}
    )


def canonical_dataset_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name == "papers.jsonl":
        return candidate
    return candidate / "papers.jsonl"


def is_canonical_dataset(path: str | Path) -> bool:
    return canonical_dataset_path(path).exists()


def read_canonical_papers(path: str | Path) -> list[dict[str, Any]]:
    data_path = canonical_dataset_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"No papers.jsonl found at {data_path}")
    papers = read_jsonl_objects(data_path)
    validate_canonical_papers(papers, root=data_path.parent)
    return papers


def write_canonical_papers(papers: Iterable[dict[str, Any] | CanonicalPaper], path: str | Path) -> Path:
    data_path = canonical_dataset_path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_paper_to_dict(paper) for paper in papers]
    validate_canonical_papers(rows, root=data_path.parent)
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return data_path




def canonical_textual_clues_path(path: str | Path) -> Path:
    return canonical_dataset_path(path).parent / "textual_clues.jsonl"


def canonical_visual_clues_path(path: str | Path) -> Path:
    return canonical_dataset_path(path).parent / "visual_clues.jsonl"


def canonical_markdown_path(root: str | Path, paper: dict[str, Any]) -> Path:
    return canonical_dataset_path(root).parent / _required_text(paper, "markdown_relpath")


def read_canonical_markdown(root: str | Path, paper: dict[str, Any]) -> str:
    return canonical_markdown_path(root, paper).read_text(encoding="utf-8")


def read_canonical_clues(path: str | Path, *, kind: str | None = None) -> list[dict[str, Any]]:
    clues = read_jsonl_objects(path, missing_ok=True)
    if kind is not None:
        clues = [clue for clue in clues if clue.get("kind") == kind]
    return clues


def write_canonical_clues(clues: Iterable[dict[str, Any]], path: str | Path) -> Path:
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(clues)
    validate_canonical_clues(rows, kind="textual" if data_path.name.startswith("textual") else "visual")
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return data_path


def prepare_canonical_papers_from_rows(
    processed_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    textual_interpretations: str | Path | None = None,
    visual_interpretations: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the canonical paper dataset from a processed figure-row artifact."""
    processed_path = Path(processed_dataset_dir)
    output_path = Path(output_dir)
    image_root = output_path / "images"
    markdown_root = output_path / "markdown"
    image_root.mkdir(parents=True, exist_ok=True)
    markdown_root.mkdir(parents=True, exist_ok=True)

    frame = read_dataset_artifact(processed_path)
    rows = [row.to_dict() for _, row in frame.iterrows()]
    matched_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("match_status") == "matched" and _clean(row.get("resolved_paper_id"))
    ]
    unmatched_rows = [index for index, row in enumerate(rows) if (index, row) not in matched_rows]

    papers_by_id: dict[str, dict[str, Any]] = {}
    figure_lookup = _build_figures(matched_rows, processed_path, output_path, image_root)
    for paper_id in sorted({str(row["resolved_paper_id"]) for _, row in matched_rows}):
        paper_rows = [(index, row) for index, row in matched_rows if str(row["resolved_paper_id"]) == paper_id]
        markdowns = {_clean(row.get("markdown")) for _, row in paper_rows if _clean(row.get("markdown"))}
        if len(markdowns) > 1:
            raise ValueError(f"Multiple markdown bodies found for paper {paper_id}")
        markdown = next(iter(markdowns), "")
        markdown_relpath = _write_markdown(markdown_root, paper_id, markdown)
        figures_with_paper = sorted(
            [figure for figure in figure_lookup["figures"] if figure["paper_id"] == paper_id],
            key=lambda figure: (figure["image_sha256"], figure["figure_id"]),
        )
        figures = []
        for figure in figures_with_paper:
            figure = dict(figure)
            figure.pop("paper_id", None)
            figures.append(figure)
        first = paper_rows[0][1]
        papers_by_id[paper_id] = {
            "paper_id": paper_id,
            "markdown_relpath": markdown_relpath.as_posix(),
            "markdown_sha256": _sha256_bytes(markdown.encode("utf-8")),
            "source": {
                "source_paper_dataset": first.get("source_paper_dataset"),
                "source_paper_config": first.get("source_paper_config"),
                "source_paper_split": first.get("source_paper_split"),
                "source_fig_dataset": first.get("source_fig_dataset"),
                "source_fig_split": first.get("source_fig_split"),
            },
            "figures": figures,
            "queries": {"visual_only": [], "visual_and_text": []},
        }

    papers = [papers_by_id[paper_id] for paper_id in sorted(papers_by_id)]
    lookups = _canonical_lookups(papers)
    textual_clues, textual_report = _align_textual_clues(
        [textual_interpretations] if textual_interpretations is not None else [],
        lookups,
    )
    visual_clues, visual_report = _align_visual_clues(
        [visual_interpretations] if visual_interpretations is not None else [],
        lookups,
    )
    paper_ids_with_text = {clue["paper_id"] for clue in textual_clues}
    figure_ids_with_visual = {clue["figure_id"] for clue in visual_clues}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matched_papers": len(papers),
        "unmatched_source_rows": len(unmatched_rows),
        "unmatched_source_row_indices": unmatched_rows,
        "source_figure_rows": len(rows),
        "matched_source_figure_rows": len(matched_rows),
        "canonical_unique_figures": sum(len(paper["figures"]) for paper in papers),
        "duplicate_figure_rows_collapsed": figure_lookup["duplicate_rows_collapsed"],
        **textual_report,
        **visual_report,
        "papers_missing_textual_clues": sum(1 for paper in papers if paper["paper_id"] not in paper_ids_with_text),
        "figures_missing_visual_clues": sum(
            1 for paper in papers for figure in paper["figures"] if figure["figure_id"] not in figure_ids_with_visual
        ),
    }
    write_canonical_papers(papers, output_path)
    write_canonical_clues(textual_clues, canonical_textual_clues_path(output_path))
    write_canonical_clues(visual_clues, canonical_visual_clues_path(output_path))
    (output_path / "migration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return papers, report


def align_canonical_clues(
    canonical_dir: str | Path,
    *,
    textual_inputs: Iterable[str | Path] = (),
    visual_inputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Rewrite canonical clue sidecars so every retained clue points to a valid paper/figure."""
    root = Path(canonical_dir)
    papers = read_canonical_papers(root)
    lookups = _canonical_lookups(papers)
    textual_clues, textual_report = _align_textual_clues(textual_inputs, lookups)
    visual_clues, visual_report = _align_visual_clues(visual_inputs, lookups)
    write_canonical_clues(textual_clues, canonical_textual_clues_path(root))
    write_canonical_clues(visual_clues, canonical_visual_clues_path(root))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matched_papers": len(papers),
        "canonical_unique_figures": len(lookups["figure_to_paper"]),
        **textual_report,
        **visual_report,
        "papers_missing_textual_clues": len(lookups["paper_ids"].difference({clue["paper_id"] for clue in textual_clues})),
        "figures_missing_visual_clues": len(
            set(lookups["figure_to_paper"]).difference({clue["figure_id"] for clue in visual_clues})
        ),
    }
    (root / "clue_alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def validate_canonical_papers(papers: list[dict[str, Any]], *, root: str | Path | None = None) -> None:
    paper_ids: set[str] = set()
    figure_ids: set[str] = set()
    paper_hashes: set[tuple[str, str]] = set()
    root_path = Path(root) if root is not None else None
    for paper in papers:
        if list(paper)[:1] != ["paper_id"]:
            raise ValueError("Canonical paper rows must start with paper_id")
        paper_id = _required_text(paper, "paper_id")
        if paper_id in paper_ids:
            raise ValueError(f"Duplicate paper_id: {paper_id}")
        paper_ids.add(paper_id)
        if "markdown" in paper:
            raise ValueError(f"Paper {paper_id} embeds markdown; expected markdown_relpath only")
        if "textual_clues" in paper or "visual_clues" in paper:
            raise ValueError(f"Paper {paper_id} embeds clues; expected clue sidecars only")
        markdown_sha = _required_text(paper, "markdown_sha256")
        markdown_relpath = _required_text(paper, "markdown_relpath")
        if root_path is not None:
            markdown_path = root_path / markdown_relpath
            if not markdown_path.exists():
                raise FileNotFoundError(f"Canonical markdown does not exist: {markdown_path}")
            actual_sha = sha256_file(markdown_path)
            if actual_sha != markdown_sha:
                raise ValueError(f"Canonical markdown SHA mismatch for paper {paper_id}: {markdown_path}")
        for figure in paper.get("figures", []):
            figure_id = _required_text(figure, "figure_id")
            image_sha = _required_text(figure, "image_sha256")
            if figure_id in figure_ids:
                raise ValueError(f"Duplicate figure_id: {figure_id}")
            figure_ids.add(figure_id)
            paper_hash = (paper_id, image_sha)
            if paper_hash in paper_hashes:
                raise ValueError(f"Duplicate (paper_id, image_sha256): {paper_id}, {image_sha}")
            paper_hashes.add(paper_hash)
            if root_path is not None:
                image_path = root_path / _required_text(figure, "image_relpath")
                if not image_path.exists():
                    raise FileNotFoundError(f"Canonical image does not exist: {image_path}")


def validate_canonical_clues(clues: list[dict[str, Any]], *, kind: str) -> None:
    expected = (
        ["paper_id", "kind", "model", "prompt_id", "prompt_version", "output"]
        if kind == "textual"
        else ["paper_id", "figure_id", "kind", "model", "prompt_id", "prompt_version", "output"]
    )
    forbidden = {
        "record_id",
        "text",
        "provenance",
        "legacy_record_id",
        "source_rows",
        "filename",
        "image_path",
        "image_relpath",
    }
    if kind == "textual":
        forbidden.update({"figure_id", "image_sha256", "image_width", "image_height", "image_mode"})
    for clue in clues:
        if list(clue) != expected:
            raise ValueError(f"Canonical {kind} clue fields must be exactly {expected}")
        if clue.get("kind") != kind:
            raise ValueError(f"Canonical clue kind must be {kind!r}")
        present_forbidden = forbidden.intersection(clue)
        if present_forbidden:
            raise ValueError(f"Canonical {kind} clue contains forbidden fields: {sorted(present_forbidden)}")
        for field_name in expected:
            if not _clean(clue.get(field_name)):
                raise ValueError(f"Canonical {kind} clue is missing {field_name}")


def _canonical_lookups(papers: list[dict[str, Any]]) -> dict[str, Any]:
    paper_ids = {str(paper["paper_id"]) for paper in papers}
    figure_to_paper: dict[str, str] = {}
    legacy_record_to_figure: dict[str, str] = {}
    source_row_to_figure: dict[int, str] = {}
    legacy_record_to_paper: dict[str, str] = {}
    source_row_to_paper: dict[int, str] = {}
    image_sha_to_figures: dict[str, set[str]] = defaultdict(set)
    for paper in papers:
        paper_id = str(paper["paper_id"])
        for figure in paper.get("figures", []):
            figure_id = str(figure["figure_id"])
            figure_to_paper[figure_id] = paper_id
            image_sha_to_figures[str(figure["image_sha256"])].add(figure_id)
            for legacy_record_id in figure.get("legacy_record_ids", []):
                legacy_record_to_figure[str(legacy_record_id)] = figure_id
                legacy_record_to_paper[str(legacy_record_id)] = paper_id
            for source_row in figure.get("source_rows", []):
                source_row_to_figure[int(source_row)] = figure_id
                source_row_to_paper[int(source_row)] = paper_id
    return {
        "paper_ids": paper_ids,
        "figure_to_paper": figure_to_paper,
        "legacy_record_to_figure": legacy_record_to_figure,
        "source_row_to_figure": source_row_to_figure,
        "legacy_record_to_paper": legacy_record_to_paper,
        "source_row_to_paper": source_row_to_paper,
        "image_sha_to_figures": image_sha_to_figures,
    }


def _align_textual_clues(inputs: Iterable[str | Path], lookups: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_paper: dict[str, dict[str, Any]] = {}
    read_count = 0
    unresolved = 0
    matched = 0
    collapsed_same_paper = 0
    for row in _read_clue_inputs(inputs):
        if row.get("kind") != "textual":
            continue
        read_count += 1
        paper_id = _resolve_textual_paper_id(row, lookups)
        output = _clean(row.get("output") or row.get("text"))
        if not paper_id or not output:
            unresolved += 1
            continue
        matched += 1
        clue = _canonicalize_textual_clue(row, paper_id, output)
        if paper_id in by_paper:
            collapsed_same_paper += 1
            by_paper[paper_id] = min(by_paper[paper_id], clue, key=_textual_clue_sort_key)
            continue
        by_paper[paper_id] = clue
    clues = [by_paper[paper_id] for paper_id in sorted(by_paper)]
    return clues, {
        "textual_clues_read": read_count,
        "textual_clue_records_matched": matched,
        "textual_clues_recovered": len(clues),
        "textual_clues_unresolved": unresolved,
        "textual_duplicate_paper_entries_collapsed": collapsed_same_paper,
        "textual_duplicates_collapsed": collapsed_same_paper,
    }


def _align_visual_clues(inputs: Iterable[str | Path], lookups: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seen_by_figure: dict[str, set[str]] = defaultdict(set)
    clues: list[dict[str, Any]] = []
    read_count = 0
    recovered = 0
    unresolved = 0
    duplicates = 0
    for row in _read_clue_inputs(inputs):
        if row.get("kind") != "visual":
            continue
        read_count += 1
        figure_id = _resolve_visual_figure_id(row, lookups)
        output = _clean(row.get("output") or row.get("text"))
        if not figure_id or not output:
            unresolved += 1
            continue
        clue = _canonicalize_visual_clue(row, lookups["figure_to_paper"][figure_id], figure_id, output)
        key = _clue_content_key(clue)
        if key in seen_by_figure[figure_id]:
            duplicates += 1
            continue
        seen_by_figure[figure_id].add(key)
        clues.append(clue)
        recovered += 1
    clues.sort(
        key=lambda clue: (
            clue["paper_id"],
            clue["figure_id"],
            clue["model"],
            clue["prompt_id"],
            clue["prompt_version"],
            clue["output"],
        )
    )
    return clues, {
        "visual_clues_read": read_count,
        "visual_clues_recovered": recovered,
        "visual_clues_unresolved": unresolved,
        "visual_duplicates_collapsed": duplicates,
    }


def _read_clue_inputs(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in inputs:
        path = Path(value)
        if not path.exists():
            continue
        data_path = path / "interpretations.jsonl" if path.is_dir() else path
        if not data_path.exists():
            continue
        rows.extend(read_jsonl_objects(data_path))
    return rows


def _resolve_textual_paper_id(row: dict[str, Any], lookups: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    provenance_metadata = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
    for candidate in (
        row.get("paper_id"),
        row.get("record_id"),
        metadata.get("paper_id"),
        metadata.get("resolved_paper_id"),
        provenance_metadata.get("paper_id"),
        provenance_metadata.get("resolved_paper_id"),
    ):
        paper_id = _clean(candidate)
        if paper_id in lookups["paper_ids"]:
            return paper_id
    for candidate in (row.get("record_id"), metadata.get("legacy_record_id"), provenance.get("legacy_record_id")):
        paper_id = lookups["legacy_record_to_paper"].get(_clean(candidate))
        if paper_id:
            return paper_id
    for source_row in (
        metadata.get("dataset_row"),
        metadata.get("dataset_index"),
        provenance_metadata.get("dataset_row"),
        provenance_metadata.get("dataset_index"),
    ):
        if source_row is None:
            continue
        try:
            paper_id = lookups["source_row_to_paper"].get(int(source_row))
        except (TypeError, ValueError):
            paper_id = None
        if paper_id:
            return paper_id
    return None


def _resolve_visual_figure_id(row: dict[str, Any], lookups: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    provenance_metadata = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
    for candidate in (
        row.get("figure_id"),
        row.get("record_id"),
        metadata.get("figure_id"),
        provenance_metadata.get("figure_id"),
    ):
        figure_id = _clean(candidate)
        if figure_id in lookups["figure_to_paper"]:
            return figure_id
    for candidate in (row.get("record_id"), metadata.get("legacy_record_id"), provenance.get("legacy_record_id")):
        figure_id = lookups["legacy_record_to_figure"].get(_clean(candidate))
        if figure_id:
            return figure_id
    for source_row in (
        metadata.get("dataset_row"),
        metadata.get("dataset_index"),
        provenance_metadata.get("dataset_row"),
        provenance_metadata.get("dataset_index"),
    ):
        if source_row is None:
            continue
        try:
            figure_id = lookups["source_row_to_figure"].get(int(source_row))
        except (TypeError, ValueError):
            figure_id = None
        if figure_id:
            return figure_id
    image_sha = _clean(row.get("image_sha256") or metadata.get("image_sha256") or provenance_metadata.get("image_sha256"))
    figure_ids = lookups["image_sha_to_figures"].get(image_sha, set())
    if len(figure_ids) == 1:
        return next(iter(figure_ids))
    return None


def _canonicalize_textual_clue(row: dict[str, Any], paper_id: str, output: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "kind": "textual",
        "model": str(row.get("model", "")),
        "prompt_id": str(row.get("prompt_id", "")),
        "prompt_version": str(row.get("prompt_version", "")),
        "output": output,
    }


def _canonicalize_visual_clue(row: dict[str, Any], paper_id: str, figure_id: str, output: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "kind": "visual",
        "model": str(row.get("model", "")),
        "prompt_id": str(row.get("prompt_id", "")),
        "prompt_version": str(row.get("prompt_version", "")),
        "output": output,
    }


def _textual_clue_sort_key(clue: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(clue.get("paper_id", "")),
        str(clue.get("model", "")),
        str(clue.get("prompt_id", "")),
        str(clue.get("prompt_version", "")),
        str(clue.get("output") or clue.get("text") or ""),
    )


def figure_id_for(paper_id: str, image_sha256: str) -> str:
    return hashlib.sha256(f"{paper_id}:{image_sha256}".encode("utf-8")).hexdigest()[:24]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_markdown(markdown_root: Path, paper_id: str, markdown: str) -> Path:
    relpath = Path("markdown") / f"{_safe_filename(paper_id)}.md"
    path = markdown_root.parent / relpath
    path.write_text(markdown, encoding="utf-8")
    return relpath


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    return cleaned or hashlib.sha1(value.encode("utf-8")).hexdigest()


def _build_figures(
    matched_rows: list[tuple[int, dict[str, Any]]],
    processed_path: Path,
    output_path: Path,
    image_root: Path,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    record_id_to_figure_id: dict[str, str] = {}
    source_row_to_figure_id: dict[int, str] = {}
    image_sha_to_figure_ids: dict[str, set[str]] = defaultdict(set)
    duplicate_rows_collapsed = 0

    for row_index, row in matched_rows:
        image_path = _resolve_image_path(row, processed_path)
        image_sha = sha256_file(image_path)
        paper_id = str(row["resolved_paper_id"])
        figure_id = figure_id_for(paper_id, image_sha)
        key = (paper_id, image_sha)
        suffix = image_path.suffix or Path(str(row.get("filename") or "")).suffix or ".png"
        canonical_relpath = Path("images") / f"{figure_id}{suffix}"
        canonical_path = output_path / canonical_relpath
        if key not in by_key:
            shutil.copyfile(image_path, canonical_path)
            by_key[key] = {
                "paper_id": paper_id,
                "figure_id": figure_id,
                "filename": str(row.get("filename") or canonical_path.name),
                "image_relpath": canonical_relpath.as_posix(),
                "image_sha256": image_sha,
                "source_rows": [],
                "legacy_record_ids": [],
                "metadata": {
                    "label": row.get("label"),
                    "label_id": row.get("label_id"),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "image_mode": row.get("image_mode"),
                },
            }
        else:
            duplicate_rows_collapsed += 1
        figure = by_key[key]
        figure["source_rows"].append(row_index)
        record_id = _clean(row.get("record_id"))
        if record_id and record_id not in figure["legacy_record_ids"]:
            figure["legacy_record_ids"].append(record_id)
            record_id_to_figure_id[record_id] = figure_id
        source_row_to_figure_id[row_index] = figure_id
        image_sha_to_figure_ids[image_sha].add(figure_id)

    return {
        "figures": list(by_key.values()),
        "record_id_to_figure_id": record_id_to_figure_id,
        "source_row_to_figure_id": source_row_to_figure_id,
        "image_sha_to_figure_ids": {key: value for key, value in image_sha_to_figure_ids.items()},
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
    }


def _clue_content_key(clue: dict[str, Any]) -> str:
    return json.dumps(
        {
            "output": clue.get("output") or clue.get("text"),
            "kind": clue.get("kind"),
            "model": clue.get("model"),
            "prompt_id": clue.get("prompt_id"),
            "prompt_version": clue.get("prompt_version"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _resolve_image_path(row: dict[str, Any], dataset_dir: Path) -> Path:
    image_value = row.get("image_path") or row.get("image_relpath")
    if not image_value:
        raise ValueError(f"Matched row {row.get('record_id')} has no image_path or image_relpath")
    image_path = Path(str(image_value)).expanduser()
    if not image_path.is_absolute():
        image_path = dataset_dir / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image referenced by row {row.get('record_id')} does not exist: {image_path}")
    return image_path.resolve()


def _resolve_extracted_image_path(paper_dir: Path, figure: dict[str, Any]) -> Path | None:
    for key in ("image_path", "image_relpath", "path"):
        value = _clean(figure.get(key))
        if not value:
            continue
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = paper_dir / image_path
        return image_path.resolve()
    return None


def _copy_source_pdf(paper_dir: Path, pdf_root: Path, paper_id: str, paper: dict[str, Any]) -> tuple[str, bool]:
    candidates = sorted(paper_dir.glob("*.pdf"))
    source_pdf = candidates[0] if candidates else None
    if source_pdf is None:
        pdf_path = _clean(paper.get("pdf_path"))
        if pdf_path:
            source_pdf = Path(pdf_path).expanduser()
            if not source_pdf.is_absolute():
                source_pdf = paper_dir / source_pdf
    if source_pdf is None or not source_pdf.exists():
        return "", False
    destination = pdf_root / f"{_safe_filename(paper_id)}.pdf"
    shutil.copyfile(source_pdf, destination)
    return destination.relative_to(pdf_root.parent).as_posix(), True


def _extracted_figure_metadata(figure: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "type",
        "page",
        "page_idx",
        "bbox",
        "caption",
        "footnote",
        "score",
        "image_width",
        "image_height",
    ):
        if key in figure:
            metadata[key] = figure[key]
    return metadata


def _paper_to_dict(paper: dict[str, Any] | CanonicalPaper) -> dict[str, Any]:
    return asdict(paper) if isinstance(paper, CanonicalPaper) else paper


def _required_text(row: dict[str, Any], field: str) -> str:
    value = _clean(row.get(field))
    if not value:
        raise ValueError(f"Missing required field: {field}")
    return value


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text == "nan" else text
