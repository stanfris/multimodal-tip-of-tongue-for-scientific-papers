"""Read extracted ACL paper directories directly."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from common.jsonl import append_jsonl_object, read_jsonl_objects


DEFAULT_DATA_DIR = Path("data")
DEFAULT_PREPROCESSED_PAPERS_DIR = DEFAULT_DATA_DIR / "preprocessed"
DEFAULT_CLUES_DIR = DEFAULT_DATA_DIR / "clues"
IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PreprocessedPaperSelection:
    paper_dirs: list[Path]
    expected_count: int
    missing_count: int
    source: str
    missing_documents: list[dict[str, Any]]


def read_preprocessed_papers(path: str | Path = DEFAULT_PREPROCESSED_PAPERS_DIR) -> list[dict[str, Any]]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Preprocessed papers directory does not exist: {root}")

    papers = []
    for paper_dir in iter_preprocessed_paper_dirs(root):
        paper_json_path = paper_dir / "paper.json"
        markdown_path = paper_dir / "markdown.md"
        if not markdown_path.exists():
            continue
        if paper_json_path.exists():
            paper = json.loads(paper_json_path.read_text(encoding="utf-8"))
        else:
            paper = _minimal_paper_record(paper_dir)
        paper_id = str(paper.get("paper_id") or paper_dir.name).strip()
        if not paper_id:
            continue
        paper["paper_id"] = paper_id
        paper["paper_dir"] = str(paper_dir)
        paper["markdown_path"] = str(markdown_path)
        figures = paper.get("figures") or _figures_from_images(paper_dir)
        paper["figures"] = _resolve_figures(paper_dir, figures)
        papers.append(paper)
    return papers


def iter_preprocessed_paper_dirs(path: str | Path) -> list[Path]:
    root = _paper_root(Path(path))
    if not root.exists():
        return []
    return sorted({markdown_path.parent for markdown_path in root.rglob("markdown.md")})


def select_preprocessed_paper_dirs(
    path: str | Path,
    *,
    split_index: Path | None = None,
    split: str = "train+test",
) -> PreprocessedPaperSelection:
    if split_index is None:
        paper_dirs = iter_preprocessed_paper_dirs(path)
        return PreprocessedPaperSelection(paper_dirs, len(paper_dirs), 0, "recursive_scan", [])
    root = _paper_root(Path(path))
    expected = _split_paper_candidates(split_index, split)
    paper_dirs = []
    missing_documents = []
    for candidate in expected:
        paper_dir = _completed_candidate(root, candidate["subset"], tuple(candidate["paper_ids"]))
        if paper_dir is not None:
            paper_dirs.append(paper_dir)
        else:
            missing_documents.append(
                {
                    "stage": "extraction",
                    "status": "missing_extraction",
                    "source_pdf_relpath": candidate["source_pdf_relpath"],
                    "source_paper_dataset": candidate["subset"],
                    "paper_id": candidate["paper_ids"][0],
                    "candidate_paper_ids": candidate["paper_ids"],
                }
            )
    return PreprocessedPaperSelection(
        paper_dirs=paper_dirs,
        expected_count=len(expected),
        missing_count=len(expected) - len(paper_dirs),
        source=str(split_index),
        missing_documents=missing_documents,
    )


def _is_completed_extraction(paper_dir: Path) -> bool:
    return all((paper_dir / name).exists() for name in ("_SUCCESS", "paper.json", "markdown.md"))


def _completed_candidate(root: Path, subset: str, paper_ids: tuple[str, ...]) -> Path | None:
    candidates = []
    for paper_id in paper_ids:
        candidates.extend(
            (
                root / "papers" / subset / paper_id,
                root / subset / "papers" / paper_id,
                root / subset / "papers" / subset / paper_id,
                root / subset / paper_id,
            )
        )
        if root.name == "papers":
            candidates.append(root / subset / paper_id)
    return next((candidate for candidate in candidates if _is_completed_extraction(candidate)), None)


def _split_paper_candidates(split_index: Path, split: str) -> list[dict[str, Any]]:
    index = json.loads(split_index.read_text(encoding="utf-8"))
    split_names = ("train", "test") if split in {"all", "train+test"} else (split,)
    relative_paths: list[Path] = []
    for split_name in split_names:
        rows = index.get(split_name)
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError(f"Split index {split_index} does not contain a string list for {split_name!r}")
        relative_paths.extend(Path(row) for row in rows)

    candidates = []
    for path in relative_paths:
        paper_id = safe_path_name(path.with_suffix("").as_posix())
        collision_id = f"{paper_id}.{sha256(path.as_posix().encode()).hexdigest()[:12]}"
        candidates.append(
            {
                "source_pdf_relpath": path.as_posix(),
                "subset": safe_path_name(path.parts[0] if len(path.parts) > 1 else "Unknown"),
                "paper_ids": (paper_id, collision_id),
            }
        )
    return candidates


def _minimal_paper_record(paper_dir: Path) -> dict[str, Any]:
    pdfs = sorted(paper_dir.glob("*.pdf"))
    metadata = _metadata_from_paper_id(paper_dir.name)
    return {
        "paper_id": paper_dir.name,
        "pdf_path": str(pdfs[0]) if pdfs else None,
        **metadata,
        "figures": _figures_from_images(paper_dir),
    }


def _figures_from_images(paper_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "figure_id": image_path.stem,
            "filename": image_path.name,
            "image_relpath": str(image_path.relative_to(paper_dir)),
        }
        for image_path in _iter_image_files(paper_dir / "images")
    ]


def _iter_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        image_path
        for image_path in images_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _metadata_from_paper_id(paper_id: str) -> dict[str, Any]:
    match = re.match(r"^(?P<year>\d{4})\.(?P<venue>[^.-]+)(?:-[^.]+)?\.\d+$", paper_id)
    if match is None:
        return {}
    return {
        "year": int(match.group("year")),
        "venue": match.group("venue").upper(),
        "volume_id": paper_id.rsplit(".", 1)[0],
    }


def _paper_root(path: Path) -> Path:
    if any((child / "paper.json").exists() for child in path.iterdir() if child.is_dir()) if path.exists() else False:
        return path
    nested = path / "papers"
    return nested if nested.exists() else path


def read_preprocessed_markdown(paper: dict[str, Any]) -> str:
    return Path(str(paper["markdown_path"])).read_text(encoding="utf-8")


def clues_root(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_CLUES_DIR


def paper_clue_dir(root: str | Path, paper_id: str) -> Path:
    return Path(root) / safe_path_name(paper_id)


def textual_clue_path(root: str | Path, paper_id: str) -> Path:
    return paper_clue_dir(root, paper_id) / "base" / "textual_clues.jsonl"


def visual_clue_path(root: str | Path, paper_id: str, figure_id: str) -> Path:
    return paper_clue_dir(root, paper_id) / "images" / f"{safe_path_name(figure_id)}.jsonl"


def read_clue_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_objects(path, missing_ok=True)


def append_clue_row(path: str | Path, row: dict[str, Any], *, append: bool = True) -> None:
    append_jsonl_object(path, row, sort_keys=False, append=append)


def safe_path_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    return cleaned or "unknown"


def _resolve_figures(paper_dir: Path, figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = []
    for index, figure in enumerate(figures):
        figure = dict(figure)
        image_path = _resolve_image_path(paper_dir, figure)
        if image_path is None:
            continue
        figure_id = str(figure.get("figure_id") or f"figure-{index + 1:04d}")
        figure["figure_id"] = figure_id
        figure["image_path"] = str(image_path)
        resolved.append(figure)
    return resolved


def _resolve_image_path(paper_dir: Path, figure: dict[str, Any]) -> Path | None:
    for key in ("image_path", "image_relpath", "path"):
        value = figure.get(key)
        if not value:
            continue
        image_path = Path(str(value)).expanduser()
        if not image_path.is_absolute():
            image_path = paper_dir / image_path
        return image_path.resolve()
    return None
