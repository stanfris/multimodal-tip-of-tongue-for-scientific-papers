"""Read extracted ACL paper directories directly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("data")
DEFAULT_PREPROCESSED_PAPERS_DIR = DEFAULT_DATA_DIR / "preprocessed"
DEFAULT_CLUES_DIR = DEFAULT_DATA_DIR / "clues"
IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def read_preprocessed_papers(path: str | Path = DEFAULT_PREPROCESSED_PAPERS_DIR) -> list[dict[str, Any]]:
    root = _paper_root(Path(path))
    if not root.exists():
        raise FileNotFoundError(f"Preprocessed papers directory does not exist: {root}")

    papers = []
    for paper_dir in sorted(candidate for candidate in root.iterdir() if candidate.is_dir()):
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
        paper["figures"] = _resolve_figures(paper_dir, paper.get("figures") or [])
        papers.append(paper)
    return papers


def _minimal_paper_record(paper_dir: Path) -> dict[str, Any]:
    pdfs = sorted(paper_dir.glob("*.pdf"))
    metadata = _metadata_from_paper_id(paper_dir.name)
    return {
        "paper_id": paper_dir.name,
        "pdf_path": str(pdfs[0]) if pdfs else None,
        **metadata,
        "figures": [
            {
                "figure_id": image_path.stem,
                "filename": image_path.name,
                "image_relpath": str(image_path.relative_to(paper_dir)),
            }
            for image_path in _iter_image_files(paper_dir / "images")
        ],
    }


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
    clue_path = Path(path)
    if not clue_path.exists():
        return []
    return [json.loads(line) for line in clue_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_clue_row(path: str | Path, row: dict[str, Any]) -> None:
    clue_path = Path(path)
    clue_path.parent.mkdir(parents=True, exist_ok=True)
    with clue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


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
