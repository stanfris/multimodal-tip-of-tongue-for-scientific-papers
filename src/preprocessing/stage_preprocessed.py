"""Stage copies of completed MinerU papers for preprocessing."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from preprocessing.preprocessed import IMAGE_EXTENSIONS, select_preprocessed_paper_dirs


def copy_markdown_and_images(
    processed_root: Path,
    preprocessed_root: Path,
    *,
    split_index: Path | None,
    split: str,
    domains: list[str] | None,
    dry_run: bool = False,
    workers: int = 1,
) -> int:
    """Stage completed MinerU papers as markdown.md and images/ only."""
    if not processed_root.is_dir():
        raise FileNotFoundError(f"Processed directory does not exist: {processed_root}")
    if processed_root.resolve() == preprocessed_root.resolve():
        raise ValueError("Processed and preprocessed directories must differ")
    if domains is not None and (not domains or any(not domain or Path(domain).name != domain for domain in domains)):
        raise ValueError("domains must contain plain directory names")
    selected = select_preprocessed_paper_dirs(processed_root, split_index=split_index, split=split)
    paper_dirs = [
        paper for paper in selected.paper_dirs
        if (domains is None or paper.parent.name in domains)
        and not paper.is_symlink()
        and all((paper / name).is_file() for name in ("_SUCCESS", "paper.json", "markdown.md"))
    ]
    if not paper_dirs:
        raise ValueError(f"No completed papers found for domains {domains} under {processed_root}")

    def stage_one(paper_dir: Path) -> int:
        destination = preprocessed_root / paper_dir.parent.name / paper_dir.name
        if destination.exists():
            if not destination.is_dir() or not (destination / "markdown.md").is_file() or not (destination / "images").is_dir():
                raise FileExistsError(f"Incomplete or incompatible destination: {destination}")
            if any(child.name not in {"markdown.md", "images"} for child in destination.iterdir()):
                raise FileExistsError(f"Destination has additional files: {destination}")
            return 0
        image_sources: dict[Path, Path] = {}
        for source in paper_dir.rglob("*"):
            if not source.is_file() or source.is_symlink() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_parent = next((parent for parent in source.parents if parent.name == "images" and parent.is_relative_to(paper_dir)), None)
            if image_parent is None:
                continue
            relative = source.relative_to(image_parent)
            existing = image_sources.get(relative)
            if existing is not None and existing.read_bytes() != source.read_bytes():
                raise ValueError(f"Conflicting image path {relative} in {paper_dir}")
            image_sources[relative] = source
        if dry_run:
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{paper_dir.name}.", dir=destination.parent) as staging:
            staged = Path(staging) / paper_dir.name
            (staged / "images").mkdir(parents=True)
            shutil.copy2(paper_dir / "markdown.md", staged / "markdown.md")
            for relative, source in image_sources.items():
                target = staged / "images" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            staged.rename(destination)
        return 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return sum(executor.map(stage_one, paper_dirs))


def copy_completed_papers(
    processed_root: Path,
    preprocessed_root: Path,
    *,
    split_index: Path | None,
    split: str,
) -> int:
    """Copy selected paper directories while preserving MinerU's originals."""
    if not processed_root.exists():
        return 0
    if processed_root.resolve() == preprocessed_root.resolve():
        raise ValueError("Processed and preprocessed directories must differ")

    selection = select_preprocessed_paper_dirs(
        processed_root, split_index=split_index, split=split
    )
    copies: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for paper_dir in selection.paper_dirs:
        if paper_dir.is_symlink() or not all(
            (paper_dir / name).exists() for name in ("_SUCCESS", "paper.json", "markdown.md")
        ):
            continue
        subset = paper_dir.parent.name
        if subset == "papers":
            subset = paper_dir.parent.parent.name
        destination = preprocessed_root / subset / paper_dir.name
        if destination.exists():
            if not all(
                (destination / name).exists() for name in ("_SUCCESS", "paper.json", "markdown.md")
            ):
                raise FileExistsError(f"Incomplete preprocessed paper already exists: {destination}")
            continue
        if destination in destinations:
            raise FileExistsError(f"Duplicate preprocessed paper destination: {destination}")
        destinations.add(destination)
        copies.append((paper_dir, destination))

    for paper_dir, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{paper_dir.name}.", dir=destination.parent) as staging:
            staged_paper = Path(staging) / paper_dir.name
            shutil.copytree(paper_dir, staged_paper)
            staged_paper.rename(destination)
    return len(copies)
