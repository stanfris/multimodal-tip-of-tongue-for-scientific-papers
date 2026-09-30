"""Stage copies of completed MinerU papers for preprocessing."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from preprocessing.preprocessed import select_preprocessed_paper_dirs


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
