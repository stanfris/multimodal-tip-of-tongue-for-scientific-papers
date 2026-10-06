"""Download Hub components and restore the local data directory."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from dataset_packaging.restore_hf_dataset import restore_dataset


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO_ID = "kasys/open-source-scientific-documents"
COMPONENTS = ("pdfs", "preprocessed", "clues", "query_collections", "splits")
BUILD_ARTIFACTS = ["duplicates.parquet", "preparation_report.json", "*.manifest.json"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face dataset components and restore them under DATA_ROOT (all components by default)."
    )
    parser.add_argument("data_root", nargs="?", type=Path, default=Path("data"), help="Local data root (default: data/); creates pdf_datasets/, preprocessed/, clues/, query_collections/, and splits/ here.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID), help="Hub dataset repository (default: %(default)s).")
    selection = parser.add_argument_group("components (all by default; flags select a subset)")
    for component in COMPONENTS:
        selection.add_argument(f"--{component.replace('_', '-')}", action="store_true", help=f"Download and restore {component.replace('_', ' ')}.")
    selection.add_argument("--generated", action="store_true", help="Select preprocessed, clues, query collections, and splits.")
    selection.add_argument("--all", action="store_true", help="Select PDFs and all generated components.")
    parser.add_argument("--workers", type=int, default=1, help="PDF TAR restore workers (default: 1).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing restored files, including changed PDFs.")
    return parser


def selected_components(args: argparse.Namespace) -> set[str]:
    selected = {name for name in COMPONENTS if getattr(args, name)}
    if args.generated:
        selected.update(COMPONENTS[1:])
    if args.all or not selected:
        selected.update(COMPONENTS)
    return selected


def allow_patterns(components: set[str]) -> list[str]:
    patterns: list[str] = []
    if "pdfs" in components:
        patterns.extend(["metadata.parquet", "data/*/shard-*.tar"])
    for name in COMPONENTS[1:]:
        if name in components:
            patterns.append(f"{name}/**")
    return patterns


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    components = selected_components(args)
    if args.workers < 1:
        parser.error("--workers must be positive")
    download_dataset(
        data_root=args.data_root,
        repo_id=args.repo_id,
        components=components,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    return 0


def download_dataset(
    *, data_root: Path, repo_id: str, components: set[str], workers: int = 1, overwrite: bool = False
) -> None:
    if not components or components - set(COMPONENTS):
        raise ValueError(f"Invalid component selection: {components}")
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError(f"Invalid dataset repository ID: {repo_id!r}")
    download_dir = data_root / ".hf_dataset_download" / repo_id.replace("/", "--")
    # local_dir retains Hub metadata and incomplete transfers for resumed runs.
    from huggingface_hub import snapshot_download

    LOGGER.info("Downloading %s from %s into %s", ", ".join(sorted(components)), repo_id, download_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=allow_patterns(components),
        ignore_patterns=BUILD_ARTIFACTS,
    )
    if "pdfs" in components:
        restore_dataset(
            dataset_dir=download_dir,
            target_dir=data_root / "pdf_datasets",
            overwrite=overwrite,
            resume=True,
            workers=workers,
        )
    for component in COMPONENTS[1:]:
        if component not in components:
            continue
        source_root = download_dir / component
        if not source_root.is_dir():
            raise FileNotFoundError(f"Missing downloaded component: {source_root}")
        destination_root = data_root / component
        if component in {"preprocessed", "clues"}:
            restore_archive_component(source_root, destination_root, overwrite=overwrite)
        else:
            restore_plain_component(source_root, destination_root, overwrite=overwrite)


def safe_destination(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe path: {relative}")
    destination = root.joinpath(*relative.parts)
    current = root
    if current.is_symlink():
        raise ValueError(f"Symlink in destination path: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in destination path: {current}")
    return destination


def member_path(name: str, *, directory: bool = False) -> PurePosixPath:
    if directory and name.endswith("/"):
        name = name[:-1]
    if not name or "\\" in name or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"Unsafe TAR member path: {name!r}")
    return PurePosixPath(name)


def same_content(source, destination: Path, size: int) -> bool:
    if not destination.is_file() or destination.stat().st_size != size:
        return False
    with destination.open("rb") as existing:
        while chunk := source.read(1024 * 1024):
            if existing.read(len(chunk)) != chunk:
                return False
    return True


def copy_stream(source, destination: Path, *, overwrite: bool, size: int) -> bool:
    if destination.exists():
        if same_content(source, destination, size):
            return False
        if not overwrite:
            raise FileExistsError(f"Existing file differs: {destination}; use --overwrite to replace it")
        source.seek(0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            shutil.copyfileobj(source, temporary, length=1024 * 1024)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        if temporary_path.stat().st_size != size:
            raise ValueError(f"Size mismatch while restoring {destination}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def restore_archive_component(source_root: Path, destination_root: Path, *, overwrite: bool = False) -> int:
    restored = 0
    if source_root.is_symlink():
        raise ValueError(f"Symlink component: {source_root}")
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in downloaded component: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if path.suffix != ".tar":
            destination = safe_destination(destination_root, PurePosixPath(*relative.parts))
            with path.open("rb") as source:
                restored += copy_stream(source, destination, overwrite=overwrite, size=path.stat().st_size)
            continue
        if len(relative.parts) != 2 or not path.name.startswith("shard-"):
            raise ValueError(f"Unexpected archive location: {path}")
        domain = relative.parts[0]
        member_path(domain)
        with tarfile.open(path, "r") as archive:
            members = archive.getmembers()
            seen: set[PurePosixPath] = set()
            for member in members:
                relative_member = member_path(member.name, directory=member.isdir())
                if relative_member in seen or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported or duplicate TAR member: {member.name}")
                seen.add(relative_member)
                safe_destination(destination_root, PurePosixPath(domain) / relative_member)
            for member in members:
                destination = safe_destination(destination_root, PurePosixPath(domain) / member_path(member.name, directory=member.isdir()))
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Unreadable TAR member: {member.name}")
                    with source:
                        restored += copy_stream(source, destination, overwrite=overwrite, size=member.size)
    LOGGER.info("Restored %s files into %s", restored, destination_root)
    return restored


def restore_plain_component(source_root: Path, destination_root: Path, *, overwrite: bool = False) -> int:
    restored = 0
    if source_root.is_symlink():
        raise ValueError(f"Symlink component: {source_root}")
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in downloaded component: {path}")
        if path.is_file():
            destination = safe_destination(destination_root, PurePosixPath(*path.relative_to(source_root).parts))
            with path.open("rb") as source:
                restored += copy_stream(source, destination, overwrite=overwrite, size=path.stat().st_size)
    LOGGER.info("Restored %s files into %s", restored, destination_root)
    return restored


if __name__ == "__main__":
    raise SystemExit(main())
