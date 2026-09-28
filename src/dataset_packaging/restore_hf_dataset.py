#!/usr/bin/env python3
"""Restore the original pdf_datasets folder layout from a packaged HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tarfile
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
REQUIRED_METADATA_COLUMNS = {
    "source",
    "original_relative_path",
    "shard",
    "member_path",
    "size_bytes",
    "sha256",
}
COPY_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class RestoreEntry:
    source: str
    original_relative_path: PurePosixPath
    shard: PurePosixPath
    member_path: str
    size_bytes: int
    sha256: str

    def destination(self, target_dir: Path) -> Path:
        return target_dir / self.source / Path(*self.original_relative_path.parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore a downloaded Hugging Face PDF dataset into the pdf_datasets folder layout."
    )
    parser.add_argument("dataset_dir", type=Path, help="Downloaded Hugging Face dataset directory.")
    parser.add_argument("target_dir", type=Path, help="Destination directory for source folders such as ACL and Physics.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PDFs at their restored paths.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of TAR shards to restore concurrently (default: 1).",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA-256 verification while restoring trusted archives.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    report = restore_dataset(
        dataset_dir=args.dataset_dir,
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        workers=args.workers,
        verify_checksum=not args.skip_checksum,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def restore_dataset(
    *,
    dataset_dir: Path,
    target_dir: Path,
    overwrite: bool = False,
    workers: int = 1,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    dataset_dir = dataset_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    metadata_path = dataset_dir / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata parquet: {metadata_path}")

    entries = load_restore_entries(metadata_path)
    destinations = [entry.destination(target_dir) for entry in entries]
    if not overwrite:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite {len(existing)} existing PDF(s), starting with {existing[0]}. "
                "Use --overwrite to replace them."
            )

    by_shard: dict[PurePosixPath, list[RestoreEntry]] = defaultdict(list)
    for entry in entries:
        by_shard[entry.shard].append(entry)

    shard_jobs = sorted(by_shard.items(), key=lambda item: str(item[0]))
    restored = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                restore_shard,
                dataset_dir=dataset_dir,
                target_dir=target_dir,
                shard=shard,
                entries=shard_entries,
                verify_checksum=verify_checksum,
            )
            for shard, shard_entries in shard_jobs
        ]
        for future in as_completed(futures):
            restored += future.result()
            LOGGER.info("Restored %s/%s PDFs", restored, len(entries))

    LOGGER.info("Restored %s PDFs into %s", restored, target_dir)
    return {
        "dataset_dir": str(dataset_dir),
        "target_dir": str(target_dir),
        "restored_pdf_count": restored,
        "sources": sorted({entry.source for entry in entries}),
        "checksum_verified": verify_checksum,
        "workers": workers,
    }


def load_restore_entries(metadata_path: Path) -> list[RestoreEntry]:
    metadata = pd.read_parquet(metadata_path)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(sorted(missing))}")

    entries: list[RestoreEntry] = []
    destinations: set[tuple[str, PurePosixPath]] = set()
    for row_number, row in enumerate(metadata.to_dict(orient="records"), start=1):
        source = source_name(row["source"], row_number)
        relative_path = relative_posix_path(row["original_relative_path"], "original_relative_path", row_number)
        shard = relative_posix_path(row["shard"], "shard", row_number)
        member_path = tar_member_path(row["member_path"], row_number)
        size_bytes = metadata_size(row["size_bytes"], row_number)
        sha256 = checksum(row["sha256"], row_number)
        destination_key = (source, relative_path)
        if destination_key in destinations:
            raise ValueError(f"Metadata row {row_number} duplicates destination {source}/{relative_path}")
        destinations.add(destination_key)
        entries.append(
            RestoreEntry(
                source=source,
                original_relative_path=relative_path,
                shard=shard,
                member_path=member_path,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
    return entries


def source_name(value: Any, row_number: int) -> str:
    path = relative_posix_path(value, "source", row_number)
    if len(path.parts) != 1:
        raise ValueError(f"Metadata row {row_number} has an invalid source: {value!r}")
    return path.name


def relative_posix_path(value: Any, field: str, row_number: int) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Metadata row {row_number} has an invalid {field}: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Metadata row {row_number} has an unsafe {field}: {value!r}")
    return path


def tar_member_path(value: Any, row_number: int) -> str:
    path = relative_posix_path(value, "member_path", row_number)
    if len(path.parts) != 1 or path.suffix.lower() != ".pdf":
        raise ValueError(f"Metadata row {row_number} has an invalid member_path: {value!r}")
    return str(path)


def metadata_size(value: Any, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"Metadata row {row_number} has an invalid size_bytes: {value!r}")
    size = int(value)
    if size < 0 or size != value:
        raise ValueError(f"Metadata row {row_number} has an invalid size_bytes: {value!r}")
    return size


def checksum(value: Any, row_number: int) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"Metadata row {row_number} has an invalid sha256: {value!r}")
    return value.lower()


def path_under_root(root: Path, relative_path: PurePosixPath, label: str) -> Path:
    path = (root / Path(*relative_path.parts)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Unsafe {label} path: {relative_path}")
    return path


def restore_shard(
    *,
    dataset_dir: Path,
    target_dir: Path,
    shard: PurePosixPath,
    entries: list[RestoreEntry],
    verify_checksum: bool,
) -> int:
    shard_path = path_under_root(dataset_dir, shard, "shard")
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing shard referenced by metadata: {shard_path}")
    with tarfile.open(shard_path, "r") as archive:
        for entry in entries:
            restore_entry(
                archive,
                entry,
                entry.destination(target_dir),
                verify_checksum=verify_checksum,
            )
    return len(entries)


def restore_entry(
    archive: tarfile.TarFile,
    entry: RestoreEntry,
    destination: Path,
    *,
    verify_checksum: bool,
) -> None:
    try:
        member = archive.getmember(entry.member_path)
    except KeyError as exc:
        raise ValueError(f"Shard is missing {entry.member_path}") from exc
    if not member.isfile() or member.size != entry.size_bytes:
        raise ValueError(f"Unexpected TAR member for {entry.member_path}")

    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read TAR member {entry.member_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        hasher = hashlib.sha256() if verify_checksum else None
        bytes_written = 0
        try:
            while chunk := source.read(COPY_BUFFER_SIZE):
                if hasher is not None:
                    hasher.update(chunk)
                temporary.write(chunk)
                bytes_written += len(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    if bytes_written != entry.size_bytes or (hasher is not None and hasher.hexdigest() != entry.sha256):
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum or size mismatch for {entry.member_path}")
    os.replace(temporary_path, destination)


if __name__ == "__main__":
    raise SystemExit(main())
