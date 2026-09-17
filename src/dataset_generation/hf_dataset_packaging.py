"""Package the shared PDF retrieval corpus for Hugging Face Datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import tarfile
import tempfile
import time
from io import BytesIO
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd

from dataset_generation.jsonl import read_jsonl_objects
from dataset_generation.managed_settings import DEFAULT_SETTINGS_PATH
from dataset_generation.managed_settings import load_managed_settings
from dataset_generation.managed_settings import resolve_path_under_root


LOGGER = logging.getLogger(__name__)

DEFAULT_REPO_ID = "kasys/open-source-scientific-documents"
DEFAULT_OUTPUT_DIR = Path("huggingface_dataset")
DEFAULT_SHARD_SIZE_GB = 1.0
PDF_BUFFER_SIZE = 1024 * 1024
METADATA_COLUMNS = [
    "document_id",
    "source",
    "original_filename",
    "original_relative_path",
    "shard",
    "member_path",
    "size_bytes",
    "sha256",
    "license",
    "title",
    "year",
    "doi",
]
DUPLICATE_COLUMNS = [
    "sha256",
    "canonical_document_id",
    "duplicate_document_id",
    "canonical_source",
    "duplicate_source",
]
EXTRA_SIDECAR_KEYS = (
    "doi",
    "authors",
    "abstract",
    "venue",
    "journal",
    "primary_category",
    "subject_categories",
    "broad_domain",
    "broad_domains",
    "pmcid",
    "pmid",
    "anthology_id",
    "arxiv_id",
)


@dataclass(frozen=True)
class SourceSpec:
    source: str
    path: Path


@dataclass(frozen=True)
class PdfEntry:
    document_id: str
    source: str
    path: Path
    original_filename: str
    original_relative_path: str
    source_metadata: dict[str, Any]
    size_bytes: int


@dataclass(frozen=True)
class ShardPlan:
    source: str
    index: int
    path: Path
    entries: tuple[PdfEntry, ...]

    @property
    def repo_relative_path(self) -> str:
        return str(PurePosixPath("data") / normalize_source_name(self.source) / self.path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, validate, and optionally upload the shared WebDataset PDF corpus "
            "for kasys/open-source-scientific-documents."
        )
    )
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="Managed settings YAML.")
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Folder containing the five source subfolders. Defaults to dataset.root/pdf_datasets from settings.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--shard-size-gb", type=float, default=DEFAULT_SHARD_SIZE_GB)
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="NAME=PATH",
        help="Explicit source folder mapping. Repeat for all sources or to override defaults.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate locally, but do not upload.")
    parser.add_argument("--upload-only", action="store_true", help="Validate and upload an existing output directory.")
    parser.add_argument("--skip-validation", action="store_true", help="With --upload-only, upload without rechecking TAR members, metadata, or source PDF checksums.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only.")
    parser.add_argument("--dry-run", action="store_true", help="Discover and plan without writing shards or uploading.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild completed shards instead of reusing them.")
    parser.add_argument(
        "--hf-cli",
        default="hf",
        help="Hugging Face CLI executable. The command uses `hf upload`, not git commits.",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if args.skip_validation:
        if not args.upload_only or args.dry_run or args.validate_only or args.prepare_only:
            raise ValueError("--skip-validation requires --upload-only and cannot be combined with other modes")
        required = [output_dir / name for name in ("README.md", "metadata.parquet", "duplicates.parquet", "preparation_report.json")]
        required.append(output_dir / "data")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Prepared dataset is missing: {', '.join(missing)}")
        LOGGER.info("Skipping local validation; uploading prepared dataset from %s", output_dir)
        upload_dataset(output_dir=output_dir, repo_id=args.repo_id, hf_cli=args.hf_cli)
        return {"output_dir": str(output_dir), "uploaded_to": args.repo_id, "validation_skipped": True}

    sources = resolve_sources(args)
    if args.dry_run:
        entries, failures = discover_entries(sources, limit_per_source=args.limit_per_source)
        plans = plan_shards(entries, output_dir=output_dir, shard_size_bytes=shard_size_bytes(args.shard_size_gb))
        return dry_run_report(entries, failures, plans, output_dir)

    if args.validate_only:
        return validate_dataset(output_dir=output_dir, sources=sources, limit_per_source=args.limit_per_source).as_dict()

    if args.upload_only:
        validation = validate_dataset(output_dir=output_dir, sources=sources, limit_per_source=args.limit_per_source)
    else:
        validation = prepare_dataset(
            sources=sources,
            output_dir=output_dir,
            shard_size_bytes=shard_size_bytes(args.shard_size_gb),
            force_rebuild=args.force_rebuild,
            limit_per_source=args.limit_per_source,
        )
    if validation.errors:
        raise RuntimeError(f"Validation failed with {len(validation.errors)} error(s); see preparation_report.json.")

    if args.prepare_only:
        return validation.as_dict()

    upload_dataset(output_dir=output_dir, repo_id=args.repo_id, hf_cli=args.hf_cli)
    result = validation.as_dict()
    result["uploaded_to"] = args.repo_id
    return result


def resolve_sources(args: argparse.Namespace) -> list[SourceSpec]:
    if args.sources:
        return [parse_source_mapping(value) for value in args.sources]

    input_root = resolve_default_input_root(args.settings, args.input_root)
    return [SourceSpec(name, input_root / name) for name in ("ACL", "Physics", "Engineering", "Biology", "Medicine")]


def resolve_default_input_root(settings_path: Path, input_root: Path | None) -> Path:
    if input_root is not None:
        return input_root.expanduser().resolve()
    raw, resolved_settings = load_managed_settings(settings_path)
    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("settings dataset section must be a mapping")
    return resolve_path_under_root("pdf_datasets", dataset.get("root"), resolved_settings.parent)


def parse_source_mapping(value: str) -> SourceSpec:
    if "=" not in value:
        raise ValueError(f"Source mapping must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Source mapping must be NAME=PATH, got {value!r}")
    return SourceSpec(name.strip(), Path(path).expanduser().resolve())


def shard_size_bytes(value_gb: float) -> int:
    if value_gb <= 0:
        raise ValueError("--shard-size-gb must be positive")
    return int(value_gb * 1024 * 1024 * 1024)


def discover_entries(sources: Iterable[SourceSpec], *, limit_per_source: int | None = None) -> tuple[list[PdfEntry], list[dict[str, Any]]]:
    entries: list[PdfEntry] = []
    failures: list[dict[str, Any]] = []
    seen_document_ids: set[str] = set()

    for source in sources:
        metadata_index = load_source_metadata(source)
        if not source.path.exists():
            failures.append({"source": source.source, "path": str(source.path), "error": "source directory does not exist"})
            continue

        pdfs = sorted(source.path.rglob("*.pdf"), key=lambda item: item.relative_to(source.path).as_posix())
        if limit_per_source is not None:
            pdfs = pdfs[:limit_per_source]
        LOGGER.info("Discovered %s PDFs under %s", len(pdfs), source.path)

        for pdf_path in pdfs:
            relative = pdf_path.relative_to(source.path).as_posix()
            try:
                size = pdf_path.stat().st_size
                with pdf_path.open("rb") as handle:
                    header = handle.read(5)
                if not header.startswith(b"%PDF"):
                    failures.append(file_failure(source, pdf_path, relative, "invalid_pdf_header"))
                    continue
            except OSError as exc:
                failures.append(file_failure(source, pdf_path, relative, repr(exc)))
                continue

            source_metadata = metadata_index.get(relative) or metadata_index.get(pdf_path.name) or {}
            document_id = make_document_id(source.source, relative, source_metadata)
            while document_id in seen_document_ids:
                document_id = f"{document_id}_{len(seen_document_ids):x}"
            seen_document_ids.add(document_id)
            entries.append(
                PdfEntry(
                    document_id=document_id,
                    source=source.source,
                    path=pdf_path,
                    original_filename=pdf_path.name,
                    original_relative_path=relative,
                    source_metadata=source_metadata,
                    size_bytes=size,
                )
            )
    return entries, failures


def file_failure(source: SourceSpec, path: Path, relative: str, error: str) -> dict[str, Any]:
    return {"source": source.source, "path": str(path), "original_relative_path": relative, "error": error}


def load_source_metadata(source: SourceSpec) -> dict[str, dict[str, Any]]:
    source_name = normalize_source_name(source.source).lower()
    candidates = candidate_manifest_paths(source)
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if path.exists() and path.suffix == ".jsonl":
            rows = read_jsonl_objects(path, missing_ok=True, skip_invalid=True)
            break
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in metadata_keys_for_row(row, source_name):
            index.setdefault(key, row)
    return index


def candidate_manifest_paths(source: SourceSpec) -> list[Path]:
    root = source.path
    for _ in range(3):
        root = root.parent
    data_root = root if root.name == "data" else source.path.parent.parent
    source_name = normalize_source_name(source.source).lower()
    if source_name == "acl":
        return [data_root / "acl_subset" / "papers.jsonl"]
    if source_name in {"physics", "engineering"}:
        return [
            data_root / "arxiv_open_reuse" / "eligible_records.jsonl",
            data_root / "arxiv_open_reuse" / "selected_arxiv_documents.jsonl",
        ]
    if source_name in {"biology", "medicine"}:
        return [
            data_root / "pmc_oa_strict" / "papers.jsonl",
            data_root / "pmc_oa_pilot_10" / "papers.jsonl",
        ]
    return []


def metadata_keys_for_row(row: dict[str, Any], source_name: str) -> list[str]:
    keys: list[str] = []
    if "anthology_id" in row:
        keys.append(f"{row['anthology_id']}.pdf")
    if "pdf_filename" in row:
        keys.append(str(row["pdf_filename"]))
    if "arxiv_id" in row:
        keys.append(f"{row['arxiv_id']}.pdf")
    if "pmcid" in row:
        keys.append(f"{row['pmcid']}.pdf")
        broad = str(row.get("broad_domain") or "")
        if source_name == "medicine" and broad == "Medical/Clinical Research":
            keys.append(f"{row['pmcid']}.pdf")
    if "pdf_path" in row and row["pdf_path"]:
        pdf_path = PurePosixPath(str(row["pdf_path"]))
        keys.append(pdf_path.name)
        if len(pdf_path.parts) >= 2:
            keys.append(str(PurePosixPath(*pdf_path.parts[-2:])))
    return keys


def make_document_id(source: str, relative: str, metadata: dict[str, Any]) -> str:
    natural_id = (
        metadata.get("anthology_id")
        or metadata.get("arxiv_id")
        or metadata.get("pmcid")
        or Path(relative).stem
    )
    digest = hashlib.sha1(f"{source}\0{relative}".encode("utf-8")).hexdigest()[:12]
    return f"{normalize_source_name(source).lower()}_{slugify(str(natural_id))}_{digest}"


def slugify(value: str, *, max_length: int = 80) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return (normalized or "document")[:max_length]


def normalize_source_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized or "source"


def plan_shards(entries: list[PdfEntry], *, output_dir: Path, shard_size_bytes: int) -> list[ShardPlan]:
    by_source: dict[str, list[PdfEntry]] = defaultdict(list)
    for entry in entries:
        by_source[entry.source].append(entry)

    plans: list[ShardPlan] = []
    for source in sorted(by_source):
        current: list[PdfEntry] = []
        current_bytes = 0
        shard_index = 0
        for entry in by_source[source]:
            if current and current_bytes + entry.size_bytes > shard_size_bytes:
                plans.append(make_shard_plan(output_dir, source, shard_index, current))
                shard_index += 1
                current = []
                current_bytes = 0
            current.append(entry)
            current_bytes += entry.size_bytes
        if current:
            plans.append(make_shard_plan(output_dir, source, shard_index, current))
    return plans


def make_shard_plan(output_dir: Path, source: str, shard_index: int, entries: list[PdfEntry]) -> ShardPlan:
    shard_dir = output_dir / "data" / normalize_source_name(source)
    return ShardPlan(source=source, index=shard_index, path=shard_dir / f"shard-{shard_index:05d}.tar", entries=tuple(entries))


def prepare_dataset(
    *,
    sources: list[SourceSpec],
    output_dir: Path,
    shard_size_bytes: int,
    force_rebuild: bool = False,
    limit_per_source: int | None = None,
) -> ValidationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries, failures = discover_entries(sources, limit_per_source=limit_per_source)
    plans = plan_shards(entries, output_dir=output_dir, shard_size_bytes=shard_size_bytes)
    metadata_rows: list[dict[str, Any]] = []

    for plan in plans:
        if not force_rebuild and valid_completed_shard(plan.path):
            LOGGER.info("Skipping completed shard %s", plan.path)
            metadata_rows.extend(load_shard_manifest(plan.path.with_suffix(".manifest.json")))
            continue
        rows = write_shard(plan)
        metadata_rows.extend(rows)
        LOGGER.info("Completed %s with %s PDFs", plan.path, len(rows))

    duplicates = duplicate_rows(metadata_rows)
    write_parquet(output_dir / "metadata.parquet", metadata_rows, columns=METADATA_COLUMNS)
    write_parquet(output_dir / "duplicates.parquet", duplicates, columns=DUPLICATE_COLUMNS)
    write_dataset_card(output_dir / "README.md", metadata_rows, duplicates)
    validation = validate_dataset(
        output_dir=output_dir,
        sources=sources,
        known_failures=failures,
        limit_per_source=limit_per_source,
    )
    return validation


def valid_completed_shard(tar_path: Path) -> bool:
    manifest_path = tar_path.with_suffix(".manifest.json")
    if not tar_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = load_shard_manifest(manifest_path)
        expected = {row["member_path"] for row in manifest}
        expected.update(row["member_path"].replace(".pdf", ".json") for row in manifest)
        with tarfile.open(tar_path, "r") as tar:
            names = set(tar.getnames())
        return expected <= names
    except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError):
        return False


def load_shard_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("metadata_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid shard manifest: {path}")
    return rows


def write_shard(plan: ShardPlan) -> list[dict[str, Any]]:
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    if plan.path.exists():
        plan.path.unlink()
    manifest_path = plan.path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest_path.unlink()
    rows: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(prefix=f".{plan.path.name}.", suffix=".tmp", dir=plan.path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w") as tar:
            for entry in plan.entries:
                sha256 = add_pdf_member(tar, entry)
                member_path = f"{entry.document_id}.pdf"
                row = metadata_row(entry, shard=plan.repo_relative_path, member_path=member_path, sha256=sha256)
                sidecar = sidecar_row(row, entry)
                add_json_member(tar, sidecar, f"{entry.document_id}.json")
                rows.append(row)
        os.replace(tmp_path, plan.path)
        manifest_path.write_text(json.dumps({"metadata_rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
        return rows
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def add_pdf_member(tar: tarfile.TarFile, entry: PdfEntry) -> str:
    hasher = hashlib.sha256()
    info = tarfile.TarInfo(name=f"{entry.document_id}.pdf")
    info.size = entry.size_bytes
    info.mtime = 0

    class HashingReader:
        def __init__(self, path: Path) -> None:
            self.handle = path.open("rb")

        def read(self, size: int = -1) -> bytes:
            chunk = self.handle.read(size)
            if chunk:
                hasher.update(chunk)
            return chunk

        def close(self) -> None:
            self.handle.close()

    reader = HashingReader(entry.path)
    try:
        tar.addfile(info, reader)  # type: ignore[arg-type]
    finally:
        reader.close()
    return hasher.hexdigest()


def add_json_member(tar: tarfile.TarFile, payload: dict[str, Any], member_name: str) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name=member_name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, BytesIO(data))


def metadata_row(entry: PdfEntry, *, shard: str, member_path: str, sha256: str) -> dict[str, Any]:
    metadata = entry.source_metadata
    return {
        "document_id": entry.document_id,
        "source": normalize_source_name(entry.source),
        "original_filename": entry.original_filename,
        "original_relative_path": entry.original_relative_path,
        "shard": shard,
        "member_path": member_path,
        "size_bytes": entry.size_bytes,
        "sha256": sha256,
        "license": extract_license(metadata),
        "title": metadata.get("title"),
        "year": extract_year(metadata),
        "doi": metadata.get("doi"),
    }


def sidecar_row(row: dict[str, Any], entry: PdfEntry) -> dict[str, Any]:
    sidecar = dict(row)
    for key in EXTRA_SIDECAR_KEYS:
        value = entry.source_metadata.get(key)
        if value is not None:
            sidecar[key] = value
    return sidecar


def extract_license(metadata: dict[str, Any]) -> str | None:
    for key in ("normalized_license_url", "license_url", "license_code", "exact_license", "license", "license_family"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def extract_year(metadata: dict[str, Any]) -> int | None:
    for key in ("year", "publication_year"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    for key in ("created", "updated"):
        value = metadata.get(key)
        if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def duplicate_rows(metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_by_sha: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in metadata_rows:
        sha256 = row["sha256"]
        canonical = first_by_sha.get(sha256)
        if canonical is None:
            first_by_sha[sha256] = row
            continue
        duplicates.append(
            {
                "sha256": sha256,
                "canonical_document_id": canonical["document_id"],
                "duplicate_document_id": row["document_id"],
                "canonical_source": canonical["source"],
                "duplicate_source": row["source"],
            }
        )
    return duplicates


def write_parquet(path: Path, rows: list[dict[str, Any]], *, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_parquet(path, index=False)


@dataclass(frozen=True)
class ValidationResult:
    output_dir: Path
    total_pdf_count: int
    packaged_pdf_count: int
    unique_content_count: int
    total_bytes: int
    source_counts: dict[str, int]
    source_bytes: dict[str, int]
    shard_counts: dict[str, int]
    duplicate_count: int
    invalid_or_unreadable: list[dict[str, Any]]
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "total_pdf_count": self.total_pdf_count,
            "packaged_pdf_count": self.packaged_pdf_count,
            "unique_content_count": self.unique_content_count,
            "total_bytes": self.total_bytes,
            "source_counts": self.source_counts,
            "source_bytes": self.source_bytes,
            "shard_counts": self.shard_counts,
            "duplicate_count": self.duplicate_count,
            "invalid_or_unreadable": self.invalid_or_unreadable,
            "errors": self.errors,
        }


def validate_dataset(
    *,
    output_dir: Path,
    sources: list[SourceSpec],
    known_failures: list[dict[str, Any]] | None = None,
    limit_per_source: int | None = None,
) -> ValidationResult:
    started = time.monotonic()
    LOGGER.info("Validation: reading metadata from %s", output_dir)
    errors: list[str] = []
    failures = list(known_failures or [])
    metadata_path = output_dir / "metadata.parquet"
    duplicates_path = output_dir / "duplicates.parquet"

    if not metadata_path.exists():
        errors.append(f"Missing metadata parquet: {metadata_path}")
        metadata_rows: list[dict[str, Any]] = []
    else:
        metadata_frame = pd.read_parquet(metadata_path)
        metadata_rows = metadata_frame.where(pd.notna(metadata_frame), None).to_dict("records")

    if not duplicates_path.exists():
        errors.append(f"Missing duplicates parquet: {duplicates_path}")
        duplicate_count = 0
    else:
        duplicate_count = len(pd.read_parquet(duplicates_path))

    LOGGER.info("Validation: discovering source PDFs")
    expected_entries, discovered_failures = discover_entries(sources, limit_per_source=limit_per_source)
    if not known_failures:
        failures.extend(discovered_failures)
    expected_by_id = {entry.document_id: entry for entry in expected_entries}
    if len(expected_by_id) != len(expected_entries):
        errors.append("Expected document IDs are not unique")

    seen_doc_ids: set[str] = set()
    seen_members: set[tuple[str, str]] = set()
    pdf_member_count = 0
    sidecar_member_count = 0
    members_by_shard: dict[str, set[str]] = {}

    shards = sorted({row.get("shard") for row in metadata_rows if row.get("shard")})
    LOGGER.info("Validation: checking %s TAR shards and their members", len(shards))
    last_progress = time.monotonic()
    for shard_number, shard in enumerate(shards, start=1):
        tar_path = output_dir / str(shard)
        if not tar_path.exists():
            errors.append(f"Metadata references missing shard: {shard}")
            continue
        try:
            with tarfile.open(tar_path, "r") as tar:
                names = set(tar.getnames())
        except tarfile.TarError as exc:
            errors.append(f"Could not iterate TAR {shard}: {exc}")
            continue
        members_by_shard[str(shard)] = names
        pdf_member_count += sum(1 for name in names if name.endswith(".pdf"))
        sidecar_member_count += sum(1 for name in names if name.endswith(".json"))
        for name in names:
            if name.endswith(".pdf") and name[:-4] + ".json" not in names:
                errors.append(f"Missing JSON sidecar for {shard}:{name}")
        now = time.monotonic()
        if now - last_progress >= 5:
            LOGGER.info("Validation: checked %s/%s shards (%.1fs elapsed)", shard_number, len(shards), now - started)
            last_progress = now

    LOGGER.info("Validation: checking metadata and SHA-256 of %s source PDFs", len(metadata_rows))
    checked_bytes = 0
    last_progress = time.monotonic()
    for row_number, row in enumerate(metadata_rows, start=1):
        document_id = row.get("document_id")
        member_path = row.get("member_path")
        shard = row.get("shard")
        if document_id in seen_doc_ids:
            errors.append(f"Duplicate document_id in metadata: {document_id}")
        seen_doc_ids.add(str(document_id))
        member_key = (str(shard), str(member_path))
        if member_key in seen_members:
            errors.append(f"Duplicate member path in metadata: {shard}:{member_path}")
        seen_members.add(member_key)
        if shard and member_path and str(member_path) not in members_by_shard.get(str(shard), set()):
            errors.append(f"Metadata member missing from TAR: {shard}:{member_path}")
        entry = expected_by_id.get(str(document_id))
        if entry is None:
            errors.append(f"Metadata row has no matching discovered PDF: {document_id}")
            continue
        if int(row.get("size_bytes") or -1) != entry.size_bytes:
            errors.append(f"Size mismatch for {document_id}")
        try:
            if str(row.get("sha256")) != sha256_file(entry.path):
                errors.append(f"Checksum mismatch for {document_id}")
            checked_bytes += entry.size_bytes
        except OSError as exc:
            errors.append(f"Could not checksum source for {document_id}: {exc}")
        now = time.monotonic()
        if now - last_progress >= 5:
            LOGGER.info(
                "Validation: checked %s/%s PDFs, %.2f GiB read (%.1fs elapsed)",
                row_number,
                len(metadata_rows),
                checked_bytes / (1024 ** 3),
                now - started,
            )
            last_progress = now

    expected_ids = set(expected_by_id)
    packaged_ids = {str(row.get("document_id")) for row in metadata_rows}
    missing = expected_ids - packaged_ids
    extra = packaged_ids - expected_ids
    if missing:
        errors.append(f"{len(missing)} expected PDFs are missing from metadata")
    if extra:
        errors.append(f"{len(extra)} unexpected metadata rows are present")
    if pdf_member_count != len(metadata_rows):
        errors.append(f"PDF member count {pdf_member_count} does not match metadata rows {len(metadata_rows)}")
    if sidecar_member_count != len(metadata_rows):
        errors.append(f"JSON sidecar count {sidecar_member_count} does not match metadata rows {len(metadata_rows)}")

    source_counts = Counter(str(row.get("source")) for row in metadata_rows)
    source_bytes: Counter[str] = Counter()
    shard_counts: Counter[str] = Counter()
    for row in metadata_rows:
        source = str(row.get("source"))
        source_bytes[source] += int(row.get("size_bytes") or 0)
        shard_counts[source] += 1 if row.get("shard") else 0
    unique_content_count = len({row.get("sha256") for row in metadata_rows})
    total_bytes = sum(int(row.get("size_bytes") or 0) for row in metadata_rows)

    result = ValidationResult(
        output_dir=output_dir,
        total_pdf_count=len(expected_entries) + len(failures),
        packaged_pdf_count=len(metadata_rows),
        unique_content_count=unique_content_count,
        total_bytes=total_bytes,
        source_counts=dict(sorted(source_counts.items())),
        source_bytes=dict(sorted(source_bytes.items())),
        shard_counts=dict(sorted(shard_counts.items())),
        duplicate_count=duplicate_count,
        invalid_or_unreadable=failures,
        errors=errors,
    )
    write_report(output_dir / "preparation_report.json", result.as_dict())
    LOGGER.info(
        "Validation complete: %s PDFs, %.2f GiB checked, %s error(s) in %.1fs",
        len(metadata_rows),
        checked_bytes / (1024 ** 3),
        len(errors),
        time.monotonic() - started,
    )
    return result


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(PDF_BUFFER_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dry_run_report(
    entries: list[PdfEntry],
    failures: list[dict[str, Any]],
    plans: list[ShardPlan],
    output_dir: Path,
) -> dict[str, Any]:
    source_counts = Counter(entry.source for entry in entries)
    return {
        "output_dir": str(output_dir),
        "discovered_pdf_count": len(entries),
        "invalid_or_unreadable_count": len(failures),
        "source_counts": dict(sorted(source_counts.items())),
        "estimated_shard_count": len(plans),
        "estimated_total_bytes": sum(entry.size_bytes for entry in entries),
        "planned_shards": [plan.repo_relative_path for plan in plans[:20]],
    }


def write_dataset_card(path: Path, metadata_rows: list[dict[str, Any]], duplicates: list[dict[str, Any]]) -> None:
    source_counts = Counter(row["source"] for row in metadata_rows)
    license_counts = Counter(row["license"] or "Unavailable" for row in metadata_rows)
    source_lines = "\n".join(f"- `{source}`: {count:,} PDFs" for source, count in sorted(source_counts.items()))
    license_lines = "\n".join(f"- `{license_value}`: {count:,}" for license_value, count in sorted(license_counts.items()))
    path.write_text(
        f"""---
dataset_info:
  features:
    - name: document_id
      dtype: string
    - name: source
      dtype: string
    - name: pdf
      dtype: binary
license: other
---

# Open-Source Scientific Documents

This dataset contains approximately 100,000 open scientific PDF documents packaged as a shared retrieval corpus. Train, validation, and test query sets are expected to reference `document_id` values from this single corpus rather than using separate document splits.

## Sources

{source_lines or "- No packaged PDFs yet."}

The source folders preserve the project corpus identities: ACL computational linguistics papers, arXiv Physics papers, arXiv Engineering papers, PMC OA Biology papers, and PMC OA Medical/Clinical Research papers. The selection and license filtering are performed by the existing project download pipeline before packaging.

## Licensing

License information is document-level when available. Do not assume one blanket license for the entire dataset.

{license_lines or "- No license metadata available."}

ACL Anthology records in this project do not provide a per-document license field, so their license metadata is nullable. arXiv open-reuse records preserve their selected Creative Commons license URL where available. PMC OA records preserve article-version license codes such as CC BY or CC0 when present.

## Format

PDFs are stored in uncompressed TAR shards under `data/<SOURCE>/shard-xxxxx.tar` using a WebDataset-compatible layout. Each example contains:

```text
DOCUMENT_ID.pdf
DOCUMENT_ID.json
```

The JSON sidecar includes `document_id`, `source`, original filename/path, size, SHA-256 checksum, shard location, and available bibliographic metadata such as title, year, DOI, and document-level license. The global `metadata.parquet` has one row per packaged PDF with:

```text
{", ".join(METADATA_COLUMNS)}
```

`duplicates.parquet` records exact duplicate content by SHA-256. Duplicate files are preserved in the shards; duplicate rows identify the canonical and duplicate document IDs. This build records {len(metadata_rows):,} PDFs, {len({row["sha256"] for row in metadata_rows}):,} unique file contents, and {len(duplicates):,} duplicate rows.

## Streaming Shards

```python
from datasets import load_dataset

dataset = load_dataset(
    "webdataset",
    data_files={{
        "ACL": "hf://datasets/{DEFAULT_REPO_ID}/data/ACL/*.tar",
    }},
    split="ACL",
    streaming=True,
)

for example in dataset:
    pdf_bytes = example["pdf"]
    metadata = example["json"]
    break
```

## Locate One PDF

```python
import io
import tarfile

import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "{DEFAULT_REPO_ID}"
document_id = "ACL_..."
metadata = pd.read_parquet("hf://datasets/" + repo_id + "/metadata.parquet")
row = metadata.loc[metadata.document_id == document_id].iloc[0]

shard_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=row.shard)
with tarfile.open(shard_path, "r") as tar:
    pdf_bytes = tar.extractfile(row.member_path).read()
```

## Limitations

Some source records have incomplete bibliographic metadata. Missing license, title, year, DOI, or author fields are left null rather than inferred. TAR shards are uncompressed because PDF files are already compressed.
""",
        encoding="utf-8",
    )


def upload_dataset(*, output_dir: Path, repo_id: str, hf_cli: str) -> None:
    create_cmd = [hf_cli, "repo", "create", repo_id, "--type", "dataset", "--yes"]
    upload_cmd = [hf_cli, "upload", repo_id, str(output_dir), ".", "--repo-type", "dataset"]
    env = os.environ.copy()
    if env.get("HF_XET_HIGH_PERFORMANCE") == "1":
        LOGGER.info("HF_XET_HIGH_PERFORMANCE=1 enabled for upload")
    try:
        started = time.monotonic()
        LOGGER.info("Upload: ensuring dataset repository %s exists", repo_id)
        subprocess.run(create_cmd, check=False, env=env)
        LOGGER.info("Upload: repository check finished in %.1fs; starting hf upload from %s", time.monotonic() - started, output_dir)
        started = time.monotonic()
        subprocess.run(upload_cmd, check=True, env=env)
        LOGGER.info("Upload complete in %.1fs", time.monotonic() - started)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Hugging Face CLI not found: {hf_cli}") from exc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
