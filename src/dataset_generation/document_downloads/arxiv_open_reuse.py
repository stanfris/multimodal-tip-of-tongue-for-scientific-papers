"""Build an arXiv PDF corpus from Kaggle snapshot metadata and bulk S3 PDFs.

The Cornell arXiv Kaggle snapshot mirrors arXiv OAI metadata in JSONL form. We
stream that snapshot, apply the strict reusable-license filter before any PDF
work, then retrieve selected PDFs from arXiv's requester-pays S3 bulk PDF tar
files. OAI helpers remain for legacy incremental use, but bulk collection does
not depend on the live OAI endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator


LOGGER = logging.getLogger(__name__)
_LAST_TMUX_STATUS_AT = 0.0

DEFAULT_OUTPUT_DIR = Path("data") / "arxiv_open_reuse"
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_SELECTED_MANIFEST = "selected_arxiv_documents.jsonl"
DEFAULT_ELIGIBLE_MANIFEST = "eligible_records.jsonl"
DEFAULT_KAGGLE_SNAPSHOT_FILENAME = "arxiv-metadata-oai-snapshot.json"
KAGGLE_ARXIV_DATASET = "Cornell-University/arxiv"
DEFAULT_OAI_CACHE = "oai_metadata.jsonl"
DEFAULT_OAI_STATE = "oai_harvest_state.json"
DEFAULT_BULK_CACHE_DIR = "bulk_s3"
DEFAULT_TARGET_COUNT = 20_000
DEFAULT_TARGET_PER_DOMAIN = DEFAULT_TARGET_COUNT
DEFAULT_CATEGORY_PREFIXES = ("physics.", "eess.")
DEFAULT_OAI_SETS: tuple[str, ...] = ()
DEFAULT_OAI_EARLIEST_DATE = "2005-09-16"
DEFAULT_OAI_WINDOW_DAYS = 1
DEFAULT_OAI_HARVEST_MODE = "auto"
DEFAULT_MAX_CONSECUTIVE_OAI_406 = 30
DEFAULT_ALLOWED_LICENSE_LABELS = ("CC BY 4.0",)
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)
DEFAULT_PROGRESS_INTERVAL = 25_000
USER_AGENT = "visual-tot-arxiv-open-reuse/3.0 (mailto:stanfris2.0@gmail.com)"
OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
OBSOLETE_OAI_BASE_URLS = {
    "http://export.arxiv.org/oai2",
    "https://export.arxiv.org/oai2",
    "http://export.arxiv.org/oai2/",
    "https://export.arxiv.org/oai2/",
}
OAI_METADATA_PREFIX = "arXiv"
S3_BUCKET = "arxiv"
S3_PDF_MANIFEST_KEY = "pdf/arXiv_pdf_manifest.xml"
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
PERMANENT_HTTP_STATUS_CODES = {400, 401, 403, 404, 410}
DEFAULT_ARXIV_LICENSE = "https://arxiv.org/licenses/nonexclusive-distrib/1.0"

LICENSE_URL_TO_LABEL = {
    "https://creativecommons.org/licenses/by/1.0": "CC BY 1.0",
    "https://creativecommons.org/licenses/by/2.0": "CC BY 2.0",
    "https://creativecommons.org/licenses/by/2.5": "CC BY 2.5",
    "https://creativecommons.org/licenses/by/3.0": "CC BY 3.0",
    "https://creativecommons.org/licenses/by/4.0": "CC BY 4.0",
    "https://creativecommons.org/licenses/by-sa/1.0": "CC BY-SA 1.0",
    "https://creativecommons.org/licenses/by-sa/2.0": "CC BY-SA 2.0",
    "https://creativecommons.org/licenses/by-sa/2.5": "CC BY-SA 2.5",
    "https://creativecommons.org/licenses/by-sa/3.0": "CC BY-SA 3.0",
    "https://creativecommons.org/licenses/by-sa/4.0": "CC BY-SA 4.0",
    "https://creativecommons.org/publicdomain/zero/1.0": "CC0 1.0",
    "https://creativecommons.org/licenses/publicdomain": "CC0 1.0",
}


@dataclass(frozen=True)
class ArxivBuildResult:
    output_dir: Path
    report_path: Path
    metadata_records: int
    domain_records: int
    eligible_records: int
    downloaded: int
    failed: int


@dataclass(frozen=True)
class BulkPdfChunk:
    filename: str
    first_item: str
    last_item: str
    yymm: str
    md5sum: str | None
    size: int | None


@dataclass(frozen=True)
class OAIHarvestItem:
    record: dict[str, Any] | None
    next_token: str | None
    window_start: date
    window_end: date
    status: str
    error: str | None = None


class HTTPFetchError(RuntimeError):
    def __init__(self, url: str, original_error: Exception, *, status_code: int | None = None) -> None:
        super().__init__(f"Failed to fetch {url}: {original_error}")
        self.url = url
        self.original_error = original_error
        self.status_code = status_code


class OAIHistoricalHarvestUnavailable(RuntimeError):
    """Raised when arXiv OAI refuses historical/full harvest requests."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the Cornell arXiv Kaggle snapshot, select strict open-license physics/eess papers, "
            "and retrieve PDFs from arXiv's bulk S3 tar archives."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help=(
            "Compatibility alias for --snapshot-path. Points to a local arXiv JSONL snapshot."
        ),
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help=(
            f"Path to {DEFAULT_KAGGLE_SNAPSHOT_FILENAME}. Defaults to the copy under --output-dir; "
            "when missing, the Kaggle CLI can download it."
        ),
    )
    parser.add_argument(
        "--download-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the Cornell arXiv Kaggle snapshot with the Kaggle CLI if --snapshot-path is missing.",
    )
    parser.add_argument("--kaggle-cli", default="kaggle", help="Kaggle CLI executable used for snapshot download.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="Write selected manifest and skip PDF retrieval.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--target-per-domain", type=int, help="Compatibility alias for --target-count.")
    parser.add_argument(
        "--category-prefix",
        action="append",
        dest="category_prefixes",
        help="Accepted arXiv category prefix. Repeat to include multiple prefixes. Defaults: physics. and eess.",
    )
    parser.add_argument(
        "--oai-set",
        action="append",
        dest="oai_sets",
        help=(
            "OAI-PMH setSpec to harvest. Repeat for more sets. By default no set is sent, because "
            "arXiv currently returns HTTP 406 for ListRecords requests with set filters; local "
            "category filtering still selects physics/eess records before downloads."
        ),
    )
    parser.add_argument("--oai-base-url", default=OAI_BASE_URL, help="OAI-PMH base URL.")
    parser.add_argument(
        "--oai-harvest-mode",
        choices=("auto", "full", "window"),
        default=DEFAULT_OAI_HARVEST_MODE,
        help=(
            "OAI harvest mode. auto first tries arXiv's documented full ListRecords harvest, "
            "then falls back to bounded windows if the service rejects it."
        ),
    )
    parser.add_argument("--oai-window-days", type=int, default=DEFAULT_OAI_WINDOW_DAYS)
    parser.add_argument("--oai-earliest-date", default=DEFAULT_OAI_EARLIEST_DATE)
    parser.add_argument("--max-consecutive-oai-406", type=int, default=DEFAULT_MAX_CONSECUTIVE_OAI_406)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--allowed-license",
        action="append",
        dest="allowed_licenses",
        help="Allowed paper-level license label or URL. Repeat for more. Default: CC BY 4.0 only.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="Recorded for reproducibility. Selection is deterministic snapshot order; no sampling is used.",
    )
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL)
    parser.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    parser.add_argument("--request-delay-seconds", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bulk-cache-dir", type=Path, help="Cache directory for S3 manifest and tar chunks.")
    parser.add_argument("--keep-bulk-archives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--s3-manifest", type=Path, help="Use an existing arXiv PDF S3 manifest XML file.")
    parser.add_argument("--aws-cli", default="aws", help="AWS CLI executable used for requester-pays S3 downloads.")

    # Kept for old scripts/commands.
    parser.add_argument("--metadata-source", choices=("oai", "snapshot"), default="snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--download-all-eligible", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-eligible-per-domain", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--api-page-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--broad-domain", action="append", dest="broad_domains", help=argparse.SUPPRESS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_count = args.target_per_domain if args.target_per_domain is not None else args.target_count
    if args.stop_after_eligible_per_domain is not None:
        target_count = args.stop_after_eligible_per_domain
    category_prefixes = tuple(args.category_prefixes or DEFAULT_CATEGORY_PREFIXES)
    allowed_licenses = normalize_allowed_license_labels(args.allowed_licenses or DEFAULT_ALLOWED_LICENSE_LABELS)
    result = build_arxiv_open_reuse_corpus(
        metadata_path=args.metadata,
        snapshot_path=args.snapshot_path,
        download_snapshot=args.download_snapshot,
        kaggle_cli=args.kaggle_cli,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        metadata_only=args.metadata_only,
        target_count=target_count,
        category_prefixes=category_prefixes,
        oai_sets=tuple(args.oai_sets or DEFAULT_OAI_SETS),
        oai_base_url=normalize_oai_base_url(args.oai_base_url),
        oai_harvest_mode=args.oai_harvest_mode,
        oai_window_days=args.oai_window_days,
        oai_earliest_date=args.oai_earliest_date,
        max_consecutive_oai_406=args.max_consecutive_oai_406,
        start_year=args.start_year,
        end_year=args.end_year,
        allowed_licenses=allowed_licenses,
        progress_interval=args.progress_interval,
        overwrite_pdfs=args.overwrite_pdfs,
        request_delay_seconds=args.request_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        resume=args.resume,
        bulk_cache_dir=args.bulk_cache_dir,
        keep_bulk_archives=args.keep_bulk_archives,
        s3_manifest_path=args.s3_manifest,
        aws_cli=args.aws_cli,
        metadata_source=args.metadata_source,
        selection_seed=args.selection_seed,
    )
    return {
        "output_dir": str(result.output_dir),
        "report": str(result.report_path),
        "metadata_records": result.metadata_records,
        "domain_records": result.domain_records,
        "eligible_records": result.eligible_records,
        "downloaded": result.downloaded,
        "failed": result.failed,
    }


def build_arxiv_open_reuse_corpus(
    *,
    metadata_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    download_snapshot: bool = True,
    kaggle_cli: str = "kaggle",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pdf_dir: str | Path | None = None,
    metadata_only: bool = False,
    target_count: int = DEFAULT_TARGET_COUNT,
    category_prefixes: tuple[str, ...] = DEFAULT_CATEGORY_PREFIXES,
    oai_sets: tuple[str, ...] = DEFAULT_OAI_SETS,
    oai_base_url: str = OAI_BASE_URL,
    oai_harvest_mode: str = DEFAULT_OAI_HARVEST_MODE,
    oai_window_days: int = DEFAULT_OAI_WINDOW_DAYS,
    oai_earliest_date: str = DEFAULT_OAI_EARLIEST_DATE,
    max_consecutive_oai_406: int = DEFAULT_MAX_CONSECUTIVE_OAI_406,
    start_year: int | None = None,
    end_year: int | None = None,
    allowed_licenses: set[str] | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
    overwrite_pdfs: bool = False,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    timeout_seconds: float = 120.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_workers: int = DEFAULT_MAX_WORKERS,
    resume: bool = True,
    bulk_cache_dir: str | Path | None = None,
    keep_bulk_archives: bool = True,
    s3_manifest_path: str | Path | None = None,
    aws_cli: str = "aws",
    metadata_source: str = "snapshot",
    selection_seed: int | None = None,
) -> ArxivBuildResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(pdf_dir) if pdf_dir is not None else output_path / "pdfs"
    selected_path = output_path / DEFAULT_SELECTED_MANIFEST
    eligible_path = output_path / DEFAULT_ELIGIBLE_MANIFEST
    cache_path = output_path / DEFAULT_OAI_CACHE
    state_path = output_path / DEFAULT_OAI_STATE
    bulk_cache_path = Path(bulk_cache_dir) if bulk_cache_dir is not None else output_path / DEFAULT_BULK_CACHE_DIR
    allowed = allowed_licenses or set(DEFAULT_ALLOWED_LICENSE_LABELS)
    oai_base_url = normalize_oai_base_url(oai_base_url)
    if metadata_source == "oai":
        publish_status(f"[arxiv:oai] base_url={oai_base_url}")

    existing_selected = load_jsonl(selected_path) if resume and selected_path.exists() else []
    if len(existing_selected) >= target_count:
        selected_records = existing_selected[:target_count]
        scan_stats = resumed_scan_stats(selected_records)
        metadata_file = Path(metadata_path or snapshot_path or output_path / DEFAULT_KAGGLE_SNAPSHOT_FILENAME)
    elif metadata_source == "snapshot" or metadata_path is not None or snapshot_path is not None:
        metadata_file = resolve_kaggle_snapshot(
            snapshot_path=Path(metadata_path or snapshot_path) if (metadata_path or snapshot_path) else None,
            output_dir=output_path,
            download_snapshot=download_snapshot,
            kaggle_cli=kaggle_cli,
        )
        selected_records, scan_stats = select_documents(
            metadata_file,
            selected_path=selected_path,
            target_count=target_count,
            category_prefixes=category_prefixes,
            start_year=start_year,
            end_year=end_year,
            allowed_licenses=allowed,
            progress_interval=progress_interval,
            resume=resume,
        )
    else:
        selected_records, scan_stats = select_documents_from_oai(
            cache_path=cache_path,
            state_path=state_path,
            selected_path=selected_path,
            target_count=target_count,
            category_prefixes=category_prefixes,
            start_year=start_year,
            end_year=end_year,
            allowed_licenses=allowed,
            progress_interval=progress_interval,
            resume=resume,
            oai_sets=oai_sets,
            oai_base_url=oai_base_url,
            oai_harvest_mode=oai_harvest_mode,
            oai_window_days=oai_window_days,
            oai_earliest_date=oai_earliest_date,
            max_consecutive_oai_406=max_consecutive_oai_406,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    write_jsonl(selected_records, eligible_path)

    download_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "failures": []}
    if not metadata_only:
        download_stats = download_eligible_pdfs(
            selected_records,
            output_dir=pdf_path,
            overwrite=overwrite_pdfs,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_workers=max_workers,
            bulk_cache_dir=bulk_cache_path,
            keep_bulk_archives=keep_bulk_archives,
            s3_manifest_path=Path(s3_manifest_path) if s3_manifest_path is not None else None,
            aws_cli=aws_cli,
        )

    report = build_report(
        metadata_path=Path(metadata_path) if metadata_path is not None else cache_path,
        selected_records=selected_records,
        scan_stats=scan_stats,
        download_stats=download_stats,
        target_count=target_count,
        category_prefixes=category_prefixes,
        start_year=start_year,
        end_year=end_year,
        allowed_licenses=allowed,
        metadata_only=metadata_only,
        oai_sets=oai_sets,
        oai_base_url=oai_base_url,
        oai_harvest_mode=oai_harvest_mode,
        oai_window_days=oai_window_days,
        oai_earliest_date=oai_earliest_date,
        max_consecutive_oai_406=max_consecutive_oai_406,
        bulk_cache_dir=bulk_cache_path,
        metadata_source=metadata_source,
        selection_seed=selection_seed,
    )
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_corpus_report(report)

    return ArxivBuildResult(
        output_dir=output_path,
        report_path=report_path,
        metadata_records=int(scan_stats["total_scanned"]),
        domain_records=int(scan_stats["category_matches"]),
        eligible_records=len(selected_records),
        downloaded=int(download_stats["downloaded"]) + int(download_stats["skipped"]),
        failed=int(download_stats["failed"]),
    )


def resumed_scan_stats(selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_scanned": 0,
        "category_matches": 0,
        "date_matches": 0,
        "allowed_license_matches": len(selected_records),
        "selected": len(selected_records),
        "rejection_counts": {},
        "license_counts": {},
        "resumed_from_manifest": True,
    }


def resolve_kaggle_snapshot(
    *,
    snapshot_path: Path | None,
    output_dir: Path,
    download_snapshot: bool,
    kaggle_cli: str,
) -> Path:
    path = snapshot_path or output_dir / DEFAULT_KAGGLE_SNAPSHOT_FILENAME
    if path.exists():
        publish_status(f"[arxiv:snapshot] reusing {path}")
        return path
    if not download_snapshot:
        raise FileNotFoundError(
            f"arXiv Kaggle snapshot not found: {path}. Download {DEFAULT_KAGGLE_SNAPSHOT_FILENAME} "
            f"from https://www.kaggle.com/datasets/{KAGGLE_ARXIV_DATASET} or rerun with --download-snapshot."
        )
    ensure_kaggle_auth(kaggle_cli)
    download_kaggle_snapshot(path, kaggle_cli=kaggle_cli)
    if not path.exists():
        raise RuntimeError(f"Kaggle download completed but {DEFAULT_KAGGLE_SNAPSHOT_FILENAME} was not found at {path}")
    return path


def ensure_kaggle_auth(kaggle_cli: str) -> None:
    if shutil.which(kaggle_cli) is None:
        raise RuntimeError(
            f"Kaggle CLI executable '{kaggle_cli}' was not found. Install it with `pip install kaggle`, then "
            "configure authentication with a Kaggle API token."
        )
    has_env_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not has_env_credentials and not kaggle_json.exists():
        raise RuntimeError(
            "Kaggle credentials are missing. Create a Kaggle API token at "
            "https://www.kaggle.com/settings/account, then either place it at "
            "~/.kaggle/kaggle.json with file mode 600 or set KAGGLE_USERNAME and KAGGLE_KEY. "
            f"After that, rerun the command so the Kaggle CLI can download {KAGGLE_ARXIV_DATASET}."
        )


def download_kaggle_snapshot(destination: Path, *, kaggle_cli: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_status(f"[arxiv:snapshot] downloading {DEFAULT_KAGGLE_SNAPSHOT_FILENAME} from Kaggle")
    cmd = [
        kaggle_cli,
        "datasets",
        "download",
        "-d",
        KAGGLE_ARXIV_DATASET,
        "-f",
        DEFAULT_KAGGLE_SNAPSHOT_FILENAME,
        "-p",
        str(destination.parent),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle snapshot download failed. Confirm your Kaggle API token is configured "
            "at ~/.kaggle/kaggle.json or via KAGGLE_USERNAME/KAGGLE_KEY, and that you can access "
            f"https://www.kaggle.com/datasets/{KAGGLE_ARXIV_DATASET}."
        ) from exc


def normalize_oai_base_url(base_url: str) -> str:
    normalized = normalize_text(base_url).rstrip("/")
    if normalized in {url.rstrip("/") for url in OBSOLETE_OAI_BASE_URLS}:
        LOGGER.warning(
            "Replacing obsolete arXiv OAI endpoint %s with current endpoint %s",
            base_url,
            OAI_BASE_URL,
        )
        return OAI_BASE_URL
    return normalized or OAI_BASE_URL


def publish_status(message: str) -> None:
    print(message, flush=True)
    display_tmux_status(message, force=True)


def display_tmux_status(message: str, *, force: bool = False, min_interval_seconds: float = 10.0) -> None:
    global _LAST_TMUX_STATUS_AT
    now = time.monotonic()
    if not force and now - _LAST_TMUX_STATUS_AT < min_interval_seconds:
        return
    _LAST_TMUX_STATUS_AT = now
    if not os.environ.get("TMUX") or shutil.which("tmux") is None:
        return
    try:
        subprocess.run(["tmux", "display-message", "-d", "5000", message[:300]], check=False, timeout=2)
    except Exception:  # noqa: BLE001 - tmux status updates are best-effort only.
        return


def iter_arxiv_metadata(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"invalid JSON: {exc}"
                continue
            if not isinstance(record, dict):
                yield line_number, None, "record is not a JSON object"
                continue
            yield line_number, record, None


def select_documents(
    metadata_path: Path,
    *,
    selected_path: Path,
    target_count: int,
    category_prefixes: tuple[str, ...],
    start_year: int | None,
    end_year: int | None,
    allowed_licenses: set[str],
    progress_interval: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = load_jsonl(selected_path) if resume else []
    if not resume:
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text("", encoding="utf-8")
    processor = SelectionProcessor(
        selected=existing,
        selected_path=selected_path,
        target_count=target_count,
        category_prefixes=category_prefixes,
        start_year=start_year,
        end_year=end_year,
        allowed_licenses=allowed_licenses,
        progress_interval=progress_interval,
    )
    for _, raw_record, parse_error in iter_arxiv_metadata(metadata_path):
        if not processor.process(raw_record, parse_error):
            break
    processor.print_progress()
    return processor.selected, processor.stats(resumed=False)


def select_documents_from_oai(
    *,
    cache_path: Path,
    state_path: Path,
    selected_path: Path,
    target_count: int,
    category_prefixes: tuple[str, ...],
    start_year: int | None,
    end_year: int | None,
    allowed_licenses: set[str],
    progress_interval: int,
    resume: bool,
    oai_sets: tuple[str, ...],
    oai_base_url: str,
    oai_harvest_mode: str,
    oai_window_days: int,
    oai_earliest_date: str,
    max_consecutive_oai_406: int,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = load_jsonl(selected_path) if resume else []
    if not resume:
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text("", encoding="utf-8")
        cache_path.write_text("", encoding="utf-8")
        write_json(state_path, {})
    processor = SelectionProcessor(
        selected=selected,
        selected_path=selected_path,
        target_count=target_count,
        category_prefixes=category_prefixes,
        start_year=start_year,
        end_year=end_year,
        allowed_licenses=allowed_licenses,
        progress_interval=progress_interval,
    )

    cached_ids: set[str] = set()
    if resume and cache_path.exists():
        LOGGER.info("Scanning cached OAI metadata from %s", cache_path)
        for _, raw_record, parse_error in iter_arxiv_metadata(cache_path):
            if raw_record and raw_record.get("id"):
                cached_ids.add(str(raw_record["id"]))
            if not processor.process(raw_record, parse_error):
                processor.print_progress()
                return processor.selected, processor.stats(resumed=True)

    state = load_json(state_path) if resume else {}
    skipped_oai_windows: list[dict[str, str]] = []
    if oai_harvest_mode in {"auto", "full"} and not state.get("full_harvest_rejected"):
        try:
            selected_done = harvest_full_oai_stream(
                processor=processor,
                cache_path=cache_path,
                state_path=state_path,
                state=state,
                cached_ids=cached_ids,
                target_count=target_count,
                oai_base_url=oai_base_url,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            if selected_done:
                stats = processor.stats(resumed=resume)
                stats["skipped_oai_windows"] = skipped_oai_windows
                return processor.selected, stats
        except HTTPFetchError as exc:
            if exc.status_code == 406:
                message = (
                    "arXiv OAI rejected the documented full ListRecords harvest with HTTP 406. "
                    "Falling back to bounded datestamp windows."
                )
                state["full_harvest_rejected"] = {
                    "status": 406,
                    "url": exc.url,
                    "at": datetime.now(UTC).isoformat(),
                }
                write_json(state_path, state)
                publish_status(f"[arxiv:oai] {message}")
                if oai_harvest_mode == "full":
                    raise OAIHistoricalHarvestUnavailable(message) from exc
            else:
                raise

    harvest_streams = tuple(oai_sets) if oai_sets else (None,)
    completed_streams = set(state.get("completed_sets") or [])
    for set_spec in harvest_streams:
        stream_key = set_spec or "__all__"
        if stream_key in completed_streams:
            continue
        stream_state = state.setdefault("streams", {}).setdefault(stream_key, {})
        token = state.get("resumption_tokens", {}).get(stream_key)
        until_date = parse_iso_date(stream_state.get("until_date")) or datetime.now(UTC).date()
        earliest_date = parse_iso_date(oai_earliest_date)
        if earliest_date is None:
            raise ValueError(f"Invalid --oai-earliest-date: {oai_earliest_date}")
        last_status_window: tuple[date, date] | None = None
        for item in iter_oai_records(
            base_url=oai_base_url,
            set_spec=set_spec,
            start_resumption_token=token,
            start_until_date=until_date,
            earliest_date=earliest_date,
            window_days=oai_window_days,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ):
            record = item.record
            next_token = item.next_token
            window_start = item.window_start
            window_end = item.window_end
            if last_status_window != (window_start, window_end):
                publish_status(
                    "[arxiv:oai] "
                    f"window={window_start.isoformat()}..{window_end.isoformat()} "
                    f"selected={len(processor.selected):,}/{target_count:,} "
                    f"scanned={processor.scanned:,}"
                )
                last_status_window = (window_start, window_end)
            stream_state["window_start"] = window_start.isoformat()
            stream_state["window_end"] = window_end.isoformat()
            if next_token is None:
                stream_state["until_date"] = (window_start - timedelta(days=1)).isoformat()
            if item.status != "record":
                if item.status == "skipped_http_406":
                    skipped = {
                        "stream": stream_key,
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "status": item.status,
                        "error": item.error or "",
                    }
                    skipped_oai_windows.append(skipped)
                    state.setdefault("skipped_oai_windows", []).append(skipped)
                    publish_status(
                        "[arxiv:oai] "
                        f"skipped {window_start.isoformat()}..{window_end.isoformat()} "
                        f"because arXiv returned HTTP 406"
                    )
                    if count_recent_consecutive_406(state.get("skipped_oai_windows", []), stream_key) >= max_consecutive_oai_406:
                        raise OAIHistoricalHarvestUnavailable(
                            "arXiv OAI has returned HTTP 406 for "
                            f"{max_consecutive_oai_406} consecutive datestamp windows. "
                            "The documented full-harvest and historical-window requests are currently unavailable; "
                            "continuing would just skip backward without discovering enough records."
                        )
                state.setdefault("resumption_tokens", {})[stream_key] = next_token
                state["updated_at"] = datetime.now(UTC).isoformat()
                write_json(state_path, state)
                continue
            assert record is not None
            arxiv_id = normalize_text(record.get("id"))
            if arxiv_id and arxiv_id not in cached_ids:
                append_jsonl(cache_path, record)
                cached_ids.add(arxiv_id)
            state.setdefault("resumption_tokens", {})[stream_key] = next_token
            state["updated_at"] = datetime.now(UTC).isoformat()
            write_json(state_path, state)
            if not processor.process(record, None):
                processor.print_progress()
                stats = processor.stats(resumed=resume)
                stats["skipped_oai_windows"] = skipped_oai_windows
                return processor.selected, stats
        completed_streams.add(stream_key)
        state["completed_sets"] = sorted(completed_streams)
        state.setdefault("resumption_tokens", {}).pop(stream_key, None)
        state["updated_at"] = datetime.now(UTC).isoformat()
        write_json(state_path, state)

    processor.print_progress()
    stats = processor.stats(resumed=resume)
    stats["skipped_oai_windows"] = skipped_oai_windows
    return processor.selected, stats


def harvest_full_oai_stream(
    *,
    processor: "SelectionProcessor",
    cache_path: Path,
    state_path: Path,
    state: dict[str, Any],
    cached_ids: set[str],
    target_count: int,
    oai_base_url: str,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> bool:
    publish_status(
        "[arxiv:oai] trying documented full harvest "
        "(ListRecords without datestamp range)"
    )
    token = state.get("full_harvest_resumption_token")
    for item in iter_oai_full_records(
        base_url=oai_base_url,
        start_resumption_token=token,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    ):
        record = item.record
        state["full_harvest_resumption_token"] = item.next_token
        state["full_harvest_updated_at"] = datetime.now(UTC).isoformat()
        write_json(state_path, state)
        if record is None:
            continue
        arxiv_id = normalize_text(record.get("id"))
        if arxiv_id and arxiv_id not in cached_ids:
            append_jsonl(cache_path, record)
            cached_ids.add(arxiv_id)
        if not processor.process(record, None):
            processor.print_progress()
            return True
        if processor.scanned and processor.scanned % max(1, processor.progress_interval or 10_000) == 0:
            display_tmux_status(
                f"[arxiv:oai-full] selected={len(processor.selected):,}/{target_count:,} "
                f"scanned={processor.scanned:,}"
            )
    state["full_harvest_completed"] = True
    state.pop("full_harvest_resumption_token", None)
    write_json(state_path, state)
    return len(processor.selected) >= target_count


def iter_oai_full_records(
    *,
    base_url: str,
    start_resumption_token: str | None,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> Iterator[OAIHarvestItem]:
    token = start_resumption_token
    synthetic_date = datetime.now(UTC).date()
    while True:
        if token:
            query = {"verb": "ListRecords", "resumptionToken": token}
        else:
            query = {"verb": "ListRecords", "metadataPrefix": OAI_METADATA_PREFIX}
        url = base_url + "?" + urllib.parse.urlencode(query)
        payload = fetch_bytes(
            url,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            headers={"User-Agent": USER_AGENT},
        )
        root = ET.fromstring(payload)
        error = first_child(root, "error")
        if error is not None:
            code = error.attrib.get("code", "unknown")
            message = normalize_text(error.text)
            if code == "noRecordsMatch":
                yield OAIHarvestItem(None, None, synthetic_date, synthetic_date, "no_records")
                break
            raise RuntimeError(f"OAI-PMH full harvest error: {code} {message}")
        list_records = first_child(root, "ListRecords")
        if list_records is None:
            break
        token_element = first_child(list_records, "resumptionToken")
        next_token = normalize_text(token_element.text if token_element is not None else None) or None
        for record_element in children(list_records, "record"):
            metadata = first_child(record_element, "metadata")
            if metadata is None:
                continue
            arxiv_element = next(iter(list(metadata)), None)
            if arxiv_element is None:
                continue
            try:
                yield OAIHarvestItem(
                    record=parse_oai_arxiv_record(arxiv_element),
                    next_token=next_token,
                    window_start=synthetic_date,
                    window_end=synthetic_date,
                    status="record",
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Skipping malformed OAI record: %s", exc)
        if not next_token:
            break
        token = next_token


def count_recent_consecutive_406(rows: Any, stream_key: str) -> int:
    if not isinstance(rows, list):
        return 0
    count = 0
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if row.get("stream") != stream_key:
            continue
        if row.get("status") == "skipped_http_406":
            count += 1
            continue
        break
    return count


class SelectionProcessor:
    def __init__(
        self,
        *,
        selected: list[dict[str, Any]],
        selected_path: Path,
        target_count: int,
        category_prefixes: tuple[str, ...],
        start_year: int | None,
        end_year: int | None,
        allowed_licenses: set[str],
        progress_interval: int,
    ) -> None:
        self.selected = selected[:target_count]
        self.selected_ids = {record["arxiv_id"] for record in self.selected if "arxiv_id" in record}
        self.selected_path = selected_path
        self.target_count = target_count
        self.category_prefixes = category_prefixes
        self.start_year = start_year
        self.end_year = end_year
        self.allowed_licenses = allowed_licenses
        self.progress_interval = progress_interval
        self.rejection_counts: Counter[str] = Counter()
        self.license_counts: Counter[str] = Counter()
        self.scanned = 0
        self.category_matches = 0
        self.date_matches = 0
        self.allowed_matches = len(self.selected)

    def process(self, raw_record: dict[str, Any] | None, parse_error: str | None) -> bool:
        if len(self.selected) >= self.target_count:
            return False
        self.scanned += 1
        if parse_error or raw_record is None:
            self.rejection_counts["malformed_metadata"] += 1
            return True
        candidate, rejection_reason = filter_candidate(
            raw_record,
            category_prefixes=self.category_prefixes,
            start_year=self.start_year,
            end_year=self.end_year,
            allowed_licenses=self.allowed_licenses,
        )
        if rejection_reason == "wrong_category":
            self.rejection_counts[rejection_reason] += 1
        elif rejection_reason == "outside_date_range":
            self.category_matches += 1
            self.rejection_counts[rejection_reason] += 1
        elif rejection_reason:
            self.category_matches += 1
            self.date_matches += 1
            self.rejection_counts[rejection_reason] += 1
            if rejection_reason == "disallowed_license":
                label = normalize_license(raw_record.get("license")).label
                self.license_counts[label or "unknown"] += 1
        else:
            assert candidate is not None
            self.category_matches += 1
            self.date_matches += 1
            self.license_counts[candidate["license"]] += 1
            if candidate["arxiv_id"] not in self.selected_ids:
                append_jsonl(self.selected_path, candidate)
                self.selected.append(candidate)
                self.selected_ids.add(candidate["arxiv_id"])
                self.allowed_matches += 1
        if self.progress_interval > 0 and self.scanned % self.progress_interval == 0:
            self.print_progress()
        return len(self.selected) < self.target_count

    def print_progress(self) -> None:
        print_scan_progress(
            self.scanned,
            self.category_matches,
            self.allowed_matches,
            len(self.selected),
            self.target_count,
            self.rejection_counts,
        )

    def stats(self, *, resumed: bool) -> dict[str, Any]:
        return {
            "total_scanned": self.scanned,
            "category_matches": self.category_matches,
            "date_matches": self.date_matches,
            "allowed_license_matches": self.allowed_matches,
            "selected": len(self.selected),
            "rejection_counts": dict(self.rejection_counts),
            "license_counts": dict(self.license_counts),
            "resumed_from_manifest": resumed,
        }


def iter_oai_records(
    *,
    base_url: str,
    set_spec: str | None,
    start_resumption_token: str | None,
    start_until_date: date,
    earliest_date: date,
    window_days: int,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> Iterator[OAIHarvestItem]:
    token = start_resumption_token
    current_until = start_until_date
    window_size = max(1, window_days)
    while current_until >= earliest_date:
        window_start = max(earliest_date, current_until - timedelta(days=window_size - 1))
        while True:
            if token:
                query = {"verb": "ListRecords", "resumptionToken": token}
            else:
                query = {
                    "verb": "ListRecords",
                    "metadataPrefix": OAI_METADATA_PREFIX,
                    "from": window_start.isoformat(),
                    "until": current_until.isoformat(),
                }
                if set_spec:
                    query["set"] = set_spec
            url = base_url + "?" + urllib.parse.urlencode(query)
            try:
                payload = fetch_bytes(
                    url,
                    request_delay_seconds=request_delay_seconds,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    headers={"User-Agent": USER_AGENT},
                )
            except HTTPFetchError as exc:
                if exc.status_code == 406 and not token:
                    error = f"HTTP 406 for OAI window {window_start.isoformat()}..{current_until.isoformat()}"
                    LOGGER.warning("%s; skipping this datestamp window", error)
                    yield OAIHarvestItem(
                        record=None,
                        next_token=None,
                        window_start=window_start,
                        window_end=current_until,
                        status="skipped_http_406",
                        error=error,
                    )
                    break
                raise
            root = ET.fromstring(payload)
            error = first_child(root, "error")
            if error is not None:
                code = error.attrib.get("code", "unknown")
                message = normalize_text(error.text)
                if code == "noRecordsMatch":
                    yield OAIHarvestItem(
                        record=None,
                        next_token=None,
                        window_start=window_start,
                        window_end=current_until,
                        status="no_records",
                    )
                    break
                raise RuntimeError(
                    f"OAI-PMH error for set {set_spec or 'all'} "
                    f"{window_start.isoformat()}..{current_until.isoformat()}: {code} {message}"
                )
            list_records = first_child(root, "ListRecords")
            if list_records is None:
                break
            token_element = first_child(list_records, "resumptionToken")
            next_token = normalize_text(token_element.text if token_element is not None else None) or None
            for record_element in children(list_records, "record"):
                metadata = first_child(record_element, "metadata")
                if metadata is None:
                    continue
                arxiv_element = next(iter(list(metadata)), None)
                if arxiv_element is None:
                    continue
                try:
                    yield OAIHarvestItem(
                        record=parse_oai_arxiv_record(arxiv_element),
                        next_token=next_token,
                        window_start=window_start,
                        window_end=current_until,
                        status="record",
                    )
                except Exception as exc:  # noqa: BLE001 - keep harvest moving.
                    LOGGER.warning("Skipping malformed OAI record: %s", exc)
            if not next_token:
                token = None
                break
            token = next_token
        if window_start == earliest_date:
            break
        current_until = window_start - timedelta(days=1)


def parse_iso_date(value: Any) -> date | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_oai_arxiv_record(element: ET.Element) -> dict[str, Any]:
    authors = []
    authors_parsed = []
    authors_element = first_child(element, "authors")
    if authors_element is not None:
        for child in children(authors_element, "author"):
            keyname = text_of(child, "keyname")
            forenames = text_of(child, "forenames")
            suffix = text_of(child, "suffix")
            name = " ".join(part for part in (forenames, keyname, suffix) if part)
            if name:
                authors.append(name)
            if keyname or forenames or suffix:
                authors_parsed.append([keyname, forenames, suffix])
    created = text_of(element, "created")
    updated = text_of(element, "updated")
    return {
        "id": text_of(element, "id"),
        "submitter": "",
        "authors": ", ".join(authors),
        "authors_parsed": authors_parsed,
        "title": text_of(element, "title"),
        "comments": text_of(element, "comments") or None,
        "journal-ref": text_of(element, "journal-ref") or text_of(element, "journal_ref") or None,
        "doi": text_of(element, "doi") or None,
        "categories": text_of(element, "categories"),
        "license": text_of(element, "license"),
        "abstract": text_of(element, "abstract"),
        "versions": [{"version": "v1", "created": created}] if created else [],
        "update_date": updated or created,
        "oai_source": "arXiv OAI-PMH",
    }


def filter_candidate(
    record: dict[str, Any],
    *,
    category_prefixes: tuple[str, ...],
    start_year: int | None,
    end_year: int | None,
    allowed_licenses: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        arxiv_id = normalize_text(record.get("id"))
        categories = split_categories(record.get("categories"))
    except Exception:
        return None, "malformed_metadata"
    if not arxiv_id or not categories:
        return None, "malformed_metadata"
    category_matches = matching_categories(categories, category_prefixes)
    if not category_matches:
        return None, "wrong_category"
    latest_version_date = latest_version_created(record)
    year = year_from_date(latest_version_date) or year_from_date(normalize_text(record.get("update_date")))
    if (start_year is not None and (year is None or year < start_year)) or (
        end_year is not None and (year is None or year > end_year)
    ):
        return None, "outside_date_range"

    license_info = normalize_license(record.get("license"))
    if not license_info.normalized_url:
        return None, "missing_license"
    if license_info.label not in allowed_licenses:
        return None, "disallowed_license"
    pdf_filename = pdf_filename_for_arxiv_id(arxiv_id)
    if not pdf_filename:
        return None, "missing_pdf_information"

    return {
        "arxiv_id": arxiv_id,
        "title": normalize_text(record.get("title")),
        "authors": normalize_text(record.get("authors")),
        "authors_parsed": record.get("authors_parsed") or [],
        "abstract": normalize_text(record.get("abstract")),
        "categories": categories,
        "subject_categories": categories,
        "category_matches": category_matches,
        "primary_category": categories[0] if categories else None,
        "license": license_info.label,
        "license_url": normalize_text(record.get("license")),
        "normalized_license_url": license_info.normalized_url,
        "license_family": license_info.family,
        "versions": record.get("versions") or [],
        "latest_version_date": latest_version_date,
        "update_date": normalize_text(record.get("update_date")),
        "doi": normalize_text(record.get("doi")) or None,
        "journal_ref": normalize_text(record.get("journal-ref")) or None,
        "comments": normalize_text(record.get("comments")) or None,
        "pdf_url": pdf_url_for_arxiv_id(arxiv_id),
        "pdf_filename": pdf_filename,
        "bulk_item_key": bulk_item_key_for_arxiv_id(arxiv_id),
    }, None


@dataclass(frozen=True)
class LicenseInfo:
    normalized_url: str | None
    label: str | None
    family: str


def normalize_allowed_license_labels(values: Iterable[str]) -> set[str]:
    labels = set()
    for value in values:
        license_info = normalize_license(value)
        labels.add(license_info.label or normalize_text(value))
    return labels


def normalize_license(value: Any) -> LicenseInfo:
    raw = normalize_text(value)
    if not raw:
        return LicenseInfo(None, None, "missing")
    if raw in set(LICENSE_URL_TO_LABEL.values()):
        return LicenseInfo(None, raw, license_family_for_label(raw))
    normalized_url = normalize_license_url(raw)
    label = LICENSE_URL_TO_LABEL.get(normalized_url)
    if label:
        return LicenseInfo(normalized_url, label, license_family_for_label(label))
    if normalized_url == DEFAULT_ARXIV_LICENSE:
        return LicenseInfo(normalized_url, "arXiv nonexclusive distribution", "arxiv-default")
    return LicenseInfo(normalized_url, None, "missing-or-other")


def normalize_license_url(url: str | None) -> str | None:
    value = normalize_text(url)
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if not hostname:
        return value
    return urllib.parse.urlunsplit(("https", hostname, path, "", ""))


def license_family_for_label(label: str) -> str:
    if label.startswith("CC BY-SA"):
        return "cc-by-sa"
    if label.startswith("CC BY"):
        return "cc-by"
    if label.startswith("CC0"):
        return "cc0"
    return "other"


def split_categories(value: Any) -> list[str]:
    if isinstance(value, str):
        return [category for category in value.split() if category]
    if isinstance(value, list):
        return [normalize_text(category) for category in value if normalize_text(category)]
    return []


def matching_categories(categories: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    prefix_tuple = tuple(prefixes)
    return [category for category in categories if any(category.startswith(prefix) for prefix in prefix_tuple)]


def latest_version_created(record: dict[str, Any]) -> str | None:
    versions = record.get("versions")
    if not isinstance(versions, list) or not versions:
        return None
    latest = versions[-1]
    if not isinstance(latest, dict):
        return None
    return normalize_text(latest.get("created")) or None


def year_from_date(value: str | None) -> int | None:
    value = normalize_text(value)
    if len(value) < 4:
        return None
    for index in range(len(value) - 3):
        fragment = value[index : index + 4]
        if fragment.isdigit():
            year = int(fragment)
            if 1991 <= year <= 2100:
                return year
    return None


def download_eligible_pdfs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
    bulk_cache_dir: Path | None = None,
    keep_bulk_archives: bool = True,
    s3_manifest_path: Path | None = None,
    aws_cli: str = "aws",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = bulk_cache_dir or output_dir.parent / DEFAULT_BULK_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "download_failures.jsonl"
    latest_failures_path = output_dir / "download_failures.latest.jsonl"
    manifest_path = output_dir / "download_manifest.jsonl"
    latest_failures_path.write_text("", encoding="utf-8")
    downloaded = skipped = failed = 0
    failures = []
    total = len(records)
    started_at = time.monotonic()

    pending = []
    for record in records:
        destination = output_dir / record["pdf_filename"]
        if destination.exists() and not overwrite and has_pdf_header(destination):
            skipped += 1
            append_download_manifest(manifest_path, record, destination, "skipped", None)
            continue
        pending.append(record)
    if not pending:
        return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "failures": failures}

    bulk_manifest_path = s3_manifest_path or cache_dir / "arXiv_pdf_manifest.xml"
    ensure_s3_object(
        S3_PDF_MANIFEST_KEY,
        bulk_manifest_path,
        aws_cli=aws_cli,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    chunks = parse_pdf_manifest(bulk_manifest_path)
    records_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pending:
        chunk = find_pdf_chunk(record, chunks)
        if chunk is None:
            failed += 1
            failure = {"arxiv_id": record["arxiv_id"], "pdf_url": record["pdf_url"], "error": "not found in S3 PDF manifest"}
            failures.append(failure)
            append_jsonl(failures_path, failure)
            append_jsonl(latest_failures_path, failure)
            append_download_manifest(manifest_path, record, output_dir / record["pdf_filename"], "failed", failure["error"])
            continue
        records_by_chunk[chunk.filename].append(record)

    completed = skipped + failed
    publish_status(f"[arxiv:s3] extracting PDFs from {len(records_by_chunk):,} required bulk chunk(s)")
    print_download_progress(completed, total, downloaded, skipped, failed, started_at)
    for chunk_filename, chunk_records in records_by_chunk.items():
        chunk_path = cache_dir / Path(chunk_filename).name
        publish_status(f"[arxiv:s3] chunk={chunk_filename} selected_pdfs={len(chunk_records):,}")
        try:
            ensure_s3_object(
                chunk_filename,
                chunk_path,
                aws_cli=aws_cli,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            chunk_results = extract_selected_pdfs_from_tar(
                chunk_path,
                chunk_records,
                output_dir=output_dir,
                overwrite=overwrite,
                max_workers=max_workers,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch moving.
            chunk_results = [
                {
                    "record": record,
                    "destination": output_dir / record["pdf_filename"],
                    "status": "failed",
                    "error": f"{chunk_filename}: {exc}",
                }
                for record in chunk_records
            ]
        for result in chunk_results:
            record = result["record"]
            destination = result["destination"]
            status = result["status"]
            error = result["error"]
            completed += 1
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failure = {"arxiv_id": record["arxiv_id"], "pdf_url": record["pdf_url"], "error": error}
                failures.append(failure)
                append_jsonl(failures_path, failure)
                append_jsonl(latest_failures_path, failure)
                LOGGER.warning("Failed to extract %s: %s", record["arxiv_id"], error)
            append_download_manifest(manifest_path, record, destination, status, error)
            print_download_progress(completed, total, downloaded, skipped, failed, started_at)
        if not keep_bulk_archives:
            chunk_path.unlink(missing_ok=True)
    if total:
        print()
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "failures": failures}


def ensure_s3_object(
    key: str,
    destination: Path,
    *,
    aws_cli: str,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        display_tmux_status(f"[arxiv:s3] reuse {destination.name}", force=True)
        return
    if shutil.which(aws_cli) is None:
        raise RuntimeError(
            f"AWS CLI executable '{aws_cli}' was not found. Configure AWS credentials and install awscli "
            "to download arXiv requester-pays S3 bulk data."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    uri = f"s3://{S3_BUCKET}/{key}"
    publish_status(f"[arxiv:s3] downloading {uri}")
    cmd = [aws_cli, "s3", "cp", uri, str(tmp_path), "--request-payer", "requester", "--only-show-errors"]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        try:
            subprocess.run(cmd, check=True, timeout=timeout_seconds)  # noqa: S603
            tmp_path.replace(destination)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("S3 download failed for %s; retrying in %.1fs", key, wait_seconds)
                time.sleep(wait_seconds)
                continue
            break
    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {uri}: {last_error}") from last_error


def parse_pdf_manifest(path: Path) -> list[BulkPdfChunk]:
    root = ET.parse(path).getroot()
    chunks = []
    for file_element in children(root, "file"):
        filename = text_of(file_element, "filename")
        first_item = text_of(file_element, "first_item")
        last_item = text_of(file_element, "last_item")
        yymm = text_of(file_element, "yymm")
        if not filename or not first_item or not last_item:
            continue
        size_text = text_of(file_element, "size")
        chunks.append(
            BulkPdfChunk(
                filename=filename,
                first_item=first_item,
                last_item=last_item,
                yymm=yymm,
                md5sum=text_of(file_element, "md5sum") or None,
                size=int(size_text) if size_text.isdigit() else None,
            )
        )
    return chunks


def find_pdf_chunk(record: dict[str, Any], chunks: list[BulkPdfChunk]) -> BulkPdfChunk | None:
    item_key = record.get("bulk_item_key") or bulk_item_key_for_arxiv_id(str(record["arxiv_id"]))
    yymm = yymm_for_bulk_item_key(item_key)
    candidates = [chunk for chunk in chunks if not yymm or chunk.yymm == yymm]
    for chunk in candidates:
        if chunk.first_item <= item_key <= chunk.last_item:
            return chunk
    if len(candidates) == 1:
        return candidates[0]
    return None


def extract_selected_pdfs_from_tar(
    tar_path: Path,
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
    max_workers: int,
) -> list[dict[str, Any]]:
    del max_workers  # Tar extraction is sequential to avoid sharing one TarFile across threads.
    wanted = {record.get("bulk_item_key") or bulk_item_key_for_arxiv_id(record["arxiv_id"]): record for record in records}
    found: set[str] = set()
    results = []
    with tarfile.open(tar_path) as tar:
        member_by_key = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            key = bulk_item_key_for_tar_member(member.name)
            if key in wanted:
                member_by_key[key] = member
        for key, record in wanted.items():
            destination = output_dir / record["pdf_filename"]
            if destination.exists() and not overwrite and has_pdf_header(destination):
                results.append({"record": record, "destination": destination, "status": "skipped", "error": None})
                found.add(key)
                continue
            member = member_by_key.get(key)
            if member is None:
                continue
            part_path = destination.with_suffix(destination.suffix + ".part")
            source = tar.extractfile(member)
            if source is None:
                results.append({"record": record, "destination": destination, "status": "failed", "error": "empty tar member"})
                found.add(key)
                continue
            with source, part_path.open("wb") as out:
                shutil.copyfileobj(source, out)
            if not has_pdf_header(part_path):
                part_path.unlink(missing_ok=True)
                results.append({"record": record, "destination": destination, "status": "failed", "error": "tar member is not a PDF"})
            else:
                part_path.replace(destination)
                results.append({"record": record, "destination": destination, "status": "downloaded", "error": None})
            found.add(key)
    for key, record in wanted.items():
        if key not in found:
            results.append(
                {
                    "record": record,
                    "destination": output_dir / record["pdf_filename"],
                    "status": "failed",
                    "error": f"PDF {key} not found in {tar_path.name}",
                }
            )
    return results


def append_download_manifest(manifest_path: Path, record: dict[str, Any], destination: Path, status: str, error: str | None) -> None:
    append_jsonl(
        manifest_path,
        {
            "arxiv_id": record["arxiv_id"],
            "pdf_url": record["pdf_url"],
            "pdf_path": str(destination),
            "license_url": record["license_url"],
            "normalized_license_url": record["normalized_license_url"],
            "license": record["license"],
            "license_family": record["license_family"],
            "status": status,
            "error": error,
        },
    )


def fetch_bytes(
    url: str,
    *,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    headers: dict[str, str] | None = None,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        try:
            request = urllib.request.Request(url, headers=headers or {"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in PERMANENT_HTTP_STATUS_CODES:
                break
            if exc.code in TRANSIENT_HTTP_STATUS_CODES and attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
                continue
            break
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
                continue
            break
    status_code = last_error.code if isinstance(last_error, urllib.error.HTTPError) else None
    raise HTTPFetchError(url, last_error or RuntimeError("unknown fetch error"), status_code=status_code) from last_error


def pdf_url_for_arxiv_id(arxiv_id: str) -> str:
    return f"s3://{S3_BUCKET}/pdf bulk archive for {urllib.parse.quote(arxiv_id, safe='/')}"


def pdf_filename_for_arxiv_id(arxiv_id: str) -> str:
    return f"{arxiv_id.replace('/', '_')}.pdf"


def bulk_item_key_for_arxiv_id(arxiv_id: str) -> str:
    unversioned = re.sub(r"v\d+$", "", normalize_text(arxiv_id))
    return unversioned.replace("/", "")


def yymm_for_bulk_item_key(item_key: str) -> str | None:
    match = re.search(r"(\d{4})", item_key)
    return match.group(1) if match else None


def bulk_item_key_for_tar_member(name: str) -> str:
    stem = Path(name).name
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    return stem.replace("/", "")


def has_pdf_header(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def print_scan_progress(
    scanned: int,
    category_matches: int,
    allowed_matches: int,
    selected: int,
    target_count: int,
    rejection_counts: Counter[str],
) -> None:
    message = (
        f"Scanned: {scanned:,} | "
        f"Category matches: {category_matches:,} | "
        f"Allowed-license matches: {allowed_matches:,} | "
        f"Selected: {selected:,} / {target_count:,} | "
        f"Rejections: {dict(sorted(rejection_counts.items()))}"
    )
    print(message)
    display_tmux_status(f"[arxiv:scan] selected={selected:,}/{target_count:,} scanned={scanned:,}")


def print_download_progress(completed: int, total: int, downloaded: int, skipped: int, failed: int, started_at: float) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    percent = (completed / total * 100.0) if total else 100.0
    message = (
        f"{completed}/{total} ({percent:5.1f}%) "
        f"downloaded={downloaded} skipped={skipped} failed={failed} "
        f"rate={rate:0.2f}/s eta={format_duration(eta_seconds)}"
    )
    print(
        f"\r{message}",
        end="",
        flush=True,
    )
    display_tmux_status(f"[arxiv:pdf] {message}")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def build_report(
    *,
    metadata_path: Path,
    selected_records: list[dict[str, Any]],
    scan_stats: dict[str, Any],
    download_stats: dict[str, Any],
    target_count: int,
    category_prefixes: tuple[str, ...],
    start_year: int | None,
    end_year: int | None,
    allowed_licenses: set[str],
    metadata_only: bool,
    oai_sets: tuple[str, ...],
    oai_base_url: str,
    oai_harvest_mode: str,
    oai_window_days: int,
    oai_earliest_date: str,
    max_consecutive_oai_406: int,
    bulk_cache_dir: Path,
    metadata_source: str,
    selection_seed: int | None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": (
                "Cornell arXiv Kaggle metadata snapshot and arXiv bulk S3 PDFs"
                if metadata_source == "snapshot"
                else "arXiv OAI-PMH metadata and arXiv bulk S3 PDFs"
            ),
            "metadata_source": metadata_source,
            "metadata_path": str(metadata_path),
            "kaggle_dataset": KAGGLE_ARXIV_DATASET if metadata_source == "snapshot" else None,
            "snapshot_filename": DEFAULT_KAGGLE_SNAPSHOT_FILENAME if metadata_source == "snapshot" else None,
            "oai_base_url": oai_base_url,
            "oai_sets": list(oai_sets),
            "oai_harvest_mode": oai_harvest_mode,
            "oai_window_days": oai_window_days,
            "oai_earliest_date": oai_earliest_date,
            "max_consecutive_oai_406": max_consecutive_oai_406,
            "metadata_prefix": OAI_METADATA_PREFIX,
            "license_field": "license",
            "s3_bucket": S3_BUCKET,
            "s3_pdf_manifest": f"s3://{S3_BUCKET}/{S3_PDF_MANIFEST_KEY}",
            "bulk_cache_dir": str(bulk_cache_dir),
            "notes": (
                "Paper-level OAI license URLs are filtered before PDF retrieval. PDFs are extracted "
                "from only the arXiv bulk S3 tar chunks selected via the official PDF manifest."
            ),
        },
        "category_prefixes": list(category_prefixes),
        "date_filter": {"start_year": start_year, "end_year": end_year},
        "allowed_licenses": sorted(allowed_licenses),
        "selection": {
            "method": "snapshot_order_first_n",
            "seed": selection_seed,
            "note": "No sampling is used; the first eligible records in JSONL order are selected deterministically.",
        },
        "target_count": target_count,
        "metadata_only": metadata_only,
        "total_scanned": scan_stats["total_scanned"],
        "category_matches": scan_stats["category_matches"],
        "in_requested_date_range": scan_stats["date_matches"],
        "open_license_verified": scan_stats["allowed_license_matches"],
        "accepted_licenses": scan_stats["allowed_license_matches"],
        "rejected_licenses": int(scan_stats["rejection_counts"].get("missing_license", 0))
        + int(scan_stats["rejection_counts"].get("disallowed_license", 0)),
        "selected": len(selected_records),
        "pdf_downloads_successful": int(download_stats.get("downloaded", 0)) + int(download_stats.get("skipped", 0)),
        "pdf_downloads_failed": int(download_stats.get("failed", 0)),
        "successfully_downloaded_this_run": int(download_stats.get("downloaded", 0)),
        "already_present_pdfs": int(download_stats.get("skipped", 0)),
        "rejection_counts": scan_stats["rejection_counts"],
        "license_counts": scan_stats["license_counts"],
        "skipped_oai_windows": scan_stats.get("skipped_oai_windows", []),
        "selected_license_counts": dict(sorted(Counter(record["license"] for record in selected_records).items())),
        "selected_category_counts": dict(sorted(count_by_category(selected_records).items())),
    }


def print_corpus_report(report: dict[str, Any]) -> None:
    print("arXiv open-reuse corpus report")
    print(f"Total scanned:               {report['total_scanned']:,}")
    print(f"Physics/engineering:         {report['category_matches']:,}")
    print(f"In requested date range:     {report['in_requested_date_range']:,}")
    print(f"Open-license verified:       {report['open_license_verified']:,}")
    print(f"Selected:                    {report['selected']:,}")
    print(f"PDF downloads successful:    {report['pdf_downloads_successful']:,}")
    print(f"PDF downloads failed:        {report['pdf_downloads_failed']:,}")
    print("Rejections:")
    for reason, count in sorted(report["rejection_counts"].items()):
        print(f"  {reason}: {count:,}")


def count_by_category(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("categories", []))
    return counts


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if local_name(child.tag) == name]


def first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def text_of(element: ET.Element, name: str) -> str:
    child = first_child(element, name)
    return normalize_text(child.text) if child is not None else ""


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object on line {line_number} of {path}")
        records.append(record)
    return records


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    try:
        result = run(args)
    except OAIHistoricalHarvestUnavailable as exc:
        raise SystemExit(f"arXiv OAI historical harvest unavailable: {exc}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
