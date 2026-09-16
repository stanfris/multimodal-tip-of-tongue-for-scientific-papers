"""Build an arXiv PDF corpus from Cornell's Kaggle arXiv dataset.

The Cornell arXiv Kaggle snapshot mirrors arXiv OAI metadata in JSONL form. We
stream that snapshot, apply the strict reusable-license filter before any PDF
work, then retrieve selected PDFs from the public Google Cloud bucket linked by
the official Kaggle dataset. OAI helpers remain for legacy incremental use, but
bulk collection does not depend on the live OAI endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
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
KAGGLE_ARXIV_GCS_BUCKET = "arxiv-dataset"
KAGGLE_ARXIV_GCS_ROOT_PREFIX = "arxiv"
DEFAULT_OAI_CACHE = "oai_metadata.jsonl"
DEFAULT_OAI_STATE = "oai_harvest_state.json"
DEFAULT_TARGET_COUNT = 20_000
DEFAULT_TARGET_PER_DOMAIN = DEFAULT_TARGET_COUNT
DEFAULT_CATEGORY_PREFIXES = ("physics.", "eess.")
DEFAULT_OAI_SETS: tuple[str, ...] = ()
DEFAULT_OAI_EARLIEST_DATE = "2005-09-16"
DEFAULT_OAI_WINDOW_DAYS = 1
DEFAULT_OAI_HARVEST_MODE = "auto"
DEFAULT_MAX_CONSECUTIVE_OAI_406 = 30
DEFAULT_ALLOWED_LICENSE_LABELS = ("CC BY 4.0",)
DEFAULT_REQUEST_DELAY_SECONDS = 0.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 4))
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
            "and retrieve matching PDFs from the official arXiv Kaggle/GCS dataset."
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
    parser.add_argument(
        "--target-per-domain",
        type=int,
        help=(
            "Select this many records for each requested category prefix. For the default physics./eess. "
            "prefixes, --target-per-domain 30000 selects up to 60,000 records total."
        ),
    )
    parser.add_argument(
        "--target-for-prefix",
        action="append",
        dest="target_for_prefix",
        metavar="PREFIX=COUNT",
        help=(
            "Override the selection target for one category prefix, e.g. "
            "--target-for-prefix eess.=60000. Repeat for multiple prefixes."
        ),
    )
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

    # Kept for old scripts/commands.
    parser.add_argument("--metadata-source", choices=("oai", "snapshot"), default="snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--download-all-eligible", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-eligible-per-domain", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--api-page-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--broad-domain", action="append", dest="broad_domains", help=argparse.SUPPRESS)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    category_prefixes = tuple(args.category_prefixes or DEFAULT_CATEGORY_PREFIXES)
    target_per_domain = args.target_per_domain
    if args.stop_after_eligible_per_domain is not None:
        target_per_domain = args.stop_after_eligible_per_domain
    target_counts = build_target_counts(
        category_prefixes=category_prefixes,
        target_count=args.target_count,
        target_per_domain=target_per_domain,
        target_for_prefix=args.target_for_prefix or (),
    )
    target_count = sum(target_counts.values()) if target_counts else args.target_count
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
        target_per_domain=target_per_domain,
        target_counts=target_counts,
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


def build_target_counts(
    *,
    category_prefixes: tuple[str, ...],
    target_count: int,
    target_per_domain: int | None,
    target_for_prefix: Iterable[str],
) -> dict[str, int] | None:
    overrides = parse_target_for_prefix(target_for_prefix)
    if target_per_domain is None and not overrides:
        return None
    if target_per_domain is None:
        if len(category_prefixes) == 1:
            default_target = target_count
        else:
            default_target = target_count // max(1, len(category_prefixes))
    else:
        default_target = target_per_domain
    counts = {prefix: default_target for prefix in category_prefixes}
    unknown = sorted(set(overrides) - set(category_prefixes))
    if unknown:
        raise ValueError(
            "--target-for-prefix used unknown prefix(es): "
            f"{', '.join(unknown)}. Requested prefixes are: {', '.join(category_prefixes)}"
        )
    counts.update(overrides)
    return counts


def parse_target_for_prefix(values: Iterable[str]) -> dict[str, int]:
    targets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --target-for-prefix value {value!r}; expected PREFIX=COUNT.")
        prefix, count_text = value.split("=", 1)
        prefix = normalize_text(prefix)
        count_text = normalize_text(count_text)
        if not prefix or not count_text.isdigit() or int(count_text) < 0:
            raise ValueError(f"Invalid --target-for-prefix value {value!r}; expected PREFIX=COUNT.")
        targets[prefix] = int(count_text)
    return targets


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
    target_per_domain: int | None = None,
    target_counts: dict[str, int] | None = None,
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
    allowed = allowed_licenses or set(DEFAULT_ALLOWED_LICENSE_LABELS)
    if target_counts is None and target_per_domain is not None:
        target_counts = {prefix: target_per_domain for prefix in category_prefixes}
        target_count = sum(target_counts.values())
    oai_base_url = normalize_oai_base_url(oai_base_url)
    if metadata_source == "oai":
        publish_status(f"[arxiv:oai] base_url={oai_base_url}")

    existing_selected = load_jsonl(selected_path) if resume and selected_path.exists() else []
    if selection_targets_met(existing_selected, target_count, category_prefixes, target_counts):
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
            target_per_domain=target_per_domain,
            target_counts=target_counts,
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
            target_per_domain=target_per_domain,
            target_counts=target_counts,
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

    selected_records = [ensure_kaggle_pdf_fields(record) for record in selected_records]
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
        )

    report = build_report(
        metadata_path=Path(metadata_path) if metadata_path is not None else cache_path,
        selected_records=selected_records,
        scan_stats=scan_stats,
        download_stats=download_stats,
        target_count=target_count,
        target_per_domain=target_per_domain,
        target_counts=target_counts,
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


def selection_targets_met(
    selected_records: list[dict[str, Any]],
    target_count: int,
    category_prefixes: tuple[str, ...],
    target_counts: dict[str, int] | None,
) -> bool:
    if target_counts is None:
        return len(selected_records) >= target_count
    counts = selected_domain_counts(selected_records, category_prefixes)
    return all(counts[prefix] >= target_counts[prefix] for prefix in category_prefixes)


def selected_domain_counts(records: Iterable[dict[str, Any]], category_prefixes: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        domain = normalize_text(record.get("selection_domain"))
        if domain in category_prefixes:
            counts[domain] += 1
            continue
        categories = record.get("categories", [])
        if isinstance(categories, list):
            for prefix in category_prefixes:
                if any(normalize_text(category).startswith(prefix) for category in categories):
                    counts[prefix] += 1
                    break
    return counts


def ensure_kaggle_pdf_fields(record: dict[str, Any]) -> dict[str, Any]:
    pdf_arxiv_id = normalize_text(record.get("pdf_arxiv_id"))
    if not pdf_arxiv_id:
        pdf_arxiv_id = versioned_arxiv_id(str(record["arxiv_id"]), record.get("versions") or [])
    pdf_object = normalize_text(record.get("kaggle_gcs_object")) or kaggle_pdf_object_name(pdf_arxiv_id)
    updated = dict(record)
    updated["pdf_arxiv_id"] = pdf_arxiv_id
    updated["pdf_filename"] = normalize_text(record.get("pdf_filename")) or pdf_filename_for_arxiv_id(pdf_arxiv_id)
    updated["pdf_url"] = kaggle_pdf_media_url(pdf_object)
    updated["kaggle_gcs_bucket"] = KAGGLE_ARXIV_GCS_BUCKET
    updated["kaggle_gcs_object"] = pdf_object
    return updated


def resumed_scan_stats(selected_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_scanned": 0,
        "category_matches": 0,
        "date_matches": 0,
        "allowed_license_matches": len(selected_records),
        "selected": len(selected_records),
        "rejection_counts": {},
        "license_counts": {},
        "selected_domain_counts": {},
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
    target_per_domain: int | None,
    target_counts: dict[str, int] | None,
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
        target_per_domain=target_per_domain,
        target_counts=target_counts,
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
    target_per_domain: int | None,
    target_counts: dict[str, int] | None,
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
        target_per_domain=target_per_domain,
        target_counts=target_counts,
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
        target_per_domain: int | None,
        target_counts: dict[str, int] | None,
        category_prefixes: tuple[str, ...],
        start_year: int | None,
        end_year: int | None,
        allowed_licenses: set[str],
        progress_interval: int,
    ) -> None:
        self.selected = selected[:target_count] if target_counts is None else list(selected)
        self.selected_ids = {record["arxiv_id"] for record in self.selected if "arxiv_id" in record}
        self.selected_path = selected_path
        self.target_count = target_count
        self.target_per_domain = target_per_domain
        self.target_counts = target_counts
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
        self.domain_counts = selected_domain_counts(self.selected, self.category_prefixes)

    def process(self, raw_record: dict[str, Any] | None, parse_error: str | None) -> bool:
        if self.targets_met():
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
                selection_domain = self.next_selection_domain(candidate)
                if selection_domain is None:
                    return True
                candidate["selection_domain"] = selection_domain
                append_jsonl(self.selected_path, candidate)
                self.selected.append(candidate)
                self.selected_ids.add(candidate["arxiv_id"])
                self.allowed_matches += 1
                self.domain_counts[selection_domain] += 1
        if self.progress_interval > 0 and self.scanned % self.progress_interval == 0:
            self.print_progress()
        return not self.targets_met()

    def targets_met(self) -> bool:
        if self.target_counts is None:
            return len(self.selected) >= self.target_count
        return all(
            self.domain_counts[prefix] >= self.target_counts[prefix]
            for prefix in self.category_prefixes
        )

    def next_selection_domain(self, candidate: dict[str, Any]) -> str | None:
        if self.target_counts is None:
            return "combined"
        matches = tuple(candidate.get("category_matches") or ())
        for prefix in self.category_prefixes:
            needs_prefix = self.domain_counts[prefix] < self.target_counts[prefix]
            if needs_prefix and any(str(category).startswith(prefix) for category in matches):
                return prefix
        return None

    def print_progress(self) -> None:
        print_scan_progress(
            self.scanned,
            self.category_matches,
            self.allowed_matches,
            len(self.selected),
            self.target_count,
            self.rejection_counts,
            self.domain_counts if self.target_counts is not None else None,
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
            "selected_domain_counts": dict(self.domain_counts),
            "target_domain_counts": dict(self.target_counts or {}),
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
    versions = record.get("versions") or []
    try:
        pdf_arxiv_id = versioned_arxiv_id(arxiv_id, versions)
        pdf_object = kaggle_pdf_object_name(pdf_arxiv_id)
        pdf_filename = pdf_filename_for_arxiv_id(pdf_arxiv_id)
    except ValueError:
        return None, "missing_pdf_information"
    if not pdf_filename:
        return None, "missing_pdf_information"

    return {
        "arxiv_id": arxiv_id,
        "pdf_arxiv_id": pdf_arxiv_id,
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
        "versions": versions,
        "latest_version_date": latest_version_date,
        "update_date": normalize_text(record.get("update_date")),
        "doi": normalize_text(record.get("doi")) or None,
        "journal_ref": normalize_text(record.get("journal-ref")) or None,
        "comments": normalize_text(record.get("comments")) or None,
        "pdf_url": kaggle_pdf_media_url(pdf_object),
        "kaggle_gcs_bucket": KAGGLE_ARXIV_GCS_BUCKET,
        "kaggle_gcs_object": pdf_object,
        "pdf_filename": pdf_filename,
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "download_failures.jsonl"
    latest_failures_path = output_dir / "download_failures.latest.jsonl"
    unavailable_path = output_dir / "kaggle_unavailable_ids.jsonl"
    latest_unavailable_path = output_dir / "kaggle_unavailable_ids.latest.jsonl"
    manifest_path = output_dir / "download_manifest.jsonl"
    latest_failures_path.write_text("", encoding="utf-8")
    latest_unavailable_path.write_text("", encoding="utf-8")
    downloaded = skipped = failed = unavailable = 0
    failures = []
    unavailable_records = []
    total = len(records)
    started_at = time.monotonic()
    worker_count = max(1, int(max_workers or 1))
    publish_status(
        f"[arxiv:kaggle] downloading selected PDFs from {KAGGLE_ARXIV_GCS_BUCKET} "
        f"with {worker_count} worker(s)"
    )

    pending: list[tuple[dict[str, Any], Path]] = []
    for record in records:
        destination = output_dir / record["pdf_filename"]
        if destination.exists() and not overwrite and has_pdf_header(destination):
            skipped += 1
            append_download_manifest(manifest_path, record, destination, "skipped", None)
            print_download_progress(
                downloaded + skipped + failed + unavailable,
                total,
                downloaded,
                skipped,
                failed,
                unavailable,
                started_at,
            )
            continue
        pending.append((record, destination))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                download_kaggle_pdf,
                record,
                destination,
                overwrite=overwrite,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            ): (record, destination)
            for record, destination in pending
        }
        for future in as_completed(futures):
            record, destination = futures[future]
            try:
                status, error = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the batch moving.
                status, error = "failed", str(exc)
            downloaded, skipped, failed, unavailable = record_download_result(
                record=record,
                destination=destination,
                status=status,
                error=error,
                manifest_path=manifest_path,
                failures_path=failures_path,
                latest_failures_path=latest_failures_path,
                unavailable_path=unavailable_path,
                latest_unavailable_path=latest_unavailable_path,
                failures=failures,
                unavailable_records=unavailable_records,
                downloaded=downloaded,
                skipped=skipped,
                failed=failed,
                unavailable=unavailable,
            )
            print_download_progress(
                downloaded + skipped + failed + unavailable,
                total,
                downloaded,
                skipped,
                failed,
                unavailable,
                started_at,
            )
    if total:
        print()
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "unavailable": unavailable,
        "failures": failures,
        "unavailable_records": unavailable_records,
        "unavailable_report": str(unavailable_path),
    }


def record_download_result(
    *,
    record: dict[str, Any],
    destination: Path,
    status: str,
    error: str | None,
    manifest_path: Path,
    failures_path: Path,
    latest_failures_path: Path,
    unavailable_path: Path,
    latest_unavailable_path: Path,
    failures: list[dict[str, Any]],
    unavailable_records: list[dict[str, Any]],
    downloaded: int,
    skipped: int,
    failed: int,
    unavailable: int,
) -> tuple[int, int, int, int]:
    if status == "downloaded":
        downloaded += 1
    elif status == "skipped":
        skipped += 1
    elif status == "unavailable":
        unavailable += 1
        row = unavailable_report_row(record, error or "not found in Kaggle arXiv PDF bucket")
        unavailable_records.append(row)
        append_jsonl(unavailable_path, row)
        append_jsonl(latest_unavailable_path, row)
    else:
        failed += 1
        failure = {"arxiv_id": record["arxiv_id"], "pdf_url": record["pdf_url"], "error": error}
        failures.append(failure)
        append_jsonl(failures_path, failure)
        append_jsonl(latest_failures_path, failure)
        LOGGER.warning("Failed to download %s: %s", record["arxiv_id"], error)
    append_download_manifest(manifest_path, record, destination, status, error)
    return downloaded, skipped, failed, unavailable


def download_kaggle_pdf(
    record: dict[str, Any],
    destination: Path,
    *,
    overwrite: bool,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[str, str | None]:
    if destination.exists() and not overwrite and has_pdf_header(destination):
        return "skipped", None
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    url = record.get("pdf_url") or kaggle_pdf_media_url(kaggle_pdf_object_name(str(record["pdf_arxiv_id"])))
    try:
        download_url_to_file(
            str(url),
            tmp_path,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    except HTTPFetchError as exc:
        tmp_path.unlink(missing_ok=True)
        if exc.status_code == 404:
            return "unavailable", f"Kaggle object not found: {record.get('kaggle_gcs_object')}"
        return "failed", str(exc)
    if not has_pdf_header(tmp_path):
        tmp_path.unlink(missing_ok=True)
        return "failed", "downloaded object is not a PDF"
    tmp_path.replace(destination)
    return "downloaded", None


def download_url_to_file(
    url: str,
    destination: Path,
    *,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> None:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as file:
                shutil.copyfileobj(response, file)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in PERMANENT_HTTP_STATUS_CODES:
                break
            if exc.code in TRANSIENT_HTTP_STATUS_CODES and attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Download failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
                continue
            break
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Download failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
                continue
            break
    status_code = last_error.code if isinstance(last_error, urllib.error.HTTPError) else None
    raise HTTPFetchError(url, last_error or RuntimeError("unknown download error"), status_code=status_code) from last_error


def unavailable_report_row(record: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "arxiv_id": record.get("arxiv_id"),
        "pdf_arxiv_id": record.get("pdf_arxiv_id"),
        "kaggle_gcs_object": record.get("kaggle_gcs_object"),
        "pdf_url": record.get("pdf_url"),
        "error": error,
    }


def append_download_manifest(manifest_path: Path, record: dict[str, Any], destination: Path, status: str, error: str | None) -> None:
    append_jsonl(
        manifest_path,
        {
            "arxiv_id": record["arxiv_id"],
            "pdf_arxiv_id": record.get("pdf_arxiv_id"),
            "pdf_url": record["pdf_url"],
            "kaggle_gcs_object": record.get("kaggle_gcs_object"),
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


def versioned_arxiv_id(arxiv_id: str, versions: Any) -> str:
    value = normalize_text(arxiv_id)
    if re.search(r"v\d+$", value):
        return value
    version = latest_version_label(versions) or "v1"
    return f"{value}{version}"


def latest_version_label(versions: Any) -> str | None:
    if not isinstance(versions, list) or not versions:
        return None
    for row in reversed(versions):
        if not isinstance(row, dict):
            continue
        version = normalize_text(row.get("version"))
        if re.fullmatch(r"v\d+", version):
            return version
    return None


def kaggle_pdf_object_name(pdf_arxiv_id: str) -> str:
    value = normalize_text(pdf_arxiv_id)
    if "/" in value:
        archive, identifier = value.split("/", 1)
        yymm = yymm_for_arxiv_identifier(identifier)
        return f"{KAGGLE_ARXIV_GCS_ROOT_PREFIX}/{archive}/pdf/{yymm}/{identifier}.pdf"
    yymm = yymm_for_arxiv_identifier(value)
    return f"{KAGGLE_ARXIV_GCS_ROOT_PREFIX}/arxiv/pdf/{yymm}/{value}.pdf"


def yymm_for_arxiv_identifier(identifier: str) -> str:
    unversioned = re.sub(r"v\d+$", "", normalize_text(identifier))
    match = re.match(r"(\d{4})", unversioned)
    if not match:
        raise ValueError(f"Could not derive Kaggle PDF month prefix from arXiv ID: {identifier}")
    return match.group(1)


def kaggle_pdf_media_url(object_name: str) -> str:
    encoded_name = urllib.parse.quote(object_name, safe="")
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{KAGGLE_ARXIV_GCS_BUCKET}/o/"
        f"{encoded_name}?alt=media"
    )


def pdf_filename_for_arxiv_id(arxiv_id: str) -> str:
    return f"{arxiv_id.replace('/', '_')}.pdf"


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
    domain_counts: Counter[str] | None = None,
) -> None:
    domain_message = ""
    if domain_counts:
        domain_message = " | Domains: " + ", ".join(
            f"{domain}={count:,}" for domain, count in sorted(domain_counts.items())
        )
    message = (
        f"Scanned: {scanned:,} | "
        f"Category matches: {category_matches:,} | "
        f"Allowed-license matches: {allowed_matches:,} | "
        f"Selected: {selected:,} / {target_count:,} | "
        f"Rejections: {dict(sorted(rejection_counts.items()))}"
        f"{domain_message}"
    )
    print(message)
    display_tmux_status(f"[arxiv:scan] selected={selected:,}/{target_count:,} scanned={scanned:,}")


def print_download_progress(
    completed: int,
    total: int,
    downloaded: int,
    skipped: int,
    failed: int,
    unavailable: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    percent = (completed / total * 100.0) if total else 100.0
    message = (
        f"{completed}/{total} ({percent:5.1f}%) "
        f"downloaded={downloaded} skipped={skipped} unavailable={unavailable} failed={failed} "
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
    target_per_domain: int | None,
    target_counts: dict[str, int] | None,
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
    metadata_source: str,
    selection_seed: int | None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": (
                "Cornell arXiv Kaggle metadata snapshot and Kaggle/GCS PDFs"
                if metadata_source == "snapshot"
                else "arXiv OAI-PMH metadata and Kaggle/GCS PDFs"
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
            "kaggle_pdf_bucket": KAGGLE_ARXIV_GCS_BUCKET,
            "kaggle_pdf_root_prefix": KAGGLE_ARXIV_GCS_ROOT_PREFIX,
            "notes": (
                "Paper-level license URLs are filtered before PDF retrieval. The downloader uses exact "
                "public GCS object paths from the official Cornell/Kaggle arXiv dataset and never "
                "downloads the complete PDF corpus."
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
        "target_per_domain": target_per_domain,
        "target_domain_counts": target_counts or scan_stats.get("target_domain_counts", {}),
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
        "pdfs_unavailable_in_kaggle": int(download_stats.get("unavailable", 0)),
        "kaggle_unavailable_report": download_stats.get("unavailable_report"),
        "successfully_downloaded_this_run": int(download_stats.get("downloaded", 0)),
        "already_present_pdfs": int(download_stats.get("skipped", 0)),
        "rejection_counts": scan_stats["rejection_counts"],
        "license_counts": scan_stats["license_counts"],
        "selected_domain_counts": scan_stats.get("selected_domain_counts", {}),
        "selected_primary_domain_counts": dict(sorted(count_by_primary_domain(selected_records).items())),
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
    print(f"Unavailable in Kaggle:       {report['pdfs_unavailable_in_kaggle']:,}")
    print("Rejections:")
    for reason, count in sorted(report["rejection_counts"].items()):
        print(f"  {reason}: {count:,}")


def count_by_category(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("categories", []))
    return counts


def count_by_primary_domain(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        primary_category = str(record.get("primary_category") or "")
        if primary_category.startswith("eess."):
            counts["eess."] += 1
        elif primary_category.startswith("physics."):
            counts["physics."] += 1
        else:
            counts["other"] += 1
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
