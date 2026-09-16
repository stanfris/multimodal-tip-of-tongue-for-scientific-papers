"""Build an arXiv PDF corpus with metadata-first open-reuse filtering."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data") / "arxiv_open_reuse"
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DEFAULT_OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
DEFAULT_OAI_SETS = ("physics", "eess", "cs:cs:RO", "math:math:OC")
DEFAULT_METADATA_SOURCE = "oai"
DEFAULT_TARGET_PER_DOMAIN = 10_000
DEFAULT_REQUEST_DELAY_SECONDS = 3.0
DEFAULT_MAX_RETRIES = 5
USER_AGENT = "visual-tot-arxiv-open-reuse/1.0 (mailto:stanfris2.0@gmail.com)"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/xml,text/xml,*/*",
    "Accept-Encoding": "identity",
}
HTML_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Encoding": "identity",
}
PDF_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,*/*",
    "Accept-Encoding": "identity",
}

OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/"}
ARXIV_NS = {
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

CC_BY_LICENSES = {
    "https://creativecommons.org/licenses/by/1.0",
    "https://creativecommons.org/licenses/by/2.0",
    "https://creativecommons.org/licenses/by/2.5",
    "https://creativecommons.org/licenses/by/3.0",
    "https://creativecommons.org/licenses/by/4.0",
}
CC0_LICENSES = {
    "https://creativecommons.org/publicdomain/zero/1.0",
    "https://creativecommons.org/licenses/publicdomain",
}
ALLOWED_LICENSES = CC_BY_LICENSES | CC0_LICENSES
DEFAULT_ARXIV_LICENSE = "https://arxiv.org/licenses/nonexclusive-distrib/1.0"

PHYSICS_ARCHIVE_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nucl-ex",
    "nucl-th",
    "physics",
    "quant-ph",
)
ENGINEERING_CATEGORIES = {
    "cs.RO",
    "eess.AS",
    "eess.IV",
    "eess.SP",
    "eess.SY",
    "math.OC",
}
API_ABS_CATEGORY_QUERIES = (
    ("physics", "physics.*"),
    ("physics", "astro-ph.*"),
    ("physics", "cond-mat.*"),
    ("physics", "quant-ph"),
    ("physics", "hep-th"),
    ("physics", "hep-ph"),
    ("engineering", "eess.*"),
    ("engineering", "cs.RO"),
    ("engineering", "math.OC"),
)


class OAIError(RuntimeError):
    """Raised for OAI-PMH protocol errors returned by the repository."""

    def __init__(self, code: str, message: str | None) -> None:
        super().__init__(f"OAI-PMH error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ArxivBuildResult:
    output_dir: Path
    report_path: Path
    metadata_records: int
    domain_records: int
    eligible_records: int
    downloaded: int
    failed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest arXiv metadata through OAI-PMH, keep only CC BY/CC0 Physics and Engineering papers, then optionally download PDFs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--oai-base-url", default=DEFAULT_OAI_BASE_URL)
    parser.add_argument(
        "--metadata-source",
        choices=("oai", "api-abs"),
        default=DEFAULT_METADATA_SOURCE,
        help=(
            "Use arXiv OAI-PMH, or the official Atom API plus arXiv abstract-page license link "
            "as a fallback when OAI ListRecords is unavailable."
        ),
    )
    parser.add_argument(
        "--oai-set",
        action="append",
        dest="oai_sets",
        help="OAI setSpec to harvest. Repeat to override defaults. Defaults: physics, eess, cs:cs:RO, math:math:OC.",
    )
    parser.add_argument("--metadata-only", action="store_true", help="Harvest/filter metadata and skip PDF downloads.")
    parser.add_argument(
        "--target-per-domain",
        type=int,
        default=DEFAULT_TARGET_PER_DOMAIN,
        help="Default PDF target for each broad domain.",
    )
    parser.add_argument(
        "--download-all-eligible",
        action="store_true",
        help="Download every eligible record instead of stopping once each domain reaches the target.",
    )
    parser.add_argument(
        "--stop-after-eligible-per-domain",
        type=int,
        help=(
            "Stop OAI harvesting once cached metadata contains this many CC BY/CC0 records "
            "for both broad domains. Checkpoints remain resumable for later expansion."
        ),
    )
    parser.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Delay before each network request. Keep conservative for arXiv API/PDF endpoints.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use existing JSONL/checkpoint files to avoid duplicate metadata and PDF work.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_arxiv_open_reuse_corpus(
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        oai_base_url=args.oai_base_url,
        metadata_source=args.metadata_source,
        oai_sets=tuple(args.oai_sets) if args.oai_sets else DEFAULT_OAI_SETS,
        metadata_only=args.metadata_only,
        target_per_domain=args.target_per_domain,
        download_all_eligible=args.download_all_eligible,
        stop_after_eligible_per_domain=args.stop_after_eligible_per_domain,
        overwrite_pdfs=args.overwrite_pdfs,
        request_delay_seconds=args.request_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        resume=args.resume,
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
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pdf_dir: str | Path | None = None,
    oai_base_url: str = DEFAULT_OAI_BASE_URL,
    metadata_source: str = DEFAULT_METADATA_SOURCE,
    oai_sets: tuple[str, ...] = DEFAULT_OAI_SETS,
    metadata_only: bool = False,
    target_per_domain: int = DEFAULT_TARGET_PER_DOMAIN,
    download_all_eligible: bool = False,
    stop_after_eligible_per_domain: int | None = None,
    overwrite_pdfs: bool = False,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    timeout_seconds: float = 120.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    resume: bool = True,
) -> ArxivBuildResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(pdf_dir) if pdf_dir is not None else output_path / "pdfs"

    metadata_path = output_path / "metadata_records.jsonl"
    domain_path = output_path / "domain_records.jsonl"
    eligible_path = output_path / "eligible_records.jsonl"
    checkpoint_path = output_path / "checkpoint.json"

    existing_records = load_jsonl(metadata_path) if resume else []
    seen_ids = {str(row["arxiv_id"]) for row in existing_records if "arxiv_id" in row}
    if metadata_source == "oai":
        harvested_records = harvest_oai_metadata(
            oai_base_url=oai_base_url,
            oai_sets=oai_sets,
            output_path=metadata_path,
            checkpoint_path=checkpoint_path,
            seen_ids=seen_ids,
            existing_records=existing_records,
            stop_after_eligible_per_domain=stop_after_eligible_per_domain,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            resume=resume,
        )
    elif metadata_source == "api-abs":
        harvested_records = harvest_api_abs_metadata(
            output_path=metadata_path,
            checkpoint_path=checkpoint_path,
            seen_ids=seen_ids,
            existing_records=existing_records,
            stop_after_eligible_per_domain=stop_after_eligible_per_domain,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            resume=resume,
        )
    else:
        raise ValueError(f"Unsupported metadata source: {metadata_source}")

    metadata_records = load_jsonl(metadata_path)
    domain_records = [record for record in metadata_records if record["broad_domains"]]
    eligible_records = [record for record in domain_records if record["license_family"] in {"cc-by", "cc0"}]
    write_jsonl(domain_records, domain_path)
    write_jsonl(eligible_records, eligible_path)

    selected_records = (
        eligible_records
        if download_all_eligible
        else select_records_for_domain_targets(eligible_records, target_per_domain=target_per_domain)
    )

    download_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "failures": []}
    if not metadata_only:
        download_stats = download_eligible_pdfs(
            selected_records,
            output_dir=pdf_path,
            overwrite=overwrite_pdfs,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    report = build_report(
        metadata_records=metadata_records,
        domain_records=domain_records,
        eligible_records=eligible_records,
        selected_records=selected_records,
        harvested_records=harvested_records,
        download_stats=download_stats,
        target_per_domain=target_per_domain,
        metadata_only=metadata_only,
        stop_after_eligible_per_domain=stop_after_eligible_per_domain,
        oai_sets=oai_sets,
        oai_base_url=oai_base_url,
        metadata_source=metadata_source,
    )
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print_corpus_report(report)

    return ArxivBuildResult(
        output_dir=output_path,
        report_path=report_path,
        metadata_records=len(metadata_records),
        domain_records=len(domain_records),
        eligible_records=len(eligible_records),
        downloaded=int(download_stats["downloaded"]) + int(download_stats["skipped"]),
        failed=int(download_stats["failed"]),
    )


def harvest_oai_metadata(
    *,
    oai_base_url: str,
    oai_sets: tuple[str, ...],
    output_path: Path,
    checkpoint_path: Path,
    seen_ids: set[str],
    existing_records: list[dict[str, Any]],
    stop_after_eligible_per_domain: int | None,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    resume: bool,
) -> int:
    harvested = 0
    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    eligible_counts = count_by_domain(
        record for record in existing_records if record.get("license_family") in {"cc-by", "cc0"}
    )
    for set_spec in oai_sets:
        if stop_after_eligible_per_domain is not None and all(
            eligible_counts[domain] >= stop_after_eligible_per_domain for domain in ("physics", "engineering")
        ):
            LOGGER.info("Stopping OAI harvest because both broad domains reached %d eligible records", stop_after_eligible_per_domain)
            break
        relevant_domains = domains_for_oai_set(set_spec)
        if (
            stop_after_eligible_per_domain is not None
            and relevant_domains
            and all(eligible_counts[domain] >= stop_after_eligible_per_domain for domain in relevant_domains)
        ):
            LOGGER.info("Skipping OAI set %s because relevant domain target is already satisfied", set_spec)
            continue
        token = checkpoint.get(set_spec, {}).get("resumption_token") if resume else None
        complete = checkpoint.get(set_spec, {}).get("complete") if resume else False
        if complete:
            LOGGER.info("Skipping completed OAI set %s", set_spec)
            continue
        LOGGER.info("Harvesting OAI set %s", set_spec)
        while True:
            if token:
                params = {"verb": "ListRecords", "resumptionToken": token}
            else:
                params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec}
            try:
                root = fetch_oai_xml(
                    oai_base_url,
                    params=params,
                    request_delay_seconds=request_delay_seconds,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                )
            except OAIError as exc:
                if token and exc.code == "badResumptionToken":
                    LOGGER.warning("Resumption token for %s expired; restarting set with local deduplication", set_spec)
                    token = None
                    checkpoint[set_spec] = {
                        "resumption_token": None,
                        "complete": False,
                        "updated_at": datetime.now(UTC).isoformat(),
                        "records_seen_total": len(seen_ids),
                        "restart_reason": "badResumptionToken",
                    }
                    save_checkpoint(checkpoint_path, checkpoint)
                    continue
                raise
            records = root.findall(".//oai:ListRecords/oai:record", OAI_NS)
            for record_element in records:
                parsed = parse_oai_record(record_element, source_set=set_spec)
                if parsed is None or parsed["arxiv_id"] in seen_ids:
                    continue
                seen_ids.add(parsed["arxiv_id"])
                append_jsonl(output_path, parsed)
                harvested += 1
                if parsed["license_family"] in {"cc-by", "cc0"}:
                    eligible_counts.update(parsed["broad_domains"])
            token = parse_resumption_token(root)
            checkpoint[set_spec] = {
                "resumption_token": token,
                "complete": token is None,
                "updated_at": datetime.now(UTC).isoformat(),
                "records_seen_total": len(seen_ids),
            }
            save_checkpoint(checkpoint_path, checkpoint)
            LOGGER.info("Set %s: harvested page with %d records; next token=%s", set_spec, len(records), bool(token))
            if (
                stop_after_eligible_per_domain is not None
                and relevant_domains
                and all(eligible_counts[domain] >= stop_after_eligible_per_domain for domain in relevant_domains)
            ):
                LOGGER.info("Pausing OAI set %s after reaching targeted eligible count; resume later to expand", set_spec)
                break
            if token is None:
                break
    return harvested


def harvest_api_abs_metadata(
    *,
    output_path: Path,
    checkpoint_path: Path,
    seen_ids: set[str],
    existing_records: list[dict[str, Any]],
    stop_after_eligible_per_domain: int | None,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
    resume: bool,
    page_size: int = 25,
) -> int:
    harvested = 0
    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    api_checkpoint = checkpoint.setdefault("api_abs", {})
    eligible_counts = count_by_domain(
        record for record in existing_records if record.get("license_family") in {"cc-by", "cc0"}
    )
    for broad_domain, category_query in API_ABS_CATEGORY_QUERIES:
        if stop_after_eligible_per_domain is not None and all(
            eligible_counts[domain] >= stop_after_eligible_per_domain for domain in ("physics", "engineering")
        ):
            LOGGER.info("Stopping API harvest because both broad domains reached %d eligible records", stop_after_eligible_per_domain)
            break
        if stop_after_eligible_per_domain is not None and eligible_counts[broad_domain] >= stop_after_eligible_per_domain:
            LOGGER.info("Skipping API category %s because %s already reached the eligible target", category_query, broad_domain)
            continue

        category_state = api_checkpoint.setdefault(category_query, {"start": 0, "complete": False})
        if resume and category_state.get("complete"):
            continue
        start = int(category_state.get("start", 0))
        while True:
            feed_root = fetch_atom_feed(
                category_query=category_query,
                start=start,
                max_results=page_size,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            entries = feed_root.findall("atom:entry", ATOM_NS)
            if not entries:
                category_state.update({"complete": True, "updated_at": datetime.now(UTC).isoformat()})
                save_checkpoint(checkpoint_path, checkpoint)
                break

            for entry in entries:
                candidate = parse_atom_entry(entry, source_query=category_query)
                if candidate["arxiv_id"] in seen_ids:
                    continue
                license_error = None
                try:
                    license_url = fetch_abs_license_url(
                        candidate["arxiv_id"],
                        request_delay_seconds=request_delay_seconds,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                    )
                except Exception as exc:  # noqa: BLE001 - save the candidate and continue harvesting.
                    license_url = None
                    license_error = str(exc)
                    LOGGER.warning("Could not verify license for %s: %s", candidate["arxiv_id"], license_error)
                normalized_license = normalize_license_url(license_url)
                license_family = classify_license(normalized_license)
                candidate.update(
                    {
                        "license_url": license_url,
                        "normalized_license_url": normalized_license,
                        "license_family": license_family,
                        "license_error": license_error,
                        "metadata_source": "arxiv-api-plus-abs-license",
                    }
                )
                seen_ids.add(candidate["arxiv_id"])
                append_jsonl(output_path, candidate)
                harvested += 1
                if license_family in {"cc-by", "cc0"}:
                    eligible_counts.update(candidate["broad_domains"])
                    LOGGER.info(
                        "Eligible %s: %s %s",
                        ",".join(candidate["broad_domains"]),
                        candidate["arxiv_id"],
                        normalized_license,
                    )
                if (
                    stop_after_eligible_per_domain is not None
                    and eligible_counts[broad_domain] >= stop_after_eligible_per_domain
                ):
                    break

            start += len(entries)
            category_state.update({"start": start, "complete": False, "updated_at": datetime.now(UTC).isoformat()})
            save_checkpoint(checkpoint_path, checkpoint)
            if stop_after_eligible_per_domain is not None and eligible_counts[broad_domain] >= stop_after_eligible_per_domain:
                LOGGER.info("Pausing API category %s after reaching targeted eligible count", category_query)
                break
    return harvested


def parse_oai_record(record: ElementTree.Element, *, source_set: str) -> dict[str, Any] | None:
    header = record.find("oai:header", OAI_NS)
    if header is not None and header.get("status") == "deleted":
        return None
    arxiv = record.find("oai:metadata/arxiv:arXiv", {**OAI_NS, **ARXIV_NS})
    if arxiv is None:
        return None

    arxiv_id = text_at(arxiv, "arxiv:id")
    if not arxiv_id:
        return None
    categories = split_categories(text_at(arxiv, "arxiv:categories"))
    license_url = text_at(arxiv, "arxiv:license")
    normalized_license = normalize_license_url(license_url)
    license_family = classify_license(normalized_license)
    broad_domains = classify_broad_domains(categories)

    return {
        "arxiv_id": arxiv_id,
        "title": normalize_text(text_at(arxiv, "arxiv:title")),
        "authors": parse_authors(arxiv),
        "abstract": normalize_text(text_at(arxiv, "arxiv:abstract")),
        "created": text_at(arxiv, "arxiv:created"),
        "updated": text_at(arxiv, "arxiv:updated"),
        "oai_datestamp": text_at(record, "oai:header/oai:datestamp", OAI_NS),
        "source_oai_set": source_set,
        "subject_categories": categories,
        "primary_category": categories[0] if categories else None,
        "broad_domains": broad_domains,
        "license_url": license_url,
        "normalized_license_url": normalized_license,
        "license_family": license_family,
        "pdf_url": pdf_url_for_arxiv_id(arxiv_id),
        "pdf_filename": pdf_filename_for_arxiv_id(arxiv_id),
    }


def parse_authors(arxiv: ElementTree.Element) -> list[dict[str, str | None]]:
    authors = []
    for author in arxiv.findall("arxiv:authors/arxiv:author", ARXIV_NS):
        keyname = normalize_text(text_at(author, "arxiv:keyname", ARXIV_NS))
        forenames = normalize_text(text_at(author, "arxiv:forenames", ARXIV_NS))
        suffix = normalize_text(text_at(author, "arxiv:suffix", ARXIV_NS))
        full_name = " ".join(part for part in (forenames, keyname, suffix) if part)
        authors.append(
            {
                "keyname": keyname,
                "forenames": forenames,
                "suffix": suffix,
                "name": full_name or keyname,
            }
        )
    return authors


def fetch_atom_feed(
    *,
    category_query: str,
    start: int,
    max_results: int,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> ElementTree.Element:
    query = f"cat:{category_query}"
    url = url_with_params(
        "https://export.arxiv.org/api/query",
        {
            "search_query": query,
            "start": str(start),
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    payload = fetch_bytes(
        url,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        headers=REQUEST_HEADERS,
    )
    return ElementTree.fromstring(payload)


def parse_atom_entry(entry: ElementTree.Element, *, source_query: str) -> dict[str, Any]:
    abs_url = text_at(entry, "atom:id", ATOM_NS)
    if not abs_url:
        raise ValueError("Atom entry is missing id")
    arxiv_id = abs_url.rstrip("/").rsplit("/", 1)[-1]
    categories = [
        category.get("term", "")
        for category in entry.findall("atom:category", ATOM_NS)
        if category.get("term")
    ]
    primary = entry.find("arxiv:primary_category", ATOM_NS)
    primary_category = primary.get("term") if primary is not None else (categories[0] if categories else None)
    if primary_category and primary_category in categories:
        categories = [primary_category] + [category for category in categories if category != primary_category]
    authors = [
        {"name": normalize_text(text_at(author, "atom:name", ATOM_NS)), "forenames": None, "keyname": None, "suffix": None}
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break
    return {
        "arxiv_id": arxiv_id,
        "title": normalize_text(text_at(entry, "atom:title", ATOM_NS)),
        "authors": authors,
        "abstract": normalize_text(text_at(entry, "atom:summary", ATOM_NS)),
        "created": normalize_atom_datetime(text_at(entry, "atom:published", ATOM_NS)),
        "updated": normalize_atom_datetime(text_at(entry, "atom:updated", ATOM_NS)),
        "oai_datestamp": None,
        "source_oai_set": None,
        "source_api_query": source_query,
        "subject_categories": categories,
        "primary_category": primary_category,
        "broad_domains": classify_broad_domains(categories),
        "pdf_url": pdf_url or pdf_url_for_arxiv_id(arxiv_id),
        "pdf_filename": pdf_filename_for_arxiv_id(arxiv_id),
    }


def fetch_abs_license_url(
    arxiv_id: str,
    *,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> str | None:
    url = f"https://arxiv.org/abs/{urllib.parse.quote(arxiv_abs_page_id(arxiv_id), safe='')}"
    html = fetch_bytes(
        url,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        headers=HTML_REQUEST_HEADERS,
    ).decode("utf-8", "replace")
    parser = ArxivAbsLicenseParser()
    parser.feed(html)
    return parser.license_url


def arxiv_abs_page_id(arxiv_id: str) -> str:
    if "/" in arxiv_id:
        return arxiv_id
    head, marker, version = arxiv_id.rpartition("v")
    if marker and head and version.isdigit():
        return head
    return arxiv_id


class ArxivAbsLicenseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_license_div = False
        self.license_url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "div" and "abs-license" in (attributes.get("class") or ""):
            self._in_license_div = True
        if self._in_license_div and tag == "a" and attributes.get("href"):
            self.license_url = attributes["href"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_license_div:
            self._in_license_div = False


def normalize_license_url(url: str | None) -> str | None:
    value = normalize_text(url)
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    scheme = "https"
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    normalized = urllib.parse.urlunsplit((scheme, hostname, path, "", ""))
    if normalized == "https://creativecommons.org/licenses/publicdomain":
        return normalized
    return normalized


def classify_license(normalized_license_url: str | None) -> str:
    if normalized_license_url in CC_BY_LICENSES:
        return "cc-by"
    if normalized_license_url in CC0_LICENSES:
        return "cc0"
    if normalized_license_url == DEFAULT_ARXIV_LICENSE:
        return "arxiv-default"
    if normalized_license_url and "creativecommons.org" in normalized_license_url:
        return "other-creative-commons"
    return "missing-or-other"


def classify_broad_domains(categories: Iterable[str]) -> list[str]:
    categories = list(categories)
    domains: list[str] = []
    if any(is_physics_category(category) for category in categories):
        domains.append("physics")
    if any(category in ENGINEERING_CATEGORIES for category in categories):
        domains.append("engineering")
    return domains


def is_physics_category(category: str) -> bool:
    return any(category == prefix or category.startswith(f"{prefix}.") for prefix in PHYSICS_ARCHIVE_PREFIXES)


def domains_for_oai_set(set_spec: str) -> set[str]:
    if set_spec == "physics" or set_spec.startswith("physics:"):
        return {"physics"}
    if set_spec == "eess" or set_spec.startswith("eess:") or set_spec in {"cs:cs:RO", "math:math:OC"}:
        return {"engineering"}
    return {"physics", "engineering"}


def select_records_for_domain_targets(records: list[dict[str, Any]], *, target_per_domain: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    counts = Counter()
    for record in sorted(records, key=lambda item: (item.get("created") or "", item["arxiv_id"])):
        if all(counts[domain] >= target_per_domain for domain in ("physics", "engineering")):
            break
        if not any(counts[domain] < target_per_domain for domain in record["broad_domains"]):
            continue
        if record["arxiv_id"] in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record["arxiv_id"])
        for domain in record["broad_domains"]:
            counts[domain] += 1
    return selected


def download_eligible_pdfs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    overwrite: bool,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "download_failures.jsonl"
    latest_failures_path = output_dir / "download_failures.latest.jsonl"
    manifest_path = output_dir / "download_manifest.jsonl"
    latest_failures_path.write_text("", encoding="utf-8")
    downloaded = skipped = failed = 0
    failures = []
    total = len(records)
    started_at = time.monotonic()
    for index, record in enumerate(records, start=1):
        destination = output_dir / record["pdf_filename"]
        if destination.exists() and not overwrite and has_pdf_header(destination):
            skipped += 1
            status = "skipped"
            error = None
        else:
            try:
                pdf_url = str(record["pdf_url"])
                try:
                    download_pdf(
                        pdf_url,
                        destination,
                        request_delay_seconds=request_delay_seconds,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                    )
                except Exception:
                    fallback_url = versionless_pdf_url(record["arxiv_id"])
                    if fallback_url == pdf_url:
                        raise
                    LOGGER.info("Retrying %s with versionless PDF URL %s", record["arxiv_id"], fallback_url)
                    download_pdf(
                        fallback_url,
                        destination,
                        request_delay_seconds=request_delay_seconds,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                    )
                downloaded += 1
                status = "downloaded"
                error = None
            except Exception as exc:  # noqa: BLE001 - corpus builds should continue.
                failed += 1
                status = "failed"
                error = str(exc)
                failure = {"arxiv_id": record["arxiv_id"], "pdf_url": record["pdf_url"], "error": error}
                failures.append(failure)
                append_jsonl(failures_path, failure)
                append_jsonl(latest_failures_path, failure)
                LOGGER.warning("Failed to download %s: %s", record["arxiv_id"], error)
        append_jsonl(
            manifest_path,
            {
                "arxiv_id": record["arxiv_id"],
                "pdf_url": record["pdf_url"],
                "pdf_path": str(destination),
                "license_url": record["license_url"],
                "normalized_license_url": record["normalized_license_url"],
                "license_family": record["license_family"],
                "status": status,
                "error": error,
            },
        )
        print_download_progress(index, total, downloaded, skipped, failed, started_at)
    if total:
        print()
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "failures": failures}


def fetch_oai_xml(
    base_url: str,
    *,
    params: dict[str, str],
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> ElementTree.Element:
    payload = fetch_bytes(
        url_with_params(base_url, params),
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    root = ElementTree.fromstring(payload)
    error = root.find("oai:error", OAI_NS)
    if error is not None:
        code = error.get("code", "unknown")
        raise OAIError(code, normalize_text("".join(error.itertext())))
    return root


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
            request = urllib.request.Request(url, headers=headers or REQUEST_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {400, 401, 403, 404, 406, 410}:
                break
            if attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = min(300.0, request_delay_seconds + 2**attempt)
                LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, wait_seconds)
                time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def download_pdf(
    url: str,
    destination: Path,
    *,
    request_delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> None:
    part_path = destination.with_suffix(destination.suffix + ".part")
    part_path.write_bytes(
        fetch_bytes(
            url,
            request_delay_seconds=request_delay_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            headers=PDF_REQUEST_HEADERS,
        )
    )
    if not has_pdf_header(part_path):
        part_path.unlink(missing_ok=True)
        raise ValueError("downloaded file is not a PDF")
    part_path.replace(destination)


def versionless_pdf_url(arxiv_id: str) -> str:
    return pdf_url_for_arxiv_id(arxiv_abs_page_id(arxiv_id))


def print_download_progress(
    completed: int,
    total: int,
    downloaded: int,
    skipped: int,
    failed: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else 0.0
    percent = (completed / total * 100.0) if total else 100.0
    print(
        f"\r{completed}/{total} ({percent:5.1f}%) "
        f"downloaded={downloaded} skipped={skipped} failed={failed} "
        f"rate={rate:0.2f}/s eta={format_duration(eta_seconds)}",
        end="",
        flush=True,
    )


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
    metadata_records: list[dict[str, Any]],
    domain_records: list[dict[str, Any]],
    eligible_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    harvested_records: int,
    download_stats: dict[str, Any],
    target_per_domain: int,
    metadata_only: bool,
    stop_after_eligible_per_domain: int | None,
    oai_sets: tuple[str, ...],
    oai_base_url: str,
    metadata_source: str,
) -> dict[str, Any]:
    eligible_by_domain = count_by_domain(eligible_records)
    selected_by_domain = count_by_domain(selected_records)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": "arXiv OAI-PMH",
            "metadata_source": metadata_source,
            "oai_base_url": oai_base_url,
            "metadata_prefix": "arXiv",
            "oai_sets": list(oai_sets),
            "api_abs_fallback": {
                "api_base_url": "https://export.arxiv.org/api/query",
                "license_verification": "https://arxiv.org/abs/{arxiv_id} abs-license link",
            },
            "pdf_download_base": "https://arxiv.org/pdf/{arxiv_id}",
        },
        "license_policy": {
            "allowed_license_families": ["cc-by", "cc0"],
            "allowed_normalized_license_urls": sorted(ALLOWED_LICENSES),
            "excluded": [
                DEFAULT_ARXIV_LICENSE,
                "CC BY-NC",
                "CC BY-ND",
                "CC BY-NC-ND",
                "CC BY-SA",
                "missing license records",
            ],
        },
        "target_per_domain": target_per_domain,
        "stop_after_eligible_per_domain": stop_after_eligible_per_domain,
        "harvest_is_partial": stop_after_eligible_per_domain is not None,
        "metadata_records_discovered": len(metadata_records),
        "new_metadata_records_harvested_this_run": harvested_records,
        "domain_matching_records": len(domain_records),
        "cc_by_records": sum(1 for record in domain_records if record["license_family"] == "cc-by"),
        "cc0_records": sum(1 for record in domain_records if record["license_family"] == "cc0"),
        "total_eligible_records": len(eligible_records),
        "selected_for_download": len(selected_records),
        "metadata_only": metadata_only,
        "downloaded_pdfs": int(download_stats.get("downloaded", 0)) + int(download_stats.get("skipped", 0)),
        "successfully_downloaded_this_run": int(download_stats.get("downloaded", 0)),
        "already_present_pdfs": int(download_stats.get("skipped", 0)),
        "failed_downloads": int(download_stats.get("failed", 0)),
        "feasibility": {
            "physics_target_feasible": eligible_by_domain["physics"] >= target_per_domain,
            "engineering_target_feasible": eligible_by_domain["engineering"] >= target_per_domain,
            "combined_20000_target_feasible": (
                eligible_by_domain["physics"] >= target_per_domain
                and eligible_by_domain["engineering"] >= target_per_domain
            ),
            "eligible_by_domain": dict(eligible_by_domain),
            "selected_by_domain": dict(selected_by_domain),
        },
        "counts": {
            "by_broad_domain": dict(eligible_by_domain),
            "by_arxiv_category": dict(sorted(count_by_category(eligible_records).items())),
            "by_year": dict(sorted(count_by_year(eligible_records).items())),
            "by_license": dict(sorted(Counter(record["license_family"] for record in eligible_records).items())),
            "by_normalized_license_url": dict(
                sorted(Counter(record["normalized_license_url"] for record in eligible_records).items())
            ),
        },
    }


def print_corpus_report(report: dict[str, Any]) -> None:
    print("arXiv open-reuse corpus report")
    print(f"metadata records discovered: {report['metadata_records_discovered']}")
    print(f"domain-matching records: {report['domain_matching_records']}")
    print(f"CC BY records: {report['cc_by_records']}")
    print(f"CC0 records: {report['cc0_records']}")
    print(f"total eligible records: {report['total_eligible_records']}")
    print(f"successfully downloaded PDFs: {report['downloaded_pdfs']}")
    print(f"failed downloads: {report['failed_downloads']}")
    print("eligible by broad domain:")
    for domain, count in sorted(report["feasibility"]["eligible_by_domain"].items()):
        print(f"  {domain}: {count}")
    feasible = report["feasibility"]["combined_20000_target_feasible"]
    target = report["target_per_domain"] * 2
    print(f"combined {target:,} PDF target feasible from discovered metadata: {feasible}")


def count_by_domain(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for domain in record["broad_domains"]:
            counts[domain] += 1
    return counts


def count_by_category(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["subject_categories"])
    return counts


def count_by_year(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        year = year_from_record(record)
        counts[str(year) if year is not None else "unknown"] += 1
    return counts


def year_from_record(record: dict[str, Any]) -> int | None:
    for key in ("created", "updated", "oai_datestamp"):
        value = record.get(key)
        if value:
            try:
                return int(str(value)[:4])
            except ValueError:
                continue
    return None


def split_categories(value: str | None) -> list[str]:
    return [category.strip() for category in (value or "").split() if category.strip()]


def pdf_url_for_arxiv_id(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{urllib.parse.quote(arxiv_id, safe='/')}"


def pdf_filename_for_arxiv_id(arxiv_id: str) -> str:
    return f"{arxiv_id.replace('/', '_')}.pdf"


def parse_resumption_token(root: ElementTree.Element) -> str | None:
    token = root.find(".//oai:resumptionToken", OAI_NS)
    if token is None:
        return None
    value = normalize_text("".join(token.itertext()))
    return value or None


def text_at(element: ElementTree.Element, path: str, namespaces: dict[str, str] | None = None) -> str | None:
    child = element.find(path, namespaces or {**OAI_NS, **ARXIV_NS})
    if child is None:
        return None
    return "".join(child.itertext())


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split())


def normalize_atom_datetime(value: str | None) -> str | None:
    value = normalize_text(value)
    if value and "T" in value:
        return value[:10]
    return value


def has_pdf_header(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def url_with_params(base_url: str, params: dict[str, str]) -> str:
    params = {
        key: urllib.parse.unquote(value) if key == "resumptionToken" else value
        for key, value in params.items()
    }
    return f"{base_url}?{urllib.parse.urlencode(params)}"


def existing_ids(path: Path) -> set[str]:
    return {str(row["arxiv_id"]) for row in load_jsonl(path) if "arxiv_id" in row}


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


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
