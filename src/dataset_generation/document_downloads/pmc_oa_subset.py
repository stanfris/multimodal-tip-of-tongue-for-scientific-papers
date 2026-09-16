"""Build strict CC BY/CC0 Biology and Medical/Clinical PMC OA PDF datasets."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import httpx

from dataset_generation.jsonl import append_jsonl_object, read_jsonl_objects


LOGGER = logging.getLogger(__name__)

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_AWS_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com"
DEFAULT_OUTPUT_DIR = Path("data") / "pmc_oa_strict"
DEFAULT_PDF_DIR = DEFAULT_OUTPUT_DIR / "pdfs"
DISCOVERED_PMCIDS_FILENAME = "discovered_pmcids.jsonl"
DISCOVERY_RANGES_FILENAME = "discovery_ranges.jsonl"
AWS_METADATA_FILENAME = "aws_metadata.jsonl"
PUBMED_METADATA_FILENAME = "pubmed_metadata.jsonl"
DEFAULT_TARGET_PER_DOMAIN = 20_000
DEFAULT_MAX_WORKERS = 32
DEFAULT_AWS_METADATA_WORKERS = 64
DEFAULT_START_YEAR = 2026
DEFAULT_END_YEAR = 2026
DEFAULT_DISCOVERY_QUERY = (
    '("cc by license"[filter] OR "cc0 license"[filter]) '
    "AND open_access[filter] AND has_pdf[filter] NOT pmc embargo[filter]"
)
ELIGIBLE_LICENSES = {"cc by", "cc0"}
PREFERRED_ARTICLE_TYPES = {
    "journal article",
    "research article",
    "review",
    "systematic review",
    "meta-analysis",
    "clinical trial",
    "clinical trial, phase i",
    "clinical trial, phase ii",
    "clinical trial, phase iii",
    "clinical trial, phase iv",
    "randomized controlled trial",
    "comparative study",
    "observational study",
    "multicenter study",
    "validation study",
}
EXCLUDED_ARTICLE_TYPE_KEYWORDS = {
    "address",
    "biography",
    "comment",
    "congress",
    "correction",
    "editorial",
    "erratum",
    "expression of concern",
    "interview",
    "letter",
    "news",
    "newspaper article",
    "published erratum",
    "retracted publication",
    "retraction",
}

MEDICAL_TERMS = {
    "clinical",
    "medicine",
    "medical",
    "patient",
    "patients",
    "humans",
    "human",
    "epidemiology",
    "public health",
    "neoplasms",
    "cancer",
    "oncology",
    "cardiology",
    "cardiovascular",
    "neurology",
    "infectious",
    "infection",
    "surgery",
    "surgical",
    "diagnosis",
    "diagnostic",
    "therapy",
    "therapeutics",
    "pharmacology",
    "drug therapy",
    "vaccine",
    "vaccination",
    "disease",
    "diseases",
    "mortality",
    "morbidity",
    "risk factors",
}
BIOLOGY_TERMS = {
    "biology",
    "molecular",
    "cell",
    "cells",
    "genetics",
    "genomics",
    "genome",
    "gene",
    "genes",
    "protein",
    "proteins",
    "biochemistry",
    "microbiology",
    "microbial",
    "bacteria",
    "virus",
    "viruses",
    "ecology",
    "evolution",
    "developmental",
    "development",
    "neuroscience",
    "neuron",
    "neurons",
    "animal",
    "animals",
    "plant",
    "plants",
    "physiology",
}


@dataclass(frozen=True)
class PmcBuildResult:
    records_path: Path
    report_path: Path
    record_count: int
    pdf_stats: dict[str, Any] | None = None


def normalize_license(value: str | None) -> str | None:
    """Return the exact accepted license code or None for every other license."""
    if value is None:
        return None
    normalized = " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        "ccby": "cc by",
        "cc by": "cc by",
        "creativecommons attribution": "cc by",
        "creative commons attribution": "cc by",
        "cc0": "cc0",
        "cc 0": "cc0",
        "public domain dedication": "cc0",
    }
    return aliases.get(normalized)


def is_eligible_license(value: str | None) -> bool:
    return normalize_license(value) in ELIGIBLE_LICENSES


def build_pmc_oa_subset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    query: str = DEFAULT_DISCOVERY_QUERY,
    target_per_domain: int = DEFAULT_TARGET_PER_DOMAIN,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int | None = DEFAULT_END_YEAR,
    email: str | None = None,
    api_key: str | None = None,
    tool: str = "dataset-generation-pmc-oa",
    metadata_only: bool = False,
    download_pdfs: bool = False,
    pdf_dir: str | Path | None = None,
    overwrite_pdfs: bool = False,
    include_non_english: bool = False,
    include_non_research: bool = False,
    max_records: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    aws_metadata_workers: int = DEFAULT_AWS_METADATA_WORKERS,
    request_sleep_seconds: float = 0.11,
    retries: int = 3,
) -> PmcBuildResult:
    """Discover, license-gate, classify, and optionally download strict PMC OA PDFs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "papers.jsonl"
    report_path = output_path / "report.json"

    config = {
        "source": "PMC Open Access Subset through NCBI E-Utilities and PMC Article Datasets on AWS",
        "discovery_query": query,
        "license_policy": "Require exact article-version license_code cc by or cc0 before PDF download.",
        "target_per_domain": target_per_domain,
        "start_year": start_year,
        "end_year": end_year or dt.date.today().year,
        "include_non_english": include_non_english,
        "include_non_research": include_non_research,
        "aws_metadata_workers": aws_metadata_workers,
        "official_sources": {
            "eutils": NCBI_EUTILS_BASE_URL,
            "pmc_aws": PMC_AWS_BASE_URL,
            "pmc_aws_readme": f"{PMC_AWS_BASE_URL}/README.txt",
        },
    }
    (output_path / "build_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=20.0),
        headers={"User-Agent": user_agent(tool, email)},
        follow_redirects=True,
    )
    try:
        print_phase("Discovering PMCIDs with strict CC BY/CC0 + OA + PDF filters")
        discovered_pmcids_path = output_path / DISCOVERED_PMCIDS_FILENAME
        discovery_ranges_path = output_path / DISCOVERY_RANGES_FILENAME
        pmcids = discover_pmcids(
            client,
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_records=max_records,
            discovered_pmcids_path=discovered_pmcids_path,
            discovery_ranges_path=discovery_ranges_path,
            email=email,
            api_key=api_key,
            tool=tool,
            request_sleep_seconds=request_sleep_seconds,
            retries=retries,
        )
        print(f"Discovered {len(pmcids):,} unique PMCIDs before AWS article-version license verification.", flush=True)

        written_pmcids = {normalize_pmcid_number(row["pmcid"]) for row in read_jsonl_objects(records_path, missing_ok=True)}
        records: list[dict[str, Any]] = read_jsonl_objects(records_path, missing_ok=True)
        existing_counts = Counter(row["broad_domain"] for row in records)
        targets_already_met = (
            existing_counts["Biology"] >= target_per_domain
            and existing_counts["Medical/Clinical Research"] >= target_per_domain
        )
        aws_metadata_path = output_path / AWS_METADATA_FILENAME
        cached_metadata = read_aws_metadata_cache(aws_metadata_path) if not targets_already_met else {}
        pending = [] if targets_already_met else [
            pmcid for pmcid in pmcids if pmcid not in written_pmcids and pmcid not in cached_metadata
        ]
        pmid_to_pmcid: dict[str, str] = {}
        metadata_by_pmcid: dict[str, dict[str, Any]] = {
            pmcid: row["metadata"]
            for pmcid, row in cached_metadata.items()
            if row.get("status") == "eligible" and pmcid not in written_pmcids
        }
        for pmcid, version_meta in metadata_by_pmcid.items():
            pmid = clean_id(version_meta.get("pmid"))
            if pmid:
                pmid_to_pmcid[pmid] = pmcid

        print_phase("Checking PMC AWS JSON metadata before any PDF download")
        if targets_already_met:
            print(
                "Target counts are already met in papers.jsonl; skipping additional metadata checks and classification.",
                flush=True,
            )
        if written_pmcids:
            print(f"Resuming with {len(written_pmcids):,} PMCIDs already present in {records_path}.", flush=True)
        if cached_metadata:
            print(
                f"Loaded {len(cached_metadata):,} cached AWS metadata decisions; "
                f"{len(metadata_by_pmcid):,} cached eligible records still need classification.",
                flush=True,
            )
        for completed, pmcid, version_meta in iter_eligible_article_versions(
            pending,
            max_workers=aws_metadata_workers,
            retries=retries,
        ):
            if isinstance(version_meta, dict) and version_meta.get("_status") == "failed":
                append_jsonl_object(
                    aws_metadata_path,
                    {
                        "pmcid": pmcid,
                        "status": "failed",
                        "error": version_meta.get("_error"),
                        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
                    },
                )
                if completed == 1 or completed % 100 == 0 or completed == len(pending):
                    print_inline_progress("AWS metadata", completed, len(pending), eligible=len(metadata_by_pmcid))
                continue
            if version_meta is None:
                append_jsonl_object(
                    aws_metadata_path,
                    {
                        "pmcid": pmcid,
                        "status": "ineligible",
                        "checked_at": dt.datetime.now(dt.UTC).isoformat(),
                    },
                )
                if completed == 1 or completed % 100 == 0 or completed == len(pending):
                    print_inline_progress("AWS metadata", completed, len(pending), eligible=len(metadata_by_pmcid))
                continue
            append_jsonl_object(
                aws_metadata_path,
                {
                    "pmcid": pmcid,
                    "status": "eligible",
                    "metadata": version_meta,
                    "checked_at": dt.datetime.now(dt.UTC).isoformat(),
                },
            )
            pmid = clean_id(version_meta.get("pmid"))
            if pmid:
                pmid_to_pmcid[pmid] = pmcid
            metadata_by_pmcid[pmcid] = version_meta
            if completed == 1 or completed % 100 == 0 or completed == len(pending):
                print_inline_progress("AWS metadata", completed, len(pending), eligible=len(metadata_by_pmcid))
        if pending:
            print()
        print(f"License/PDF-gated article versions ready for PubMed enrichment: {len(metadata_by_pmcid):,}", flush=True)

        print_phase("Fetching PubMed subject metadata")
        pubmed_metadata_path = output_path / PUBMED_METADATA_FILENAME
        cached_pubmed_meta = read_pubmed_metadata_cache(pubmed_metadata_path)
        pmids_for_metadata = sorted(pmid_to_pmcid)
        pmids_to_fetch = [pmid for pmid in pmids_for_metadata if pmid not in cached_pubmed_meta]
        if cached_pubmed_meta:
            print(f"Loaded {len(cached_pubmed_meta):,} cached PubMed metadata records.", flush=True)
        fetched_pubmed_meta = {} if targets_already_met else fetch_pubmed_metadata(
            client,
            pmids_to_fetch,
            email=email,
            api_key=api_key,
            tool=tool,
            request_sleep_seconds=request_sleep_seconds,
            retries=retries,
            cache_path=pubmed_metadata_path,
        )
        pubmed_meta = {**cached_pubmed_meta, **fetched_pubmed_meta}
        print(f"Fetched PubMed metadata for {len(pubmed_meta):,} PMID-linked records.", flush=True)

        print_phase("Classifying records into Biology and Medical/Clinical Research")
        accepted_before = len(records)
        for pmcid, version_meta in ([] if targets_already_met else metadata_by_pmcid.items()):
            pmid = clean_id(version_meta.get("pmid"))
            record = build_record(version_meta, pubmed_meta.get(pmid, {}))
            if not include_non_english and record["language"] and record["language"].lower() != "eng":
                continue
            if not include_non_research and should_exclude_article_type(record["article_type"]):
                continue
            if record["broad_domain"] not in {"Biology", "Medical/Clinical Research"}:
                continue
            append_jsonl_object(records_path, record)
            records.append(record)
            counts = Counter(row["broad_domain"] for row in records)
            accepted_now = len(records) - accepted_before
            if accepted_now == 1 or accepted_now % 100 == 0:
                print(
                    "\rAccepted records: "
                    f"{accepted_now:,} new; Biology={counts['Biology']:,}; "
                    f"Medical/Clinical={counts['Medical/Clinical Research']:,}",
                    end="",
                    flush=True,
                )
            if counts["Biology"] >= target_per_domain and counts["Medical/Clinical Research"] >= target_per_domain:
                break
        if len(records) > accepted_before:
            print()
    finally:
        client.close()

    print_phase("Writing manifest and report")
    resolved_pdf_dir = Path(pdf_dir) if pdf_dir is not None else output_path / "pdfs"
    assign_pdf_paths(records, resolved_pdf_dir)
    write_jsonl(records, records_path)

    report = build_report(records, total_discovered=len(pmcids), target_per_domain=target_per_domain)
    report["resume_state"] = {
        "discovered_pmcids_path": str(discovered_pmcids_path),
        "discovered_pmcids_rows": count_jsonl_rows(discovered_pmcids_path),
        "discovery_ranges_path": str(discovery_ranges_path),
        "discovery_ranges_rows": count_jsonl_rows(discovery_ranges_path),
        "aws_metadata_path": str(aws_metadata_path),
        "aws_metadata_rows": count_jsonl_rows(aws_metadata_path),
        "pubmed_metadata_path": str(pubmed_metadata_path),
        "pubmed_metadata_rows": count_jsonl_rows(pubmed_metadata_path),
    }
    pdf_stats = None
    if download_pdfs and not metadata_only:
        print_phase("Downloading PDFs after strict license gating")
        pdf_stats = download_pmc_pdfs(
            records,
            output_dir=resolved_pdf_dir,
            overwrite=overwrite_pdfs,
            max_workers=max_workers,
            retries=retries,
        )
        report["pdf_downloads"] = pdf_stats
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(output_path / "report.md", report)
    print(f"Wrote {len(records):,} eligible records to {records_path}", flush=True)
    print(f"Wrote report to {report_path}", flush=True)

    return PmcBuildResult(records_path=records_path, report_path=report_path, record_count=len(records), pdf_stats=pdf_stats)


def user_agent(tool: str, email: str | None) -> str:
    suffix = f" ({email})" if email else ""
    return f"{tool}/1.0{suffix}"


def print_phase(message: str) -> None:
    print(f"\n== {message} ==", flush=True)


def print_inline_progress(label: str, completed: int, total: int, **extra: int) -> None:
    percent = (completed / total * 100) if total else 100.0
    extra_text = " ".join(f"{key}={value:,}" for key, value in extra.items())
    suffix = f" {extra_text}" if extra_text else ""
    print(f"\r{label}: {completed:,}/{total:,} ({percent:5.1f}%){suffix}", end="", flush=True)


def discover_pmcids(
    client: httpx.Client,
    *,
    query: str,
    start_year: int,
    end_year: int | None,
    max_records: int | None,
    discovered_pmcids_path: Path,
    discovery_ranges_path: Path,
    email: str | None,
    api_key: str | None,
    tool: str,
    request_sleep_seconds: float,
    retries: int,
) -> list[str]:
    year_end = end_year or dt.date.today().year
    discovered_rows = read_jsonl_objects(discovered_pmcids_path, missing_ok=True)
    pmcids: list[str] = []
    seen: set[str] = set()
    for row in discovered_rows:
        pmcid = normalize_pmcid_number(row.get("pmcid"))
        if pmcid not in seen:
            seen.add(pmcid)
            pmcids.append(pmcid)
    completed_ranges = {
        str(row.get("range_key"))
        for row in read_jsonl_objects(discovery_ranges_path, missing_ok=True)
        if row.get("status") == "fetched"
    }
    print(
        f"Scanning publication years {start_year}-{year_end}. "
        f"Loaded {len(pmcids):,} discovered PMCIDs and {len(completed_ranges):,} completed date ranges.",
        flush=True,
    )
    for year in range(start_year, year_end + 1):
        before = len(pmcids)
        print(f"  {year}: querying date ranges...", flush=True)
        collect_pmcids_for_date_range(
            client,
            query=query,
            start_date=dt.date(year, 1, 1),
            end_date=dt.date(year, 12, 31),
            pmcids=pmcids,
            seen=seen,
            completed_ranges=completed_ranges,
            discovered_pmcids_path=discovered_pmcids_path,
            discovery_ranges_path=discovery_ranges_path,
            max_records=max_records,
            email=email,
            api_key=api_key,
            tool=tool,
            request_sleep_seconds=request_sleep_seconds,
            retries=retries,
        )
        added = len(pmcids) - before
        print(f"  {year}: added {added:,} PMCIDs; cumulative {len(pmcids):,}.", flush=True)
        if max_records is not None and len(pmcids) >= max_records:
            break
    return pmcids


def collect_pmcids_for_date_range(
    client: httpx.Client,
    *,
    query: str,
    start_date: dt.date,
    end_date: dt.date,
    pmcids: list[str],
    seen: set[str],
    completed_ranges: set[str],
    discovered_pmcids_path: Path,
    discovery_ranges_path: Path,
    max_records: int | None,
    email: str | None,
    api_key: str | None,
    tool: str,
    request_sleep_seconds: float,
    retries: int,
    max_per_query: int = 9_000,
) -> None:
    if max_records is not None and len(pmcids) >= max_records:
        return
    range_key = date_range_key(start_date, end_date)
    if range_key in completed_ranges:
        print(f"    {range_key} already fetched; skipping.", flush=True)
        return
    term = f"({query}) AND {range_key}[pdat]"
    count = esearch_count(client, term, email=email, api_key=api_key, tool=tool, retries=retries)
    time.sleep(request_sleep_seconds)
    if count == 0:
        return
    if count > max_per_query and start_date < end_date:
        print(
            f"    {format_ncbi_date(start_date)}:{format_ncbi_date(end_date)} has {count:,} hits; splitting.",
            flush=True,
        )
        midpoint = start_date + ((end_date - start_date) / 2)
        collect_pmcids_for_date_range(
            client,
            query=query,
            start_date=start_date,
            end_date=midpoint,
            pmcids=pmcids,
            seen=seen,
            completed_ranges=completed_ranges,
            discovered_pmcids_path=discovered_pmcids_path,
            discovery_ranges_path=discovery_ranges_path,
            max_records=max_records,
            email=email,
            api_key=api_key,
            tool=tool,
            request_sleep_seconds=request_sleep_seconds,
            retries=retries,
            max_per_query=max_per_query,
        )
        collect_pmcids_for_date_range(
            client,
            query=query,
            start_date=midpoint + dt.timedelta(days=1),
            end_date=end_date,
            pmcids=pmcids,
            seen=seen,
            completed_ranges=completed_ranges,
            discovered_pmcids_path=discovered_pmcids_path,
            discovery_ranges_path=discovery_ranges_path,
            max_records=max_records,
            email=email,
            api_key=api_key,
            tool=tool,
            request_sleep_seconds=request_sleep_seconds,
            retries=retries,
            max_per_query=max_per_query,
        )
        return
    remaining = None if max_records is None else max(max_records - len(pmcids), 0)
    ids = esearch_all_ids(
        client,
        term,
        email=email,
        api_key=api_key,
        tool=tool,
        request_sleep_seconds=request_sleep_seconds,
        retries=retries,
        limit=remaining,
    )
    print(
        f"    {format_ncbi_date(start_date)}:{format_ncbi_date(end_date)} fetched {len(ids):,}/{count:,} IDs.",
        flush=True,
    )
    for pmcid in ids:
        pmcid = normalize_pmcid_number(pmcid)
        if pmcid not in seen:
            seen.add(pmcid)
            pmcids.append(pmcid)
            append_jsonl_object(
                discovered_pmcids_path,
                {
                    "pmcid": pmcid,
                    "range_key": range_key,
                    "discovered_at": dt.datetime.now(dt.UTC).isoformat(),
                },
            )
        if max_records is not None and len(pmcids) >= max_records:
            break
    range_status = "fetched" if len(ids) >= count else "partial"
    append_jsonl_object(
        discovery_ranges_path,
        {
            "range_key": range_key,
            "count": count,
            "fetched": len(ids),
            "status": range_status,
            "finished_at": dt.datetime.now(dt.UTC).isoformat(),
        },
    )
    if range_status == "fetched":
        completed_ranges.add(range_key)


def format_ncbi_date(value: dt.date) -> str:
    return value.strftime("%Y/%m/%d")


def date_range_key(start_date: dt.date, end_date: dt.date) -> str:
    return f"{format_ncbi_date(start_date)}:{format_ncbi_date(end_date)}"


def esearch_count(client: httpx.Client, term: str, **kwargs: Any) -> int:
    xml_text = eutils_get(client, "esearch.fcgi", {"db": "pmc", "term": term, "retmode": "xml", "retmax": "0"}, **kwargs)
    return parse_esearch_count(xml_text)


def esearch_all_ids(
    client: httpx.Client,
    term: str,
    *,
    email: str | None,
    api_key: str | None,
    tool: str,
    request_sleep_seconds: float,
    retries: int,
    limit: int | None = None,
    retmax: int = 10_000,
) -> list[str]:
    if limit == 0:
        return []
    page_size = min(retmax, limit) if limit is not None else retmax
    first = eutils_get(
        client,
        "esearch.fcgi",
        {"db": "pmc", "term": term, "retmode": "xml", "retmax": str(page_size), "retstart": "0"},
        email=email,
        api_key=api_key,
        tool=tool,
        retries=retries,
    )
    count, ids = parse_esearch_result(first)
    if limit is not None and len(ids) >= limit:
        return ids[:limit]
    for retstart in range(retmax, count, retmax):
        time.sleep(request_sleep_seconds)
        remaining = None if limit is None else max(limit - len(ids), 0)
        if remaining == 0:
            break
        page_size = min(retmax, remaining) if remaining is not None else retmax
        page = eutils_get(
            client,
            "esearch.fcgi",
            {"db": "pmc", "term": term, "retmode": "xml", "retmax": str(page_size), "retstart": str(retstart)},
            email=email,
            api_key=api_key,
            tool=tool,
            retries=retries,
        )
        _, page_ids = parse_esearch_result(page)
        ids.extend(page_ids)
    return ids[:limit] if limit is not None else ids


def parse_esearch_count(xml_text: str) -> int:
    count, _ = parse_esearch_result(xml_text)
    return count


def parse_esearch_result(xml_text: str) -> tuple[int, list[str]]:
    root = ElementTree.fromstring(xml_text)
    count_text = root.findtext("Count") or "0"
    ids = [node.text for node in root.findall("./IdList/Id") if node.text]
    return int(count_text), ids


def read_aws_metadata_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_objects(path, missing_ok=True):
        pmcid = normalize_pmcid_number(row.get("pmcid"))
        if row.get("status") == "failed":
            continue
        cache[pmcid] = row
    return cache


def read_pubmed_metadata_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_objects(path, missing_ok=True):
        pmid = clean_id(row.get("pmid"))
        if pmid:
            cache[pmid] = row
    return cache


def iter_eligible_article_versions(
    pmcids: list[str],
    *,
    max_workers: int,
    retries: int,
) -> Iterable[tuple[int, str, dict[str, Any] | None]]:
    total = len(pmcids)
    if total == 0:
        return
    worker_count = max(1, max_workers)
    limits = httpx.Limits(max_connections=worker_count * 2, max_keepalive_connections=worker_count)
    with httpx.Client(
        timeout=httpx.Timeout(60.0, connect=20.0),
        headers={"User-Agent": "dataset-generation-pmc-oa-aws-metadata/1.0"},
        limits=limits,
        follow_redirects=True,
    ) as client:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(select_eligible_article_version, client, pmcid, retries=retries): pmcid
                for pmcid in pmcids
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                pmcid = futures[future]
                try:
                    yield completed, pmcid, future.result()
                except Exception as exc:  # noqa: BLE001 - keep discovery moving.
                    LOGGER.warning("Failed AWS metadata check for PMC%s: %s", pmcid, exc)
                    yield completed, pmcid, {"_status": "failed", "_error": str(exc)}


def eutils_get(
    client: httpx.Client,
    endpoint: str,
    params: dict[str, str],
    *,
    email: str | None,
    api_key: str | None,
    tool: str,
    retries: int,
) -> Any:
    full_params = dict(params)
    full_params["tool"] = tool
    if email:
        full_params["email"] = email
    if api_key:
        full_params["api_key"] = api_key
    response = retry_get(client, f"{NCBI_EUTILS_BASE_URL}/{endpoint}", params=full_params, retries=retries)
    if full_params.get("retmode") == "json":
        return response.json()
    return response.text


def retry_get(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    retries: int,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GET failed for {url}: {last_error}") from last_error


def select_eligible_article_version(client: httpx.Client, pmcid_number: str, *, retries: int) -> dict[str, Any] | None:
    fast_metadata = fetch_optional_aws_json(client, f"metadata/PMC{pmcid_number}.1.json", retries=retries)
    if isinstance(fast_metadata, dict) and is_eligible_article_metadata(fast_metadata):
        return fast_metadata

    versions = list_article_versions(client, pmcid_number, retries=retries)
    eligible: list[dict[str, Any]] = []
    for version_prefix in versions:
        metadata = fetch_aws_json(client, f"metadata/{version_prefix.rstrip('/')}.json", retries=retries)
        if not isinstance(metadata, dict):
            continue
        if is_eligible_article_metadata(metadata):
            eligible.append(metadata)
    if not eligible:
        return None
    eligible.sort(key=lambda row: (flag_enabled(row.get("is_manuscript")), int(row.get("version") or 0)))
    return eligible[0]


def is_eligible_article_metadata(metadata: dict[str, Any]) -> bool:
    return (
        not flag_enabled(metadata.get("is_retracted"))
        and flag_enabled(metadata.get("is_pmc_openaccess"))
        and is_eligible_license(str(metadata.get("license_code", "")))
        and bool(metadata.get("pdf_url"))
    )


def flag_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def list_article_versions(client: httpx.Client, pmcid_number: str, *, retries: int) -> list[str]:
    prefix = f"PMC{pmcid_number}."
    params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
    response = retry_get(client, f"{PMC_AWS_BASE_URL}/", params=params, retries=retries)
    root = ElementTree.fromstring(response.text)
    versions: list[str] = []
    for prefix_node in root.findall(".//{*}CommonPrefixes/{*}Prefix"):
        if prefix_node.text:
            versions.append(prefix_node.text.strip("/"))
    return sorted(set(versions))


def fetch_aws_json(client: httpx.Client, key: str, *, retries: int) -> Any:
    response = retry_get(client, f"{PMC_AWS_BASE_URL}/{urllib.parse.quote(key)}", retries=retries)
    return response.json()


def fetch_optional_aws_json(client: httpx.Client, key: str, *, retries: int) -> Any:
    url = f"{PMC_AWS_BASE_URL}/{urllib.parse.quote(key)}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"GET failed for {url}: {last_error}") from last_error


def fetch_pubmed_metadata(
    client: httpx.Client,
    pmids: list[str],
    *,
    email: str | None,
    api_key: str | None,
    tool: str,
    request_sleep_seconds: float,
    retries: int,
    cache_path: Path | None = None,
    batch_size: int = 200,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    total = len(pmids)
    if total == 0:
        return records
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        if not batch:
            continue
        xml_text = eutils_get(
            client,
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            email=email,
            api_key=api_key,
            tool=tool,
            retries=retries,
        )
        batch_records = parse_pubmed_xml(xml_text)
        records.update(batch_records)
        if cache_path is not None:
            for record in batch_records.values():
                append_jsonl_object(cache_path, record)
        time.sleep(request_sleep_seconds)
        completed = min(start + len(batch), total)
        print_inline_progress("PubMed batches", completed, total, records=len(records))
    print()
    return records


def parse_pubmed_xml(xml_text: str) -> dict[str, dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = text_at(medline, "PMID")
        if not pmid:
            continue
        article_node = medline.find("Article")
        journal_node = article_node.find("Journal") if article_node is not None else None
        records[pmid] = {
            "pmid": pmid,
            "title": text_at(article_node, "ArticleTitle") if article_node is not None else None,
            "authors": parse_authors(article_node),
            "journal": text_at(journal_node, "Title") or text_at(journal_node, "ISOAbbreviation"),
            "publication_year": publication_year(article_node),
            "article_type": [normalize_space(node.text) for node in article.findall(".//PublicationType") if node.text],
            "language": text_at(article_node, "Language"),
            "mesh_terms": [
                normalize_space("".join(heading.itertext()))
                for heading in medline.findall(".//MeshHeading")
            ],
            "keywords": [
                normalize_space("".join(keyword.itertext()))
                for keyword in medline.findall(".//KeywordList/Keyword")
            ],
            "substances": [
                normalize_space(name.text)
                for name in medline.findall(".//Chemical/NameOfSubstance")
                if name.text
            ],
        }
    return records


def parse_authors(article_node: ElementTree.Element | None) -> list[dict[str, str | None]]:
    if article_node is None:
        return []
    authors: list[dict[str, str | None]] = []
    for author in article_node.findall(".//AuthorList/Author"):
        collective = text_at(author, "CollectiveName")
        if collective:
            authors.append({"first": None, "last": collective})
        else:
            authors.append({"first": text_at(author, "ForeName"), "last": text_at(author, "LastName")})
    return authors


def publication_year(article_node: ElementTree.Element | None) -> int | None:
    if article_node is None:
        return None
    for path in (
        "Journal/JournalIssue/PubDate/Year",
        "ArticleDate/Year",
        "Journal/JournalIssue/PubDate/MedlineDate",
    ):
        value = text_at(article_node, path)
        year = parse_year(value)
        if year:
            return year
    return None


def text_at(node: ElementTree.Element | None, path: str) -> str | None:
    if node is None:
        return None
    child = node.find(path)
    if child is None:
        return None
    return normalize_space("".join(child.itertext()))


def normalize_space(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def parse_year(value: str | None) -> int | None:
    if not value:
        return None
    for token in value.replace("-", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def clean_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_pmcid_number(value: Any) -> str:
    return str(value).strip().removeprefix("PMC").removeprefix("pmc")


def build_record(aws_meta: dict[str, Any], pubmed_meta: dict[str, Any]) -> dict[str, Any]:
    detailed_subject_metadata = {
        "mesh_terms": pubmed_meta.get("mesh_terms", []),
        "keywords": pubmed_meta.get("keywords", []),
        "substances": pubmed_meta.get("substances", []),
        "publication_types": pubmed_meta.get("article_type", []),
        "citation": aws_meta.get("citation"),
    }
    article_type = pubmed_meta.get("article_type") or []
    broad_domain = classify_broad_domain(detailed_subject_metadata, aws_meta, pubmed_meta)
    pmcid = normalize_pmcid_number(aws_meta["pmcid"])
    exact_license = normalize_license(str(aws_meta.get("license_code", "")))
    pdf_url = normalize_pmc_object_url(str(aws_meta["pdf_url"]))
    return {
        "pmcid": f"PMC{pmcid}",
        "pmid": clean_id(aws_meta.get("pmid")),
        "doi": clean_id(aws_meta.get("doi")),
        "title": pubmed_meta.get("title") or aws_meta.get("title"),
        "authors": pubmed_meta.get("authors", []),
        "publication_year": pubmed_meta.get("publication_year"),
        "journal": pubmed_meta.get("journal") or aws_meta.get("citation"),
        "article_type": article_type,
        "broad_domain": broad_domain,
        "detailed_subject_metadata": detailed_subject_metadata,
        "language": pubmed_meta.get("language"),
        "exact_license": exact_license,
        "license_code": aws_meta.get("license_code"),
        "article_version": aws_meta.get("version"),
        "is_manuscript": aws_meta.get("is_manuscript"),
        "pdf_source": pdf_url,
        "pdf_path": None,
        "pdf_md5": extract_md5(pdf_url),
        "xml_source": normalize_pmc_object_url(aws_meta.get("xml_url")),
        "text_source": normalize_pmc_object_url(aws_meta.get("text_url")),
    }


def normalize_pmc_object_url(value: Any) -> str | None:
    if value is None:
        return None
    url = str(value)
    if url.startswith("s3://pmc-oa-opendata/"):
        return f"{PMC_AWS_BASE_URL}/{url.removeprefix('s3://pmc-oa-opendata/')}"
    return url


def classify_broad_domain(
    detailed_subject_metadata: dict[str, Any],
    aws_meta: dict[str, Any],
    pubmed_meta: dict[str, Any],
) -> str | None:
    terms = []
    for value in detailed_subject_metadata.values():
        if isinstance(value, list):
            terms.extend(str(item) for item in value)
        elif value:
            terms.append(str(value))
    terms.extend(str(pubmed_meta.get(key, "")) for key in ("journal", "title"))
    terms.extend(str(aws_meta.get(key, "")) for key in ("title", "citation"))
    haystack = " | ".join(terms).lower()
    medical_score = sum(1 for term in MEDICAL_TERMS if term in haystack)
    biology_score = sum(1 for term in BIOLOGY_TERMS if term in haystack)
    if medical_score > biology_score:
        return "Medical/Clinical Research"
    if biology_score > 0:
        return "Biology"
    if medical_score > 0:
        return "Medical/Clinical Research"
    return None


def should_exclude_article_type(article_types: Iterable[str]) -> bool:
    lowered = {value.lower() for value in article_types}
    if any(any(keyword in value for keyword in EXCLUDED_ARTICLE_TYPE_KEYWORDS) for value in lowered):
        return True
    return bool(lowered) and not any(value in PREFERRED_ARTICLE_TYPES for value in lowered)


def extract_md5(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    values = query.get("md5")
    return values[0] if values else None


def download_pmc_pdfs(
    records: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_PDF_DIR,
    overwrite: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    retries: int = 3,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    total = len(rows)
    downloaded = skipped = failed = 0
    failures: list[dict[str, str]] = []
    started_at = time.monotonic()
    worker_count = max(1, max_workers)
    print(f"Downloading {total} PMC PDFs with {worker_count} workers into {output_path}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(download_pmc_pdf_task, row, output_path, overwrite=overwrite, retries=retries)
            for row in rows
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "downloaded":
                downloaded += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(result)
                LOGGER.warning("Failed to download %s: %s", result["pmcid"], result["error"])
            print_download_progress(completed, total, downloaded, skipped, failed, started_at)
    print()
    if failures:
        (output_path / "download_failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return {"total": total, "downloaded": downloaded, "skipped": skipped, "failed": failed, "failures": failures}


def assign_pdf_paths(records: list[dict[str, Any]], pdf_dir: Path) -> None:
    for row in records:
        domain_dir = str(row["broad_domain"]).replace("/", "_").replace(" ", "_")
        row["pdf_path"] = str(pdf_dir / domain_dir / f"{row['pmcid']}.pdf")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in materialized)
        + ("\n" if materialized else ""),
        encoding="utf-8",
    )


def count_jsonl_rows(path: str | Path) -> int:
    data_path = Path(path)
    if not data_path.exists():
        return 0
    return sum(1 for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())


def download_pmc_pdf_task(row: dict[str, Any], output_dir: Path, *, overwrite: bool, retries: int) -> dict[str, str]:
    pmcid = str(row["pmcid"])
    license_code = normalize_license(str(row.get("exact_license") or row.get("license_code") or ""))
    if license_code not in ELIGIBLE_LICENSES:
        return {"status": "failed", "pmcid": pmcid, "pdf_url": str(row.get("pdf_source")), "error": "ineligible license"}
    url = str(row["pdf_source"])
    destination = Path(row["pdf_path"]) if row.get("pdf_path") else output_dir / row["broad_domain"].replace("/", "_").replace(" ", "_") / f"{pmcid}.pdf"
    expected_md5 = row.get("pdf_md5")
    if destination.exists() and not overwrite and verify_pdf(destination, expected_md5):
        return {"status": "skipped", "pmcid": pmcid, "pdf_url": url, "path": str(destination), "error": ""}
    try:
        download_one_pmc_pdf(url, destination, expected_md5=expected_md5, retries=retries)
        return {"status": "downloaded", "pmcid": pmcid, "pdf_url": url, "path": str(destination), "error": ""}
    except Exception as exc:  # noqa: BLE001 - keep batch downloads moving.
        return {"status": "failed", "pmcid": pmcid, "pdf_url": url, "path": str(destination), "error": str(exc)}


def download_one_pmc_pdf(url: str, destination: Path, *, expected_md5: str | None, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0), follow_redirects=True) as client:
        response = retry_get(client, url, retries=retries)
    part_path.write_bytes(response.content)
    if not verify_pdf(part_path, expected_md5):
        part_path.unlink(missing_ok=True)
        raise ValueError("downloaded file failed PDF header or MD5 verification")
    part_path.replace(destination)


def verify_pdf(path: Path, expected_md5: str | None = None) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(4)
            if header != b"%PDF":
                return False
            if expected_md5:
                digest = hashlib.md5(header + file.read()).hexdigest()  # noqa: S324 - compares provider checksum.
                return digest.lower() == expected_md5.lower()
            return True
    except OSError:
        return False


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
    percent = (completed / total * 100) if total else 100.0
    eta_seconds = max(total - completed, 0) / rate if rate > 0 else 0
    print(
        f"\r{completed}/{total} ({percent:5.1f}%) downloaded={downloaded} skipped={skipped} "
        f"failed={failed} rate={rate:0.1f}/s eta={format_duration(eta_seconds)}",
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


def build_report(records: list[dict[str, Any]], *, total_discovered: int, target_per_domain: int) -> dict[str, Any]:
    license_counts = Counter(row["exact_license"] for row in records)
    domain_counts = Counter(row["broad_domain"] for row in records)
    year_counts = Counter(str(row["publication_year"] or "unknown") for row in records)
    subject_counts: Counter[str] = Counter()
    for row in records:
        for term in row.get("detailed_subject_metadata", {}).get("mesh_terms", []):
            subject_counts[term] += 1
    return {
        "total_records_discovered": total_discovered,
        "eligible_records_written": len(records),
        "cc_by_count": license_counts["cc by"],
        "cc0_count": license_counts["cc0"],
        "biology_count": domain_counts["Biology"],
        "medical_clinical_count": domain_counts["Medical/Clinical Research"],
        "target_per_domain": target_per_domain,
        "strict_filter_has_enough_material": {
            "biology": domain_counts["Biology"] >= target_per_domain,
            "medical_clinical": domain_counts["Medical/Clinical Research"] >= target_per_domain,
            "both_targets_met": (
                domain_counts["Biology"] >= target_per_domain
                and domain_counts["Medical/Clinical Research"] >= target_per_domain
            ),
        },
        "counts_by_year": dict(sorted(year_counts.items())),
        "counts_by_subject": dict(subject_counts.most_common(100)),
        "pdf_downloads": {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0},
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Strict PMC OA Dataset Report",
        "",
        f"- Total records discovered: {report['total_records_discovered']}",
        f"- Eligible records written: {report['eligible_records_written']}",
        f"- CC BY count: {report['cc_by_count']}",
        f"- CC0 count: {report['cc0_count']}",
        f"- Biology count: {report['biology_count']}",
        f"- Medical/Clinical count: {report['medical_clinical_count']}",
        f"- PDFs successfully downloaded: {report['pdf_downloads'].get('downloaded', 0) + report['pdf_downloads'].get('skipped', 0)}",
        f"- Failed or missing PDFs: {report['pdf_downloads'].get('failed', 0)}",
        f"- Biology target met: {report['strict_filter_has_enough_material']['biology']}",
        f"- Medical/Clinical target met: {report['strict_filter_has_enough_material']['medical_clinical']}",
        f"- Both targets met: {report['strict_filter_has_enough_material']['both_targets_met']}",
        "",
        "## Top Subjects",
    ]
    for subject, count in list(report["counts_by_subject"].items())[:25]:
        lines.append(f"- {subject}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_objects(path)


def write_manifest_csv(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    fieldnames = [
        "pmcid",
        "pmid",
        "doi",
        "title",
        "publication_year",
        "journal",
        "broad_domain",
        "language",
        "exact_license",
        "pdf_source",
        "pdf_path",
    ]
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a strict CC BY/CC0 PMC OA Biology and Medical PDF dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--query", default=DEFAULT_DISCOVERY_QUERY)
    parser.add_argument("--target-per-domain", type=int, default=DEFAULT_TARGET_PER_DOMAIN)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--email", help="NCBI contact email, recommended for E-Utilities.")
    parser.add_argument("--api-key", help="Optional NCBI API key.")
    parser.add_argument("--tool", default="dataset-generation-pmc-oa")
    parser.add_argument("--metadata-only", action="store_true", help="Do not download PDFs.")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs after strict license gating.")
    parser.add_argument("--overwrite-pdfs", action="store_true")
    parser.add_argument("--include-non-english", action="store_true")
    parser.add_argument("--include-non-research", action="store_true")
    parser.add_argument("--max-records", type=int, help="Discovery cap for smoke tests.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--aws-metadata-workers",
        type=int,
        default=DEFAULT_AWS_METADATA_WORKERS,
        help="Concurrent PMC AWS JSON metadata/license checks before PDF download.",
    )
    parser.add_argument("--request-sleep-seconds", type=float, default=0.11)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = build_pmc_oa_subset(
        output_dir=args.output_dir,
        query=args.query,
        target_per_domain=args.target_per_domain,
        start_year=args.start_year,
        end_year=args.end_year,
        email=args.email,
        api_key=args.api_key,
        tool=args.tool,
        metadata_only=args.metadata_only,
        download_pdfs=args.download_pdfs,
        pdf_dir=args.pdf_dir,
        overwrite_pdfs=args.overwrite_pdfs,
        include_non_english=args.include_non_english,
        include_non_research=args.include_non_research,
        max_records=args.max_records,
        max_workers=args.max_workers,
        aws_metadata_workers=args.aws_metadata_workers,
        request_sleep_seconds=args.request_sleep_seconds,
        retries=args.retries,
    )
    records = load_records(result.records_path)
    write_manifest_csv(records, Path(args.output_dir) / "manifest.csv")
    return {
        "records": str(result.records_path),
        "report": str(result.report_path),
        "record_count": result.record_count,
        "pdf_stats": result.pdf_stats,
    }
