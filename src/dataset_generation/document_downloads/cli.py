"""Download source-document PDFs from prepared corpus manifests."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_generation.document_downloads.acl_subset import DEFAULT_MAX_WORKERS as ACL_DEFAULT_MAX_WORKERS
from dataset_generation.document_downloads.acl_subset import DEFAULT_OUTPUT_DIR as ACL_DEFAULT_OUTPUT_DIR
from dataset_generation.document_downloads.acl_subset import DEFAULT_PDF_DIR as ACL_DEFAULT_PDF_DIR
from dataset_generation.document_downloads.acl_subset import build_acl_subset
from dataset_generation.document_downloads.acl_subset import download_acl_pdfs
from dataset_generation.document_downloads.arxiv_open_reuse import DEFAULT_MAX_RETRIES as ARXIV_DEFAULT_MAX_RETRIES
from dataset_generation.document_downloads.arxiv_open_reuse import DEFAULT_OUTPUT_DIR as ARXIV_DEFAULT_OUTPUT_DIR
from dataset_generation.document_downloads.arxiv_open_reuse import DEFAULT_PDF_DIR as ARXIV_DEFAULT_PDF_DIR
from dataset_generation.document_downloads.arxiv_open_reuse import (
    DEFAULT_REQUEST_DELAY_SECONDS as ARXIV_DEFAULT_REQUEST_DELAY_SECONDS,
)
from dataset_generation.document_downloads.arxiv_open_reuse import (
    DEFAULT_TARGET_PER_DOMAIN as ARXIV_DEFAULT_TARGET_PER_DOMAIN,
)
from dataset_generation.document_downloads.arxiv_open_reuse import download_eligible_pdfs
from dataset_generation.document_downloads.arxiv_open_reuse import select_records_for_domain_targets
from dataset_generation.jsonl import read_jsonl_objects
from dataset_generation.document_downloads.pmc_oa_subset import DEFAULT_MAX_WORKERS as PMC_DEFAULT_MAX_WORKERS
from dataset_generation.document_downloads.pmc_oa_subset import DEFAULT_OUTPUT_DIR as PMC_DEFAULT_OUTPUT_DIR
from dataset_generation.document_downloads.pmc_oa_subset import DEFAULT_PDF_DIR as PMC_DEFAULT_PDF_DIR
from dataset_generation.document_downloads.pmc_oa_subset import assign_pdf_paths
from dataset_generation.document_downloads.pmc_oa_subset import download_pmc_pdfs


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    corpus: str
    manifest_path: Path
    output_dir: Path
    stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "manifest": str(self.manifest_path),
            "output_dir": str(self.output_dir),
            "stats": self.stats,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download-documents",
        description="Download document PDFs from source manifests.",
    )
    subparsers = parser.add_subparsers(dest="corpus", required=True)

    acl = subparsers.add_parser("acl", help="Download ACL Anthology subset PDFs.")
    acl.add_argument("--manifest", type=Path, default=ACL_DEFAULT_OUTPUT_DIR / "papers.jsonl")
    acl.add_argument("--output-dir", type=Path, default=ACL_DEFAULT_PDF_DIR)
    acl.add_argument(
        "--build-metadata-if-missing",
        action="store_true",
        help="Build the ACL papers.jsonl manifest before downloading when it does not exist.",
    )
    acl.add_argument("--refresh-metadata", action="store_true", help="Refetch ACL XML before downloading.")
    acl.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    acl.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay after each ACL PDF download.")
    acl.add_argument("--max-workers", type=int, default=ACL_DEFAULT_MAX_WORKERS)

    arxiv = subparsers.add_parser("arxiv-open-reuse", help="Download eligible arXiv open-reuse PDFs.")
    arxiv.add_argument("--manifest", type=Path, default=ARXIV_DEFAULT_OUTPUT_DIR / "eligible_records.jsonl")
    arxiv.add_argument("--output-dir", type=Path, default=ARXIV_DEFAULT_PDF_DIR)
    arxiv.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    arxiv.add_argument(
        "--download-all-eligible",
        action="store_true",
        help="Download every record in the manifest instead of the per-domain target selection.",
    )
    arxiv.add_argument("--target-per-domain", type=int, default=ARXIV_DEFAULT_TARGET_PER_DOMAIN)
    arxiv.add_argument("--request-delay-seconds", type=float, default=ARXIV_DEFAULT_REQUEST_DELAY_SECONDS)
    arxiv.add_argument("--timeout-seconds", type=float, default=120.0)
    arxiv.add_argument("--max-retries", type=int, default=ARXIV_DEFAULT_MAX_RETRIES)

    pmc = subparsers.add_parser("pmc-oa", help="Download strict PMC OA subset PDFs.")
    pmc.add_argument("--manifest", type=Path, default=PMC_DEFAULT_OUTPUT_DIR / "papers.jsonl")
    pmc.add_argument("--output-dir", type=Path, default=PMC_DEFAULT_PDF_DIR)
    pmc.add_argument("--overwrite-pdfs", action="store_true", help="Replace existing valid PDF files.")
    pmc.add_argument("--max-workers", type=int, default=PMC_DEFAULT_MAX_WORKERS)
    pmc.add_argument("--retries", type=int, default=3)

    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.corpus == "acl":
        return download_acl_documents(args).as_dict()
    if args.corpus == "arxiv-open-reuse":
        return download_arxiv_open_reuse_documents(args).as_dict()
    if args.corpus == "pmc-oa":
        return download_pmc_oa_documents(args).as_dict()
    raise ValueError(f"Unsupported corpus: {args.corpus}")


def download_acl_documents(args: argparse.Namespace) -> DownloadResult:
    manifest_path = Path(args.manifest)
    if args.refresh_metadata or (args.build_metadata_if_missing and not manifest_path.exists()):
        build_acl_subset(
            output_dir=manifest_path.parent,
            refresh_metadata=args.refresh_metadata,
            download_pdfs=False,
        )
    papers = read_jsonl_objects(manifest_path)
    stats = download_acl_pdfs(
        papers,
        output_dir=args.output_dir,
        overwrite=args.overwrite_pdfs,
        sleep_seconds=args.sleep_seconds,
        max_workers=args.max_workers,
    )
    return DownloadResult("acl", manifest_path, Path(args.output_dir), stats)


def download_arxiv_open_reuse_documents(args: argparse.Namespace) -> DownloadResult:
    manifest_path = Path(args.manifest)
    eligible_records = read_jsonl_objects(manifest_path)
    records = (
        eligible_records
        if args.download_all_eligible
        else select_records_for_domain_targets(eligible_records, target_per_domain=args.target_per_domain)
    )
    stats = download_eligible_pdfs(
        records,
        output_dir=args.output_dir,
        overwrite=args.overwrite_pdfs,
        request_delay_seconds=args.request_delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    stats = {"total": len(records), **stats}
    return DownloadResult("arxiv-open-reuse", manifest_path, Path(args.output_dir), stats)


def download_pmc_oa_documents(args: argparse.Namespace) -> DownloadResult:
    manifest_path = Path(args.manifest)
    records = read_jsonl_objects(manifest_path)
    output_dir = Path(args.output_dir)
    assign_pdf_paths(records, output_dir)
    stats = download_pmc_pdfs(
        records,
        output_dir=output_dir,
        overwrite=args.overwrite_pdfs,
        max_workers=args.max_workers,
        retries=args.retries,
    )
    return DownloadResult("pmc-oa", manifest_path, output_dir, stats)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
