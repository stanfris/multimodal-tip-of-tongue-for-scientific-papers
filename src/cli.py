"""Command line interface for dataset generation."""

from __future__ import annotations

import argparse
import json
import sys

from hydra_entry import main as hydra_main
from pathlib import Path

from extraction.mineru_extraction import build_benchmark_parser as build_mineru_benchmark_parser
from extraction.mineru_extraction import build_extract_parser as build_mineru_extract_parser
from extraction.mineru_extraction import probe_environment as probe_mineru_environment
from extraction.mineru_extraction import run_benchmark as run_mineru_benchmark
from extraction.mineru_extraction import run_extract as run_mineru_extract
from document_downloads.arxiv_open_reuse import build_parser as build_arxiv_open_reuse_parser
from document_downloads.arxiv_open_reuse import run as run_arxiv_open_reuse
from document_downloads.cli import build_parser as build_document_downloads_parser
from document_downloads.cli import run as run_document_downloads
from preprocessing.parsed_dataset_stats import build_parser as build_parsed_stats_parser
from preprocessing.parsed_dataset_stats import run as run_parsed_stats
from document_downloads.pmc_oa_subset import build_parser as build_pmc_oa_subset_parser
from document_downloads.pmc_oa_subset import run as run_pmc_oa_subset
from dataset_packaging.hf_dataset_packaging import build_parser as build_hf_dataset_parser
from dataset_packaging.hf_dataset_packaging import run as run_hf_dataset
from dataset_packaging.storage import read_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("probe-mineru-env", help="Inspect local MinerU, CUDA, VLLM, and GPU availability.")
    subparsers.add_parser(
        "extract-mineru-pdfs",
        parents=[build_mineru_extract_parser()],
        add_help=False,
        help="Extract PDF Markdown and figures through a persistent MinerU API/router.",
    )
    subparsers.add_parser(
        "benchmark-mineru-pdfs",
        parents=[build_mineru_benchmark_parser()],
        add_help=False,
        help="Benchmark MinerU backends/efforts on a PDF sample.",
    )
    subparsers.add_parser(
        "build-arxiv-open-reuse",
        parents=[build_arxiv_open_reuse_parser()],
        add_help=False,
        help="Harvest arXiv OAI-PMH metadata, filter strict open licenses, then retrieve PDFs from bulk S3.",
    )
    subparsers.add_parser(
        "build-pmc-oa-subset",
        parents=[build_pmc_oa_subset_parser()],
        add_help=False,
        help="Build a strict CC BY/CC0 Biology and Medical/Clinical PMC OA PDF dataset.",
    )
    subparsers.add_parser(
        "download-documents",
        parents=[build_document_downloads_parser()],
        add_help=False,
        help="Download source-document PDFs from prepared corpus manifests.",
    )
    subparsers.add_parser(
        "prepare-hf-dataset",
        parents=[build_hf_dataset_parser()],
        add_help=False,
        help="Package, validate, and upload the shared PDF corpus to Hugging Face.",
    )

    stats = subparsers.add_parser("stats", help="Print stats for a generated artifact.")
    stats.add_argument("--dataset", required=True, help="Generated artifact directory.")

    parsed_stats = subparsers.add_parser(
        "parsed-dataset-stats",
        parents=[build_parsed_stats_parser()],
        add_help=False,
        help="Compute parsed-paper abstract overlap and figure/table image count statistics.",
    )

    return parser


def legacy_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "probe-mineru-env":
        print(json.dumps(probe_mineru_environment(), indent=2, sort_keys=True))
        return 0

    if args.command == "extract-mineru-pdfs":
        run_mineru_extract(args)
        return 0

    if args.command == "benchmark-mineru-pdfs":
        run_mineru_benchmark(args)
        return 0

    if args.command == "build-arxiv-open-reuse":
        print(json.dumps(run_arxiv_open_reuse(args), indent=2, sort_keys=True))
        return 0

    if args.command == "build-pmc-oa-subset":
        print(json.dumps(run_pmc_oa_subset(args), indent=2, sort_keys=True))
        return 0

    if args.command == "download-documents":
        print(json.dumps(run_document_downloads(args), indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-hf-dataset":
        print(json.dumps(run_hf_dataset(args), indent=2, sort_keys=True))
        return 0

    if args.command == "stats":
        print(json.dumps(read_stats(args.dataset), indent=2, sort_keys=True))
        return 0

    if args.command == "parsed-dataset-stats":
        print(json.dumps(run_parsed_stats(args), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def main() -> None:
    """Hydra owns pipeline stages; operational utilities keep their own CLI."""
    legacy_utilities = {
        "probe-mineru-env", "extract-mineru-pdfs", "benchmark-mineru-pdfs",
        "build-arxiv-open-reuse", "build-pmc-oa-subset", "download-documents",
        "prepare-hf-dataset", "stats", "parsed-dataset-stats",
    }
    if len(sys.argv) > 1 and sys.argv[1] in legacy_utilities:
        raise SystemExit(legacy_main())
    hydra_main()


if __name__ == "__main__":
    main()
