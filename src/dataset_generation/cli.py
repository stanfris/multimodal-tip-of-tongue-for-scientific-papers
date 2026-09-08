"""Command line interface for dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_generation.mineru_extraction import build_benchmark_parser as build_mineru_benchmark_parser
from dataset_generation.mineru_extraction import build_extract_parser as build_mineru_extract_parser
from dataset_generation.mineru_extraction import probe_environment as probe_mineru_environment
from dataset_generation.mineru_extraction import run_benchmark as run_mineru_benchmark
from dataset_generation.mineru_extraction import run_extract as run_mineru_extract
from dataset_generation.storage import read_stats
from dataset_generation.query_generation import build_parser as build_generate_queries_parser
from dataset_generation.query_generation import run as run_generate_queries
from dataset_generation.textual_clue_descriptions import build_parser as build_describe_text_parser
from dataset_generation.textual_clue_descriptions import run as run_describe_text
from dataset_generation.vl_figure_descriptions import build_parser as build_describe_figures_parser
from dataset_generation.vl_figure_descriptions import run as run_describe_figures


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

    describe = subparsers.add_parser(
        "describe-figures",
        parents=[build_describe_figures_parser()],
        add_help=False,
        help="Generate Qwen-VL visual descriptions for ACL set figure images.",
    )
    describe_text = subparsers.add_parser(
        "describe-textual-clues",
        parents=[build_describe_text_parser()],
        add_help=False,
        help="Generate textual memory cues from ACL set paper markdown.",
    )
    generate_queries = subparsers.add_parser(
        "generate-queries",
        parents=[build_generate_queries_parser()],
        add_help=False,
        help="Generate visual-only and visual-and-text query collections from clue sidecars.",
    )

    stats = subparsers.add_parser("stats", help="Print stats for a generated artifact.")
    stats.add_argument("--dataset", required=True, help="Generated artifact directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "describe-figures":
        output_path = run_describe_figures(args)
        print(json.dumps({"output": str(Path(output_path))}, indent=2, sort_keys=True))
        return 0

    if args.command == "describe-textual-clues":
        output_path = run_describe_text(args)
        print(json.dumps({"output": str(Path(output_path))}, indent=2, sort_keys=True))
        return 0

    if args.command == "generate-queries":
        output_path = run_generate_queries(args)
        print(json.dumps({"output": str(Path(output_path))}, indent=2, sort_keys=True))
        return 0

    if args.command == "probe-mineru-env":
        print(json.dumps(probe_mineru_environment(), indent=2, sort_keys=True))
        return 0

    if args.command == "extract-mineru-pdfs":
        run_mineru_extract(args)
        return 0

    if args.command == "benchmark-mineru-pdfs":
        run_mineru_benchmark(args)
        return 0

    if args.command == "stats":
        print(json.dumps(read_stats(args.dataset), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
