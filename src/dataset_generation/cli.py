"""Command line interface for dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_generation.canonical import (
    DEFAULT_CANONICAL_DIR,
    DEFAULT_EXTRACTED_PAPERS_DIR,
    align_canonical_clues,
    build_canonical_from_extracted_papers,
)
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

    describe = subparsers.add_parser(
        "describe-figures",
        parents=[build_describe_figures_parser()],
        add_help=False,
        help="Generate Qwen-VL visual descriptions for figure images.",
    )
    describe_text = subparsers.add_parser(
        "describe-textual-clues",
        parents=[build_describe_text_parser()],
        add_help=False,
        help="Generate textual memory cues from matched paper markdown.",
    )
    generate_queries = subparsers.add_parser(
        "generate-queries",
        parents=[build_generate_queries_parser()],
        add_help=False,
        help="Generate visual-only and visual-and-text query collections from clue sidecars.",
    )
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

    canonical = subparsers.add_parser(
        "build-canonical",
        help="Build canonical papers.jsonl from ACL subset PDF extraction output.",
    )
    canonical.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_EXTRACTED_PAPERS_DIR,
        help="Directory containing one extracted paper directory per ACL paper.",
    )
    canonical.add_argument("--output-dir", type=Path, default=DEFAULT_CANONICAL_DIR, help="Canonical output directory.")

    align_clues = subparsers.add_parser(
        "align-canonical-clues",
        help="Rewrite canonical clue sidecars so they align with papers.jsonl.",
    )
    align_clues.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    align_clues.add_argument(
        "--textual-input",
        type=Path,
        action="append",
        default=[],
        help="Textual clue JSONL file or legacy interpretation directory. May be repeated.",
    )
    align_clues.add_argument(
        "--visual-input",
        type=Path,
        action="append",
        default=[],
        help="Visual clue JSONL file or legacy interpretation directory. May be repeated.",
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

    if args.command == "build-canonical":
        _, report = build_canonical_from_extracted_papers(args.input_dir, args.output_dir)
        print(json.dumps({"output": str(Path(args.output_dir)), "report": report}, indent=2, sort_keys=True))
        return 0

    if args.command == "align-canonical-clues":
        textual_inputs = args.textual_input or [Path(args.canonical_dir) / "textual_clues.jsonl"]
        visual_inputs = args.visual_input or [Path(args.canonical_dir) / "visual_clues.jsonl"]
        report = align_canonical_clues(
            args.canonical_dir,
            textual_inputs=textual_inputs,
            visual_inputs=visual_inputs,
        )
        print(json.dumps({"output": str(Path(args.canonical_dir)), "report": report}, indent=2, sort_keys=True))
        return 0

    if args.command == "stats":
        print(json.dumps(read_stats(args.dataset), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
