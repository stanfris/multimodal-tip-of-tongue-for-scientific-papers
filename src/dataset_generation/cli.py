"""Command line interface for dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_generation.build import BuildConfig, build_combined_dataset
from dataset_generation.canonical import align_canonical_clues, build_canonical_papers
from dataset_generation.manager import run_from_config
from dataset_generation.sources import DEFAULT_DATA_DIR, download_hf_sources
from dataset_generation.sources import materialize_acl_fig_images
from dataset_generation.storage import SUPPORTED_FORMATS, read_stats, write_dataset_artifact
from dataset_generation.query_generation import build_parser as build_generate_queries_parser
from dataset_generation.query_generation import run as run_generate_queries
from dataset_generation.textual_clue_descriptions import build_parser as build_describe_text_parser
from dataset_generation.textual_clue_descriptions import run as run_describe_text
from dataset_generation.vl_figure_descriptions import build_parser as build_describe_figures_parser
from dataset_generation.vl_figure_descriptions import run as run_describe_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset-generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-sources",
        help="Download source HF datasets into the local data folder.",
    )
    download.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Local data root.")
    download.add_argument("--split", default="train", help="ACL-fig split to download.")
    download.add_argument("--paper-split", default="train", help="ACL Anthology markdown split to download.")

    run = subparsers.add_parser("run", help="Run a managed dataset generation config.")
    run.add_argument("--config", required=True, help="YAML config file describing the run.")

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

    build = subparsers.add_parser("build", help="Build and persist the combined dataset artifact.")
    build.add_argument("--split", default="train", help="ACL-fig split to build.")
    build.add_argument("--paper-split", default="train", help="ACL Anthology markdown split.")
    build.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Local data root.")
    build.add_argument("--output", default=None, help="Output artifact directory.")
    build.add_argument("--format", default="parquet", choices=sorted(SUPPORTED_FORMATS))
    build.add_argument("--limit", type=int, default=None, help="Optional ACL-fig record limit for debugging.")
    build.add_argument("--paper-limit", type=int, default=None, help="Optional markdown stream limit for debugging.")
    build.add_argument("--seed", type=int, default=13)
    build.add_argument(
        "--no-images",
        action="store_true",
        help="Do not write ACL-Fig image files next to the processed artifact.",
    )
    build.add_argument("--no-streaming", action="store_true", help=argparse.SUPPRESS)
    build.add_argument(
        "--streaming",
        action="store_true",
        help="Stream markdown from Hugging Face when no local cache exists.",
    )
    build.add_argument(
        "--remote-sources",
        action="store_true",
        help="Ignore local source cache and load source datasets remotely.",
    )

    canonical = subparsers.add_parser("build-canonical", help="Build canonical paper-level papers.jsonl.")
    canonical.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local data root.")
    canonical.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Processed figure-row artifact directory. Defaults to data/processed/acl_fig_markdown/<split>.",
    )
    canonical.add_argument("--split", default="train", help="Processed dataset split used when --dataset is omitted.")
    canonical.add_argument("--output-dir", type=Path, default=None, help="Canonical output directory.")
    canonical.add_argument("--textual-interpretations", type=Path, default=None)
    canonical.add_argument("--visual-interpretations", type=Path, default=None)

    align_clues = subparsers.add_parser(
        "align-canonical-clues",
        help="Rewrite canonical clue sidecars so they align with papers.jsonl.",
    )
    align_clues.add_argument("--canonical-dir", type=Path, default=Path(DEFAULT_DATA_DIR) / "canonical")
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

    if args.command == "download-sources":
        paths = download_hf_sources(args.data_dir, fig_split=args.split, paper_split=args.paper_split)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        effective_argv = argv if argv is not None else sys.argv[1:]
        output_path = run_from_config(args.config, command=["dataset-generation", *effective_argv])
        print(json.dumps({"output": str(Path(output_path))}, indent=2, sort_keys=True))
        return 0

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

    if args.command == "build":
        effective_argv = argv if argv is not None else sys.argv[1:]
        config = BuildConfig(
            fig_split=args.split,
            paper_split=args.paper_split,
            data_dir=args.data_dir,
            prefer_local_sources=not args.remote_sources,
            streaming=args.streaming,
            limit=args.limit,
            paper_limit=args.paper_limit,
            seed=args.seed,
        )
        records, stats, metadata = build_combined_dataset(config)
        metadata["command"] = ["dataset-generation", *effective_argv]
        output = args.output or Path(args.data_dir) / "processed" / "acl_fig_markdown" / args.split
        if not args.no_images:
            image_stats = materialize_acl_fig_images(
                records,
                output,
                split=args.split,
                data_dir=args.data_dir,
                prefer_local=not args.remote_sources,
            )
            stats["images_written"] = image_stats["images_written"]
            stats["missing_images"] = image_stats["missing_images"]
            metadata["image_artifacts"] = image_stats
            metadata["layout"]["images"] = "images/<record_id>.<ext>"
        output_path = write_dataset_artifact(records, output, metadata, stats, args.format)
        print(json.dumps({"output": str(Path(output_path)), "stats": stats}, indent=2, sort_keys=True))
        return 0

    if args.command == "build-canonical":
        dataset = args.dataset or Path(args.data_dir) / "processed" / "acl_fig_markdown" / args.split
        output_dir = args.output_dir or Path(args.data_dir) / "canonical"
        textual = args.textual_interpretations or Path(args.data_dir) / "interim" / "interpretations" / "qwen3_textual_clue_description"
        visual = args.visual_interpretations or Path(args.data_dir) / "interim" / "interpretations" / "qwen3_vl_figure_description"
        _, report = build_canonical_papers(
            dataset,
            output_dir,
            textual_interpretations=textual if textual.exists() else None,
            visual_interpretations=visual if visual.exists() else None,
        )
        print(json.dumps({"output": str(Path(output_dir)), "report": report}, indent=2, sort_keys=True))
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
