"""Execute resolved Hydra dataset stages and preserve a domain settings snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


STAGES = {
    "describe_figures", "describe_textual_clues", "generate_queries",
    "judge_queries", "reduce_and_compact", "extract_mineru",
}


def absolute(path: str, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def managed_settings(config: dict[str, Any], data_root: Path, repo_root: Path) -> dict[str, Any]:
    """Adapt the resolved config to the existing generation functions."""
    import copy
    settings = copy.deepcopy({key: config[key] for key in (
        "dataset", "visual_descriptions", "textual_descriptions", "visual_query",
    )})
    settings["dataset"]["root"] = str(data_root)
    settings["dataset"]["preprocessed"] = str(absolute(config["dataset"]["processed_root"], repo_root))
    for section in ("visual_descriptions", "textual_descriptions", "visual_query"):
        settings[section]["prompt"]["template"] = str(absolute(settings[section]["prompt"]["template"], repo_root))
    settings["visual_query"]["judgement"]["template"] = str(absolute(settings["visual_query"]["judgement"]["template"], repo_root))
    settings["visual_query"]["output"]["dir"] = str(absolute(settings["visual_query"]["output"]["dir"], repo_root))
    return settings


def build_plan(config: dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage = config["stage"]
    name = stage["name"]
    if name not in STAGES:
        raise ValueError(f"Unsupported stage: {name}")
    split = config["split"]["name"]
    if split not in {"train", "test", "train+test"}:
        raise ValueError(f"Unsupported split: {split}")
    if name not in {"reduce_and_compact", "extract_mineru"} and split == "train+test":
        raise ValueError("Generation stages require split=train or split=test")
    launcher = config["launcher"]
    kind = launcher["kind"]
    if kind not in {"local", "slurm", "pbs"}:
        raise ValueError(f"Unsupported launcher kind: {kind}")
    data_root = absolute(config["dataset"]["root"], repo_root)
    run_root = absolute(launcher["log_root"], repo_root)
    run_dir = run_dir or run_root / name
    plan = {
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "launcher": launcher,
        "dataset": config["dataset"],
        "stage": stage,
        "split": split,
        "settings_snapshot": str(run_dir / "settings.yaml") if name not in {"reduce_and_compact", "extract_mineru"} else None,
    }
    settings = managed_settings(config, data_root, repo_root) if name not in {"reduce_and_compact", "extract_mineru"} else None
    return plan, settings


def scheduler_command(plan: dict[str, Any], script: Path, log_dir: Path) -> list[str]:
    launcher = plan["launcher"]
    cpus, gpus = int(launcher["cpus"]), int(launcher["gpus"])
    if cpus < 1 or gpus < 0:
        raise ValueError("launcher.cpus must be positive and launcher.gpus nonnegative")
    if launcher["kind"] != "pbs":
        raise ValueError("Only PBS uses the custom scheduler adapter")
    select = f"select=1:ncpus={cpus}:mem={launcher['memory']}"
    if gpus:
        select += f":ngpus={gpus}"
    return ["qsub", "-N", plan["stage"]["name"], "-q", launcher["queue"],
            "-l", select, "-l", f"walltime={launcher['walltime']}",
            "-o", str(log_dir), "-e", str(log_dir), str(script)]


def job_script(plan_path: Path, plan: dict[str, Any]) -> str:
    root = plan["repo_root"]
    python = absolute(plan["launcher"]["python"], Path(root))
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"cd {shlex.quote(root)}",
        f"export PYTHONPATH={shlex.quote(str(Path(root) / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"exec {shlex.quote(str(python))} -m hydra_run --worker {shlex.quote(str(plan_path))}",
        "",
    ])


def execute_stage(plan: dict[str, Any]) -> None:
    """Invoke normal Python functions with a resolved settings file or typed arguments."""
    repo_root = Path(plan["repo_root"])
    os.environ["HF_HOME"] = str(absolute(plan["launcher"]["model_cache"], repo_root))
    os.environ["XDG_CACHE_HOME"] = str(absolute(plan["launcher"]["cache_root"], repo_root))

    from clues.textual_clue_descriptions import build_parser as textual_parser, run as describe_text
    from clues.vl_figure_descriptions import build_parser as visual_parser, run as describe_figures
    from queries.query_generation import build_parser as query_parser, run as generate_queries
    from queries.judge_queries import build_parser as judge_parser, main as judge_queries

    name = plan["stage"]["name"]
    settings = plan["settings_snapshot"]
    split = plan["split"]
    if name == "describe_figures":
        describe_figures(visual_parser().parse_args(["--settings", settings, "--set", split]))
    elif name == "describe_textual_clues":
        describe_text(textual_parser().parse_args(["--settings", settings, "--set", split]))
    elif name == "generate_queries":
        args = query_parser().parse_args(["--settings", settings, "--set", split])
        args.limit = plan["stage"].get("limit")
        generate_queries(args)
    elif name == "judge_queries":
        judge_queries(judge_parser().parse_args(["--settings", settings, "--set", split]))
    elif name == "extract_mineru":
        from extraction.mineru_extraction import build_extract_parser, run_extract
        repo_root = Path(plan["repo_root"])
        dataset, stage = plan["dataset"], plan["stage"]
        args = build_extract_parser().parse_args([])
        args.input_dir = absolute(dataset["pdf_dir"], repo_root)
        args.output_dir = absolute(dataset["processed_root"], repo_root)
        args.split_index = absolute(dataset["split_index"], repo_root) if dataset["split_index"] else None
        args.split = split
        for key in ("api_url", "backend", "limit", "max_in_flight", "retries",
                    "start_index", "end_index", "all_domain_pdfs", "retry_incomplete_only"):
            setattr(args, key, stage[key])
        run_extract(args)
    elif name == "reduce_and_compact":
        from preprocessing.reduce_preprocessed_collection import main as reduce
        from preprocessing.compact_preprocessed_collection import main as compact
        dataset, stage = plan["dataset"], plan["stage"]
        repo_root = Path(plan["repo_root"])
        preprocessed = absolute(dataset["processed_root"], repo_root)
        pdf_dir = absolute(dataset["pdf_dir"], repo_root)
        split_index = absolute(dataset["split_index"], repo_root) if dataset["split_index"] else None
        common = dict(preprocessed_dir=preprocessed, split_index=split_index, split=split,
                      workers=int(stage["workers"]), dry_run=bool(stage["dry_run"]),
                      fail_fast=bool(stage["fail_fast"]), progress_every=int(stage["progress_every"]))
        reduce(argparse.Namespace(**common, report=preprocessed.parent / "preprocessed_analysis_report.json"))
        compact(argparse.Namespace(**common, root_dir=Path(plan["data_root"]), pdf_dir=pdf_dir))


def execute_worker(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    os.chdir(plan["repo_root"])
    execute_stage(plan)
    return 0


def launch(config: dict[str, Any], repo_root: str | Path, *, native_slurm: bool = False,
           hydra_run_dir: str | Path | None = None) -> None:
    repo_root = Path(repo_root).resolve()
    plan, settings = build_plan(config, repo_root, Path(hydra_run_dir) if hydra_run_dir else None)
    kind = plan["launcher"]["kind"]
    if kind == "slurm" and not native_slurm and not config["dry_run"]:
        raise ValueError("Slurm profiles use Hydra Submitit; launch with -m")
    run_dir = Path(plan["run_dir"])
    plan_path = run_dir / "plan.json"
    script = run_dir / "job.sh"
    submission = scheduler_command(plan, script, run_dir) if kind == "pbs" else (
        {"hydra_launcher": "submitit_slurm", "mode": "MULTIRUN"} if kind == "slurm" else "local")
    if config["dry_run"]:
        print(json.dumps({"plan": plan, "settings": settings, "submission": submission}, indent=2))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        Path(plan["settings_snapshot"]).write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if kind == "pbs":
        script.write_text(job_script(plan_path, plan), encoding="utf-8")
        result = subprocess.run(submission, cwd=repo_root, text=True, capture_output=True, check=False)
        (run_dir / "submission.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"qsub failed ({result.returncode}): {result.stderr.strip()}")
        print(f"Submitted PBS job {result.stdout.strip()}\nPlan: {plan_path}")
        return
    execute_stage(plan)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(execute_worker(Path(sys.argv[2])))
    raise SystemExit("Use dataset-generation stage=<stage> launcher=<profile>")
