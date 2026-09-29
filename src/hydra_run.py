"""Hydra front door for local, Slurm, and PBS dataset jobs.

The scheduler only transports a frozen plan. Existing dataset-generation
commands still own the actual work and managed generation settings.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


MANAGED_COMMANDS = {"describe-figures", "describe-textual-clues", "generate-queries", "judge-queries"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def absolute(path: str, root: Path, *, follow_symlinks: bool = True) -> Path:
    candidate = Path(path).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    return candidate.resolve() if follow_symlinks else Path(os.path.abspath(candidate))


def managed_settings(config: dict[str, Any], data_root: Path, repo_root: Path) -> dict[str, Any]:
    """Materialize Hydra's managed sections for the existing CLI boundary."""
    settings = {key: config[key] for key in ("dataset", "visual_descriptions", "textual_descriptions", "visual_query")}
    settings["dataset"]["root"] = str(data_root)
    for section_name in ("visual_descriptions", "textual_descriptions", "visual_query"):
        section = settings[section_name]
        section["prompt"]["template"] = str(absolute(section["prompt"]["template"], repo_root))
    judgement = settings["visual_query"]["judgement"]
    judgement["template"] = str(absolute(judgement["template"], repo_root))
    settings["visual_query"]["output"]["dir"] = str(absolute(settings["visual_query"]["output"]["dir"], repo_root))
    return settings


def build_plan(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    launcher = config["launcher"]
    task = config["task"]
    kind = launcher["kind"]
    if kind not in {"local", "slurm", "pbs"}:
        raise ValueError(f"Unsupported launcher kind: {kind}")
    if not SAFE_NAME.fullmatch(task["name"]):
        raise ValueError("Task name must contain only letters, digits, underscore, dash, or dot")
    if not SAFE_NAME.fullmatch(task["script_folder"]):
        raise ValueError("task.script_folder must be a safe folder name")
    command = task["command"]
    if command not in MANAGED_COMMANDS | {"preprocess"}:
        raise ValueError(f"Unsupported task command: {command}")
    data_root = absolute(config["dataset"]["root"], repo_root)
    run_root = absolute(launcher["run_dir"], repo_root)
    log_root = absolute(config["log_root"], repo_root)
    # Keep the venv entry point: resolving its symlink can bypass pyvenv.cfg.
    python = absolute(launcher["python"], repo_root, follow_symlinks=False)
    args = task.get("args", [])
    if not isinstance(args, list) or any(isinstance(arg, (list, dict)) or arg is None for arg in args):
        raise ValueError("task.args must be a list of scalar values")
    args = [str(arg) for arg in args]
    task = {**task, "args": args}
    snapshot = None
    if command in MANAGED_COMMANDS:
        if task.get("set") not in {"train", "test"}:
            raise ValueError("Managed tasks require task.set=train or task.set=test")
        snapshot = managed_settings(config, data_root, repo_root)
    plan = {
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "run_root": str(run_root),
        "log_root": str(log_root),
        "python": str(python),
        "launcher": launcher,
        "task": task,
        "settings_snapshot": None,
    }
    return plan, snapshot


def worker_command(plan: dict[str, Any]) -> list[str]:
    task = plan["task"]
    command = task["command"]
    if command == "preprocess":
        return [
            str(Path(plan["repo_root"]) / "scripts/preprocessing/reduce_and_compact_preprocessed.sh"),
            "--root-dir", plan["data_root"], "--workers", str(plan["launcher"]["cpus"]),
            *task["args"],
        ]
    if command == "judge-queries":
        return [
            plan["python"], "-m", "queries.judge_queries",
            "--settings", plan["settings_snapshot"], "--set", task["set"], *task["args"],
        ]
    return [
        plan["python"], "-m", "cli", command,
        "--settings", plan["settings_snapshot"], "--set", task["set"], *task["args"],
    ]


def job_script(plan_path: Path, plan: dict[str, Any]) -> str:
    root = plan["repo_root"]
    python = plan["python"]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.extend([
        f"cd {shlex.quote(root)}",
        f"export PYTHONPATH={shlex.quote(str(Path(root) / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"exec {shlex.quote(python)} -m hydra_run --worker {shlex.quote(str(plan_path))}",
    ])
    return "\n".join(lines) + "\n"


def scheduler_command(plan: dict[str, Any], script: Path, log_dir: Path) -> list[str]:
    launcher = plan["launcher"]
    name = plan["task"]["name"]
    cpus = int(launcher["cpus"])
    gpus = int(launcher["gpus"])
    if cpus < 1 or gpus < 0:
        raise ValueError("launcher.cpus must be positive and launcher.gpus nonnegative")
    if launcher["kind"] == "pbs":
        select = f"select=1:ncpus={cpus}:mem={launcher['memory']}"
        if gpus:
            select += f":ngpus={gpus}"
        return [
            "qsub", "-N", name, "-q", launcher["queue"],
            "-l", select, "-l", f"walltime={launcher['walltime']}",
            "-o", str(log_dir), "-e", str(log_dir), str(script),
        ]
    raise ValueError("Only PBS uses the custom scheduler adapter")


def execute_worker(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    os.chdir(plan["repo_root"])
    log_dir = Path(plan["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "worker.log").open("a", encoding="utf-8") as log_file:
        with subprocess.Popen(
            worker_command(plan), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return process.wait()


def launch(config: dict[str, Any], repo_root: Path, *, native_slurm: bool = False) -> None:
    plan, snapshot = build_plan(config, repo_root)
    kind = plan["launcher"]["kind"]
    if kind == "slurm" and not native_slurm and not config["dry_run"]:
        raise ValueError("Slurm profiles use Hydra Submitit; launch with -m (or --multirun)")
    if kind != "local" and (not plan["launcher"].get("partition") and kind == "slurm" or not plan["launcher"].get("queue") and kind == "pbs"):
        raise ValueError("Scheduler profile needs a partition or queue")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(plan["run_root"]) / plan["task"]["name"] / timestamp
    log_dir = Path(plan["log_root"]) / plan["task"]["script_folder"] / plan["task"]["name"] / timestamp
    plan["log_dir"] = str(log_dir)
    plan_path = run_dir / "plan.json"
    script = run_dir / "job.sh"
    if snapshot is not None:
        plan["settings_snapshot"] = str(run_dir / "settings.yaml")
    submission = (
        {"hydra_launcher": "submitit_slurm", "mode": "MULTIRUN"} if kind == "slurm"
        else scheduler_command(plan, script, log_dir) if kind == "pbs"
        else worker_command(plan)
    )
    if config["dry_run"]:
        print(json.dumps({"plan": plan, "settings": snapshot, "submission": submission}, indent=2))
        return
    if not Path(plan["python"]).is_file():
        raise FileNotFoundError(f"Python environment not found: {plan['python']}")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        (run_dir / "settings.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if kind == "pbs":
        script.write_text(job_script(plan_path, plan), encoding="utf-8")
    if kind == "local":
        raise SystemExit(execute_worker(plan_path))
    if kind == "slurm":
        result = execute_worker(plan_path)
        if result:
            raise RuntimeError(f"Slurm worker failed with exit code {result}; plan: {plan_path}")
        return
    result = subprocess.run(submission, cwd=repo_root, text=True, capture_output=True, check=False)
    (log_dir / "submission.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{submission[0]} failed ({result.returncode}): {result.stderr.strip()}")
    print(f"Submitted {kind} job {result.stdout.strip()}\nPlan: {plan_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(execute_worker(Path(sys.argv[2])))
    raise SystemExit("Use python -m hydra_entry launcher=local_gpu task=describe_figures")
