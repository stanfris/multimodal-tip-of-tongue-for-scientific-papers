"""Execute resolved Hydra dataset stages and preserve a domain settings snapshot."""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STAGES = {
    "describe_figures", "describe_textual_clues", "generate_queries",
    "judge_queries", "reduce_and_compact", "extract_mineru",
}

STAGE_LOG_NAMES = {
    "reduce_and_compact": "preprocessing.log",
    "describe_figures": "describe_figures.log",
    "describe_textual_clues": "describe_textual_clues.log",
    "generate_queries": "generate_queries.log",
    "judge_queries": "judge_queries.log",
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
    settings["dataset"]["preprocessed"] = str(absolute(config["dataset"]["preprocessed"], repo_root))
    if config["dataset"]["split_index"] is not None:
        settings["dataset"]["split_index"] = str(absolute(config["dataset"]["split_index"], repo_root))
    for section in ("visual_descriptions", "textual_descriptions", "visual_query"):
        settings[section]["prompt"]["template"] = str(absolute(settings[section]["prompt"]["template"], repo_root))
    settings["visual_query"]["judgement"]["template"] = str(absolute(settings["visual_query"]["judgement"]["template"], repo_root))
    settings["visual_query"]["output"]["dir"] = str(absolute(settings["visual_query"]["output"]["dir"], repo_root))
    launcher = config["launcher"]
    runtime = {key: launcher.get(key) for key in ("device_map", "dtype", "attn_implementation")}
    vllm = dict(launcher.get("vllm") or {})
    model_limits = vllm.pop("model_max_num_seqs", {})
    if vllm.get("max_num_seqs") is None and config["model"]["name"] in model_limits:
        vllm["max_num_seqs"] = model_limits[config["model"]["name"]]
    if config["model"]["name"] not in {"Qwen/Qwen3-VL-4B-Instruct", "google/gemma-3-27b-it"}:
        vllm.pop("limit_mm_per_prompt", None)
    settings["visual_descriptions"]["runtime"] = {"transformers": runtime, "vllm": vllm}
    settings["textual_descriptions"]["runtime"] = {"transformers": runtime, "vllm": vllm}
    settings["visual_query"]["runtime"] = {"transformers": runtime, "vllm": vllm}
    judge_vllm = dict(vllm)
    judgement_model = settings["visual_query"]["judgement"].get("model")
    if launcher.get("vllm", {}).get("max_num_seqs") is None and judgement_model in model_limits:
        judge_vllm["max_num_seqs"] = model_limits[judgement_model]
    if judgement_model in {"Qwen/Qwen3-VL-4B-Instruct", "google/gemma-3-27b-it"}:
        judge_vllm["limit_mm_per_prompt"] = launcher.get("vllm", {}).get("limit_mm_per_prompt", {"image": 16})
    settings["visual_query"]["judgement"]["runtime"] = {"transformers": runtime, "vllm": judge_vllm}
    return settings


def build_plan(config: dict[str, Any], repo_root: Path, run_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage = config["stage"]
    name = stage["name"]
    if name not in STAGES:
        raise ValueError(f"Unsupported stage: {name}")
    split = config["split"]["name"]
    if split not in {"train", "test", "train+test", "other"}:
        raise ValueError(f"Unsupported split: {split}")
    if name != "extract_mineru" and split == "other":
        raise ValueError("split=other is supported only for extract_mineru")
    if name not in {"reduce_and_compact", "extract_mineru"} and split == "train+test":
        raise ValueError("Generation stages require split=train or split=test")
    launcher = config["launcher"]
    kind = launcher["kind"]
    if kind not in {"local", "slurm", "pbs"}:
        raise ValueError(f"Unsupported launcher kind: {kind}")
    if name == "extract_mineru" and int(launcher["gpus"]) < 1:
        raise ValueError("extract_mineru requires a GPU launcher (launcher.gpus>=1)")
    if name == "extract_mineru":
        if stage["all_domain_pdfs"] and split == "other":
            raise ValueError("split=other cannot be combined with stage.all_domain_pdfs=true")
        port = int(stage["server_port"])
        if not 0 <= port <= 65535:
            raise ValueError("stage.server_port must be 0 or a valid TCP port")
        if int(stage["server_startup_timeout"]) < 1 or int(stage["server_concurrency"]) < 1:
            raise ValueError("MinerU server timeout and concurrency must be positive")
        if int(stage["max_in_flight"]) < 1:
            raise ValueError("MinerU max_in_flight must be positive")
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
    if name == "extract_mineru":
        plan["execution"] = {
            "jobs": 1, "components": ["MinerU server", "extraction caller"],
            "gpu_required": True,
            "resources": {key: launcher[key] for key in ("gpus", "cpus", "memory", "walltime")},
            "server_concurrency": int(stage["server_concurrency"]),
            "max_in_flight": int(stage["max_in_flight"]),
        }
        for key in ("partition", "queue", "project"):
            if key in launcher:
                plan["execution"]["resources"][key] = launcher[key]
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
    return [
        "qsub",
        "-N", plan["stage"]["name"],
        "-P", launcher["project"],
        "-q", launcher["queue"],
        "-l", select,
        "-l", f"walltime={launcher['walltime']}",
        "-o", str(log_dir),
        "-e", str(log_dir),
        str(script),
    ]


def job_script(plan_path: Path, plan: dict[str, Any]) -> str:
    root = plan["repo_root"]
    # Keep the venv entry point intact: resolving its symlink runs the base
    # interpreter and bypasses the venv's installed packages on PBS nodes.
    python = Path(plan["launcher"]["python"]).expanduser()
    if not python.is_absolute():
        python = Path(root) / python
    environment = []
    for key, value in (plan["launcher"].get("environment") or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid launcher environment variable name: {key!r}")
        environment.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"cd {shlex.quote(root)}",
        f"export PYTHONPATH={shlex.quote(str(Path(root) / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
        *environment,
        f"exec {shlex.quote(str(python))} -m hydra_run --worker {shlex.quote(str(plan_path))}",
        "",
    ])


def mineru_caller_args(plan: dict[str, Any], api_url: str) -> list[str]:
    """Translate resolved Hydra options to the dedicated environment's extraction CLI."""
    root = Path(plan["repo_root"])
    dataset, stage = plan["dataset"], plan["stage"]
    args = ["--input-dir", str(absolute(dataset["pdf_dir"], root)),
            "--output-dir", str(absolute(dataset["processed_root"], root)),
            "--api-url", api_url, "--split", plan["split"]]
    if dataset["split_index"] and not stage["all_domain_pdfs"]:
        args += ["--split-index", str(absolute(dataset["split_index"], root))]
    for key in ("backend", "max_in_flight", "retries", "start_index", "effort",
                "parse_method", "lang", "start_page_id", "poll_interval",
                "request_timeout", "result_timeout", "min_markdown_chars"):
        args += ["--" + key.replace("_", "-"), str(stage[key])]
    for key in ("limit", "end_index", "end_page_id"):
        if stage[key] is not None:
            args += ["--" + key.replace("_", "-"), str(stage[key])]
    if stage["domains"]:
        args += ["--domains", *stage["domains"]]
    for key in ("all_domain_pdfs", "retry_incomplete_only", "allow_tiny_markdown",
                "no_formula", "no_table"):
        if stage[key]:
            args.append("--" + key.replace("_", "-"))
    if stage["image_analysis"] is not None:
        args.append("--image-analysis" if stage["image_analysis"] else "--no-image-analysis")
    return args


def normalize_mineru_cuda_visibility(env: dict[str, str]) -> None:
    """Convert scheduler GPU UUID masks to indices accepted by pinned vLLM."""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return
    requested = [device.strip() for device in visible.split(",")]
    if any(not device for device in requested):
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES for MinerU: {visible!r}")
    if all(device.isdecimal() for device in requested):
        return
    if any(not (device.isdecimal() or device.startswith("GPU-")) for device in requested):
        raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES for MinerU: {visible!r}")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot map CUDA_VISIBLE_DEVICES GPU UUIDs with nvidia-smi") from exc
    devices = [(row[0].strip(), row[1].strip()) for row in csv.reader(result.stdout.splitlines())
               if len(row) >= 2]
    mapped = []
    for device in requested:
        if device.isdecimal():
            mapped.append(device)
            continue
        matches = [index for index, uuid in devices if uuid == device or uuid.startswith(device)]
        if len(matches) != 1:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES UUID {device!r} matched {len(matches)} GPUs in nvidia-smi; "
                "refusing to select another GPU"
            )
        mapped.append(matches[0])
    env["CUDA_VISIBLE_DEVICES"] = ",".join(mapped)
    print(f"MinerU CUDA_VISIBLE_DEVICES: {visible} -> {env['CUDA_VISIBLE_DEVICES']}", flush=True)


def run_mineru_job(plan: dict[str, Any]) -> None:
    """Run server and caller together inside one GPU allocation."""
    root = Path(plan["repo_root"])
    stage = plan["stage"]
    port = int(stage["server_port"])
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    if not 1 <= port <= 65535:
        raise ValueError("stage.server_port must be 0 or a valid TCP port")
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or str(os.getpid())
    scratch_root = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or
                        f"/tmp/{os.environ.get('USER', 'mineru')}")
    temp_base = scratch_root / f"mineru_{run_id}"
    temp_dir = temp_base / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in
                (plan["launcher"].get("environment") or {}).items()})
    normalize_mineru_cuda_visibility(env)
    env.update({
        "TMPDIR": str(temp_dir), "MINERU_PORT": str(port),
        "MINERU_PARENT_TMPDIR": os.environ.get("TMPDIR", ""),
        "MINERU_VENV": str(absolute(stage["mineru_venv"], root)),
        "MINERU_STARTUP_TIMEOUT": str(stage["server_startup_timeout"]),
        "MINERU_API_MAX_CONCURRENT_REQUESTS": str(stage["server_concurrency"]),
        "MINERU_SERVER_LOG": str(run_dir / "mineru.server.log"),
        "MINERU_CALLER_LOG": str(run_dir / "mineru.caller.log"),
        "MINERU_SERVER_PID_FILE": str(run_dir / "mineru.server.pid"),
        "MINERU_CALLER_PID_FILE": str(run_dir / "mineru.caller.pid"),
    })
    cmd = [str(root / "scripts/extraction/run_mineru_full_extraction_gpu.sh"),
           "--no-default-caller-args"]
    if stage["server_command"]:
        cmd += ["--server-cmd", stage["server_command"].replace("{port}", str(port))]
    cmd += ["--", *mineru_caller_args(plan, f"http://127.0.0.1:{port}")]
    print(f"One GPU {plan['launcher']['kind']} job: MinerU server and caller; "
          f"server_concurrency={stage['server_concurrency']} "
          f"max_in_flight={stage['max_in_flight']} temp={temp_dir}", flush=True)
    if not stage["monitor_usage"]:
        subprocess.run(cmd, cwd=root, env=env, check=True)
        return

    from extraction.mineru_telemetry import monitor_job, summarize_telemetry

    paths = {
        "repository": root,
        "data": Path(plan["data_root"]),
        "input_pdfs": absolute(plan["dataset"]["pdf_dir"], root),
        "output": absolute(plan["dataset"]["processed_root"], root),
        "cache": absolute(plan["launcher"]["cache_root"], root),
        "model_cache": absolute(plan["launcher"]["model_cache"], root),
        "tmp": temp_dir,
    }
    domains = stage.get("domains") or []
    state_dir = paths["output"] / "_runs" / domains[0] if len(domains) == 1 else paths["output"]
    started_at = time.time()
    try:
        with monitor_job(run_dir, paths, env):
            subprocess.run(cmd, cwd=root, env=env, check=True)
    finally:
        try:
            summary = summarize_telemetry(run_dir, state_dir / "last_run_summary.json", started_at=started_at)
            print("MinerU performance: " + json.dumps(summary, sort_keys=True), flush=True)
        except Exception as exc:
            print(f"Warning: MinerU telemetry summary unavailable: {exc}", file=sys.stderr, flush=True)


def execute_stage(plan: dict[str, Any]) -> None:
    """Invoke normal Python functions with a resolved settings file or typed arguments."""
    repo_root = Path(plan["repo_root"])
    os.environ["HF_HOME"] = str(absolute(plan["launcher"]["model_cache"], repo_root))
    os.environ["XDG_CACHE_HOME"] = str(absolute(plan["launcher"]["cache_root"], repo_root))

    if plan["stage"]["name"] == "extract_mineru":
        run_mineru_job(plan)
        return

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
    elif name == "reduce_and_compact":
        from preprocessing.reduce_preprocessed_collection import main as reduce
        from preprocessing.compact_preprocessed_collection import main as compact
        from preprocessing.stage_preprocessed import copy_completed_papers
        dataset, stage = plan["dataset"], plan["stage"]
        repo_root = Path(plan["repo_root"])
        processed = absolute(dataset["processed_root"], repo_root)
        preprocessed = absolute(dataset["preprocessed"], repo_root)
        pdf_dir = absolute(dataset["pdf_dir"], repo_root)
        split_index = absolute(dataset["split_index"], repo_root) if dataset["split_index"] else None
        if stage["dry_run"]:
            work_dir = processed
        else:
            copied = copy_completed_papers(processed, preprocessed, split_index=split_index, split=split)
            print(f"Copied {copied} completed MinerU papers to {preprocessed}", flush=True)
            work_dir = preprocessed
        common = dict(preprocessed_dir=work_dir, split_index=split_index, split=split,
                      workers=int(stage["workers"]), dry_run=bool(stage["dry_run"]),
                      fail_fast=bool(stage["fail_fast"]), progress_every=int(stage["progress_every"]))
        reduce(argparse.Namespace(**common, report=preprocessed.parent / "preprocessed_analysis_report.json"))
        compact(argparse.Namespace(**common, root_dir=Path(plan["data_root"]), pdf_dir=pdf_dir))


def execute_worker(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    os.chdir(plan["repo_root"])
    if plan["stage"]["name"] in STAGE_LOG_NAMES:
        with stage_log(plan["run_dir"], plan["stage"]["name"]):
            execute_stage(plan)
    else:
        execute_stage(plan)
    return 0


class _Tee:
    """Write worker output to the run log and the original stream."""

    def __init__(self, original: Any, log_handle: Any) -> None:
        self.original = original
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.log_handle.write(value)
        self.log_handle.flush()
        return self.original.write(value)

    def flush(self) -> None:
        self.log_handle.flush()
        self.original.flush()

    def isatty(self) -> bool:
        return self.original.isatty()

    def fileno(self) -> int:
        """Expose the real stream descriptor to libraries using os.dup2."""
        return self.original.fileno()


@contextlib.contextmanager
def stage_log(run_dir: str | Path, stage_name: str):
    """Capture stage stdout/stderr in an immediately visible text file."""
    log_path = Path(run_dir) / STAGE_LOG_NAMES[stage_name]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] {stage_name} worker started\n"
        )
        handle.flush()
        stdout, stderr = sys.stdout, sys.stderr
        with contextlib.redirect_stdout(_Tee(stdout, handle)), contextlib.redirect_stderr(_Tee(stderr, handle)):
            print(f"Stage log: {log_path.resolve()}", flush=True)
            try:
                yield
            except BaseException:
                traceback.print_exc(file=handle)
                raise


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
    stage_name = plan["stage"]["name"]
    if stage_name in STAGE_LOG_NAMES:
        log_path = run_dir / STAGE_LOG_NAMES[stage_name]
        if not log_path.exists():
            log_path.write_text(
                f"[{datetime.now(timezone.utc).isoformat()}] {stage_name} job submitted\n",
                encoding="utf-8",
            )
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
    if stage_name in STAGE_LOG_NAMES:
        with stage_log(plan["run_dir"], stage_name):
            execute_stage(plan)
    else:
        execute_stage(plan)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(execute_worker(Path(sys.argv[2])))
    raise SystemExit("Use dataset-generation stage=<stage> launcher=<profile>")
