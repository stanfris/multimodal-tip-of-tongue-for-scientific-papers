"""Low-overhead, best-effort telemetry for one allocated MinerU GPU job."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import signal
import shutil
import statistics
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


GPU_FIELDS = (
    "timestamp", "index", "utilization.gpu", "utilization.memory", "memory.used",
    "memory.total", "power.draw", "temperature.gpu", "clocks.sm", "clocks.mem",
)
THREAD_VARS = (
    "MINERU_INTRA_OP_NUM_THREADS", "MINERU_INTER_OP_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command(args: list[str], *, timeout: float = 5) -> str:
    try:
        return (subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=True).stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def write_system_info(run_dir: Path, paths: dict[str, Path], env: dict[str, str]) -> None:
    """Capture allocation, affinity and storage mounts before launching MinerU."""
    report = [f"timestamp_utc: {_utc()}", f"hostname: {_command(['hostname'])}",
              f"pwd: {Path.cwd()}", f"nproc: {_command(['nproc'])}",
              f"PBS_JOBID: {env.get('PBS_JOBID', '')}",
              f"PBS_NODEFILE: {env.get('PBS_NODEFILE', '')}",
              f"SLURM_JOB_ID: {env.get('SLURM_JOB_ID', '')}",
              f"TMPDIR: {env.get('TMPDIR', '')}",
              f"MINERU_PARENT_TMPDIR: {env.get('MINERU_PARENT_TMPDIR', '')}",
              f"SLURM_TMPDIR: {env.get('SLURM_TMPDIR', '')}"]
    for key in THREAD_VARS:
        report.append(f"{key}: {env.get(key, '')}")
    report.extend([f"taskset -pc {os.getpid()}: {_command(['taskset', '-pc', str(os.getpid())])}"])
    try:
        report.append("Cpus_allowed_list: " + next(line.strip().split(':', 1)[1].strip()
                      for line in Path(f"/proc/{os.getpid()}/status").read_text().splitlines()
                      if line.startswith("Cpus_allowed_list:")))
    except (OSError, StopIteration):
        report.append("Cpus_allowed_list: unavailable")
    report.append("lscpu:\n" + _command(["lscpu"]))
    for label, path in paths.items():
        report.append(f"{label}: {path}")
        report.append(f"df -h {path}:\n{_command(['df', '-h', str(path)])}")
        report.append(f"df -T {path}:\n{_command(['df', '-T', str(path)])}")
    (run_dir / "system_info.log").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_proc_stat(content: str, allowed: set[int]) -> dict[str, tuple[int, int, int, int, int]]:
    """Return total, idle, user, system and iowait ticks for allocated CPUs."""
    result = {}
    for line in content.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu") or not parts[0][3:].isdigit():
            continue
        index = int(parts[0][3:])
        if index not in allowed or len(parts) < 6:
            continue
        values = [int(value) for value in parts[1:]]
        result[parts[0]] = (sum(values), values[3] + values[4], values[0] + values[1],
                            values[2] + (values[5] if len(values) > 5 else 0), values[4])
    return result


def cpu_samples(previous: dict, current: dict, timestamp: str) -> list[dict]:
    rows = []
    for cpu in sorted(previous.keys() & current.keys(), key=lambda name: int(name[3:])):
        delta = [b - a for a, b in zip(previous[cpu], current[cpu])]
        total = delta[0]
        if total <= 0:
            continue
        rows.append({"timestamp": timestamp, "cpu": cpu, "utilization_pct": round(100 * (1 - delta[1] / total), 2),
                     "idle_pct": round(100 * delta[1] / total, 2),
                     "user_pct": round(100 * delta[2] / total, 2),
                     "system_pct": round(100 * delta[3] / total, 2),
                     "iowait_pct": round(100 * delta[4] / total, 2)})
    if rows:
        rows.append({key: (sum(row[key] for row in rows) / len(rows) if key.endswith("_pct") else
                           "all" if key == "cpu" else timestamp) for key in rows[0]})
    return rows


def mpstat_samples(output: str, allowed: set[int], timestamp: str) -> list[dict]:
    """Read sysstat JSON when available; /proc/stat remains the fallback."""
    try:
        loads = json.loads(output)["sysstat"]["hosts"][0]["statistics"][-1]["cpu-load"]
        rows = []
        for load in loads:
            name = str(load["cpu"])
            if not name.isdigit() or int(name) not in allowed:
                continue
            idle = float(load["%idle"])
            rows.append({"timestamp": timestamp, "cpu": f"cpu{name}",
                         "utilization_pct": round(100 - idle, 2), "idle_pct": idle,
                         "user_pct": float(load["%usr"]), "system_pct": float(load["%sys"]),
                         "iowait_pct": float(load["%iowait"])})
        if rows:
            rows.append({key: (round(sum(row[key] for row in rows) / len(rows), 2) if key.endswith("_pct") else
                               "all" if key == "cpu" else timestamp) for key in rows[0]})
        return rows
    except (KeyError, IndexError, TypeError, ValueError):
        return []


def parse_gpu_rows(output: str) -> list[dict[str, str]]:
    return [dict(zip(GPU_FIELDS, [item.strip() for item in row])) for row in csv.reader(output.splitlines())
            if len(row) == len(GPU_FIELDS)]


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _sample_processes(run_dir: Path, handle) -> None:
    try:
        load = os.getloadavg()
    except OSError:
        load = (None, None, None)
    try:
        runnable = Path("/proc/loadavg").read_text().split()[3]
    except (OSError, IndexError):
        runnable = None
    for role in ("server", "caller"):
        pid = _read_pid(run_dir / f"mineru.{role}.pid")
        if pid is None:
            continue
        ps = _command(["ps", "-p", str(pid), "-o", "pid=,%cpu=,%mem=,rss=,nlwp=,comm="])
        try:
            io = {key: int(value) for key, value in (line.split(": ", 1) for line in
                  Path(f"/proc/{pid}/io").read_text().splitlines()) if key in {"read_bytes", "write_bytes"}}
        except (OSError, ValueError):
            io = {}
        handle.write(json.dumps({"timestamp": _utc(), "role": role, "pid": pid,
                                 "ps": ps, "load_average": load, "runnable_tasks": runnable,
                                 "io_bytes": io}) + "\n")


def _monitor_loop(run_dir: Path, stop: threading.Event, interval: float, env: dict[str, str]) -> None:
    gpu_available = shutil.which("nvidia-smi") is not None
    if not gpu_available:
        print("Warning: nvidia-smi unavailable; GPU telemetry disabled", flush=True)
    mpstat_available = shutil.which("mpstat") is not None
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    try:
        previous = parse_proc_stat(Path("/proc/stat").read_text(), allowed)
    except OSError:
        previous = {}
    with (run_dir / "gpu_usage.csv").open("w", newline="", encoding="utf-8") as gpu_file, \
         (run_dir / "cpu_usage.log").open("w", encoding="utf-8") as cpu_file, \
         (run_dir / "process_usage.log").open("w", encoding="utf-8") as process_file:
        writer = csv.DictWriter(gpu_file, fieldnames=GPU_FIELDS)
        writer.writeheader()
        tick = 0
        while not stop.is_set():
            if gpu_available:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                         "--format=csv,noheader,nounits"], text=True, capture_output=True,
                        timeout=3, check=True)
                    visible = env.get("CUDA_VISIBLE_DEVICES", "")
                    indices = {item.strip() for item in visible.split(",") if item.strip().isdigit()}
                    writer.writerows(row for row in parse_gpu_rows(result.stdout or "")
                                     if not visible or row["index"] in indices)
                    gpu_file.flush()
                except (OSError, subprocess.SubprocessError, AttributeError) as exc:
                    print(f"Warning: GPU telemetry sample failed: {exc}", flush=True)
                    gpu_available = False
            rows = []
            if mpstat_available:
                try:
                    result = subprocess.run(["mpstat", "-o", "JSON", "-P", "ALL", "1", "1"],
                                            text=True, capture_output=True, timeout=3, check=True)
                    rows = mpstat_samples(result.stdout or "", allowed, _utc())
                    if not rows:
                        mpstat_available = False
                except (OSError, subprocess.SubprocessError, AttributeError):
                    mpstat_available = False
            try:
                current = parse_proc_stat(Path("/proc/stat").read_text(), allowed)
                if not rows:
                    rows = cpu_samples(previous, current, _utc())
                previous = current
            except OSError:
                pass
            for row in rows:
                cpu_file.write(json.dumps(row) + "\n")
            cpu_file.flush()
            if tick % max(round(15 / interval), 1) == 0:
                _sample_processes(run_dir, process_file)
                process_file.flush()
            tick += 1
            stop.wait(interval)


@contextlib.contextmanager
def monitor_job(run_dir: Path, paths: dict[str, Path], env: dict[str, str], interval: float = 3) -> Iterator[None]:
    """Start before server launch and stop on normal exit, exception or signal unwind."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for role in ("server", "caller"):
        (run_dir / f"mineru.{role}.pid").unlink(missing_ok=True)
    try:
        write_system_info(run_dir, paths, env)
    except Exception as exc:
        print(f"Warning: system telemetry unavailable: {exc}", flush=True)
    stop = threading.Event()
    thread = threading.Thread(target=_monitor_loop, args=(run_dir, stop, interval, env), daemon=True)
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        def stop_on_term(signum, frame) -> None:
            raise SystemExit(128 + signum)

        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop_on_term)
    started = False
    try:
        thread.start()
        started = True
    except RuntimeError as exc:
        print(f"Warning: MinerU telemetry monitor unavailable: {exc}", flush=True)
    try:
        yield
    finally:
        stop.set()
        if started:
            thread.join(timeout=6)
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


def _float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics(values: list[float]) -> tuple[float | None, float | None]:
    return (round(statistics.mean(values), 2), round(max(values), 2)) if values else (None, None)


def summarize_telemetry(run_dir: Path, extraction_summary: Path, *, started_at: float | None = None) -> dict:
    """Summarize persisted samples; missing or malformed telemetry is nonfatal."""
    gpu_rows = []
    cpu_rows = []
    try:
        with (run_dir / "gpu_usage.csv").open(newline="", encoding="utf-8") as handle:
            gpu_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        pass
    try:
        cpu_rows = [json.loads(line) for line in (run_dir / "cpu_usage.log").read_text().splitlines() if line]
    except (OSError, ValueError):
        pass
    gpu_util = [value for row in gpu_rows if (value := _float(row.get("utilization.gpu"))) is not None]
    gpu_memory = [value for row in gpu_rows if (value := _float(row.get("memory.used"))) is not None]
    gpu_power = [value for row in gpu_rows if (value := _float(row.get("power.draw"))) is not None]
    aggregate = [row for row in cpu_rows if row.get("cpu") == "all"]
    per_cpu = [row for row in cpu_rows if row.get("cpu") != "all"]
    active_by_time = {}
    for row in per_cpu:
        active_by_time.setdefault(row.get("timestamp"), 0)
        if (_float(row.get("utilization_pct")) or 0) > 50:
            active_by_time[row.get("timestamp")] += 1
    aggregate_util = [value for row in aggregate if (value := _float(row.get("utilization_pct"))) is not None]
    iowait = [value for row in aggregate if (value := _float(row.get("iowait_pct"))) is not None]
    extraction = None
    try:
        if started_at is None or extraction_summary.stat().st_mtime >= started_at:
            extraction = json.loads(extraction_summary.read_text())["stats"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result = {
        "gpu": {"samples": len(gpu_util), "average_utilization_pct": _metrics(gpu_util)[0],
                "median_utilization_pct": round(statistics.median(gpu_util), 2) if gpu_util else None,
                "p95_utilization_pct": sorted(gpu_util)[math.ceil(0.95 * len(gpu_util)) - 1] if gpu_util else None,
                "percent_samples_above_80_pct": round(100 * sum(value > 80 for value in gpu_util) / len(gpu_util), 2) if gpu_util else None,
                "average_memory_used_mib": _metrics(gpu_memory)[0], "peak_memory_used_mib": _metrics(gpu_memory)[1],
                "average_power_w": _metrics(gpu_power)[0]},
        "cpu": {"samples": len(aggregate_util), "average_aggregate_utilization_pct": _metrics(aggregate_util)[0],
                "maximum_aggregate_utilization_pct": _metrics(aggregate_util)[1],
                "average_active_cores_above_50_pct": _metrics(list(active_by_time.values()))[0],
                "average_iowait_pct": _metrics(iowait)[0]},
        "extraction": {key: extraction.get(key) if extraction else None for key in
                       ("pages_per_second", "papers_per_second", "completed", "failed", "elapsed_seconds")},
    }
    (run_dir / "performance_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
