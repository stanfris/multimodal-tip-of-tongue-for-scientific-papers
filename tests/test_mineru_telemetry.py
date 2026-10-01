"""Synthetic telemetry coverage without a GPU or sysstat installation."""

import json
import signal
import threading
import time
from pathlib import Path

from extraction import mineru_telemetry as telemetry


def test_gpu_cpu_parsing_and_permitted_core_summary(tmp_path: Path) -> None:
    gpu = tmp_path / "gpu_usage.csv"
    gpu.write_text(
        ",".join(telemetry.GPU_FIELDS) + "\n"
        "2026/10/01 01:00:00,0,90,45,1000,80000,300,55,1500,1200\n"
        "2026/10/01 01:00:03,0,70,35,2000,80000,200,53,1450,1150\n"
    )
    before = telemetry.parse_proc_stat(
        "cpu 1 2 3 4 5 6\ncpu0 10 0 10 80 0 0\ncpu1 10 0 10 80 0 0\n"
        "cpu2 10 0 10 80 0 0\n", {0, 1})
    after = telemetry.parse_proc_stat(
        "cpu0 70 0 20 100 10 0\ncpu1 20 0 20 160 0 0\n"
        "cpu2 70 0 20 100 10 0\n", {0, 1})
    rows = telemetry.cpu_samples(before, after, "2026-10-01T01:00:03Z")
    assert len(rows) == 3
    assert rows[0]["cpu"] == "cpu0"
    assert rows[1]["cpu"] == "cpu1"
    assert rows[2]["cpu"] == "all"
    assert rows[0]["utilization_pct"] > 50
    assert rows[1]["utilization_pct"] < 50
    (tmp_path / "cpu_usage.log").write_text("".join(json.dumps(row) + "\n" for row in rows))
    extraction = tmp_path / "last_run_summary.json"
    extraction.write_text(json.dumps({"stats": {"completed": 10, "failed": 1,
        "pages_per_second": 5.2, "papers_per_second": 0.7, "elapsed_seconds": 14.2}}))
    summary = telemetry.summarize_telemetry(tmp_path, extraction)
    assert summary["gpu"]["average_utilization_pct"] == 80
    assert summary["gpu"]["peak_memory_used_mib"] == 2000
    assert summary["gpu"]["percent_samples_above_80_pct"] == 50
    assert summary["cpu"]["average_active_cores_above_50_pct"] == 1
    assert summary["extraction"]["completed"] == 10
    assert json.loads((tmp_path / "performance_summary.json").read_text()) == summary


def test_mpstat_json_filters_to_allocated_cpus() -> None:
    output = json.dumps({"sysstat": {"hosts": [{"statistics": [{"cpu-load": [
        {"cpu": "0", "%idle": 10, "%usr": 60, "%sys": 20, "%iowait": 10},
        {"cpu": "1", "%idle": 90, "%usr": 5, "%sys": 5, "%iowait": 0},
        {"cpu": "2", "%idle": 0, "%usr": 100, "%sys": 0, "%iowait": 0},
    ]}]}]}})
    rows = telemetry.mpstat_samples(output, {0, 1}, "timestamp")
    assert [row["cpu"] for row in rows] == ["cpu0", "cpu1", "all"]
    assert rows[-1]["utilization_pct"] == 50


def test_monitor_starts_in_run_dir_and_cleans_up_without_nvidia_or_mpstat(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(telemetry, "write_system_info", lambda run_dir, paths, env:
                        (run_dir / "system_info.log").write_text("allocation=16\n"))
    monkeypatch.setattr(telemetry.shutil, "which", lambda command: None)
    run_dir = tmp_path / "hydra-run"
    with telemetry.monitor_job(run_dir, {}, {}, interval=0.02):
        time.sleep(0.07)
        assert (run_dir / "gpu_usage.csv").exists()
        assert (run_dir / "cpu_usage.log").exists()
        assert (run_dir / "process_usage.log").exists()
    assert (run_dir / "system_info.log").exists()
    if Path("/proc/stat").exists():
        assert "\"cpu\": \"all\"" in (run_dir / "cpu_usage.log").read_text()


def test_monitor_cleanup_after_failure(tmp_path: Path, monkeypatch) -> None:
    stopped = threading.Event()
    monkeypatch.setattr(telemetry, "write_system_info", lambda *args: None)
    monkeypatch.setattr(telemetry, "_monitor_loop", lambda run_dir, stop, interval, env:
                        (stop.wait(), stopped.set()))
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with telemetry.monitor_job(tmp_path, {}, {}, interval=0.01):
            assert signal.getsignal(signal.SIGTERM) is not previous
            raise RuntimeError("extraction failed")
    except RuntimeError:
        pass
    assert stopped.is_set()
    assert signal.getsignal(signal.SIGTERM) is previous


def test_summary_handles_missing_telemetry_and_extraction(tmp_path: Path) -> None:
    summary = telemetry.summarize_telemetry(tmp_path, tmp_path / "missing.json")
    assert summary["gpu"]["samples"] == 0
    assert summary["cpu"]["samples"] == 0
    assert summary["extraction"]["completed"] is None
