"""Hydra composition and the stage execution boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from hydra_run import build_plan, job_script, scheduler_command, mineru_caller_args, execute_stage, execute_worker, launch, stage_log


ROOT = Path(__file__).resolve().parents[1]


def config(*overrides: str) -> dict:
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


def test_stage_split_model_and_limit_compose() -> None:
    cfg = config("stage=generate_queries", "split=test", "model=noop", "stage.limit=3")
    plan, settings = build_plan(cfg, ROOT)
    assert plan["stage"]["name"] == "generate_queries"
    assert plan["stage"]["limit"] == 3
    assert plan["split"] == "test"
    assert settings["visual_query"]["model"]["provider"] == "null"
    assert settings["visual_query"]["model"]["name"] == "null"
    assert settings["visual_query"]["query_sets"]["test"]["split_name"] == "test"


def test_stage_default_models_and_explicit_override() -> None:
    assert config("stage=generate_queries")["model"]["name"] == "microsoft/phi-4"
    assert config("stage=describe_textual_clues")["model"]["name"] == "Qwen/Qwen3-4B"
    assert config("stage=judge_queries")["visual_query"]["judgement"]["model"] == "google/gemma-3-27b-it"
    assert config("stage=describe_figures", "model=qwen3_text")["model"]["name"] == "Qwen/Qwen3-4B"


def test_launcher_paths_override_dataset_defaults() -> None:
    for profile in ("local_gpu", "slurm_a100", "slurm_cpu", "pbs_rt_hg", "pbs_rt_hc"):
        cfg = config(f"launcher={profile}")
        assert cfg["dataset"]["root"] == cfg["launcher"]["dataset_root"]
        assert cfg["dataset"]["processed_root"] == cfg["launcher"]["processed_root"]
        assert cfg["dataset"]["pdf_dir"] == cfg["launcher"]["pdf_dir"]
        assert cfg["launcher"]["cache_root"]
    cfg = config("launcher=slurm_a100", "launcher.dataset_root=/tmp/corpus")
    assert cfg["dataset"]["root"] == "/tmp/corpus"
    assert cfg["dataset"]["processed_root"] == "/tmp/corpus/processed"
    assert cfg["dataset"]["pdf_dir"] == "/tmp/corpus/pdf_datasets"


def test_slurm_profiles_compose_submitit_resources() -> None:
    for profile, partition, gpus in (("slurm_a100", "gpu_a100", 1), ("slurm_cpu", "rome", 0)):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            composed = compose(config_name="config", overrides=[f"launcher={profile}"], return_hydra_config=True)
        launcher = OmegaConf.to_container(composed.hydra.launcher, resolve=False)
        assert launcher["_target_"].endswith(".SlurmLauncher")
        assert launcher["partition"] == partition
        assert launcher["gpus_per_node"] == gpus
        assert launcher["cpus_per_task"] == 16
        assert launcher["array_parallelism"] == 1000


def test_pbs_adapter_receives_resources() -> None:
    plan, _ = build_plan(config("launcher=pbs_rt_hg"), ROOT)
    submission = scheduler_command(plan, Path("/tmp/job.sh"), Path("/tmp/logs"))
    assert submission[0] == "qsub"
    assert "select=1:ncpus=16:mem=32gb:ngpus=1" in submission
    assert "-m hydra_run --worker" in job_script(Path("/tmp/plan.json"), plan)


def test_invalid_generation_split_rejected() -> None:
    with pytest.raises(ValueError, match="split=train or split=test"):
        build_plan(config("stage=generate_queries", "split=all"), ROOT)


def test_mineru_requires_gpu_slurm_and_describes_one_job() -> None:
    for profile in ("slurm_cpu", "local_gpu", "pbs_rt_hg"):
        with pytest.raises(ValueError, match="GPU Slurm launcher"):
            build_plan(config("stage=extract_mineru", f"launcher={profile}"), ROOT)
    plan, _ = build_plan(config("stage=extract_mineru", "launcher=slurm_a100"), ROOT)
    assert plan["execution"]["jobs"] == 1
    assert plan["execution"]["components"] == ["MinerU server", "extraction caller"]
    assert plan["execution"]["resources"] == {
        "partition": "gpu_a100", "gpus": 1, "cpus": 16, "memory": "32G", "walltime": "03:00:00"}
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        composed = compose(config_name="config", overrides=["stage=extract_mineru", "launcher=slurm_a100"],
                           return_hydra_config=True)
    assert "module load CUDA/12.6.0" in list(composed.hydra.launcher.setup)


def test_mineru_worker_propagates_overrides_to_one_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config("stage=extract_mineru", "launcher=slurm_a100", "split=train+test",
                 "stage.limit=20", "stage.max_in_flight=2", "stage.server_concurrency=4",
                 "stage.retries=5", "stage.retry_incomplete_only=true",
                 "stage.start_page_id=1", "stage.end_page_id=3",
                 "dataset.split_index=/tmp/custom-splits.json")
    plan, _ = build_plan(cfg, ROOT, tmp_path / "run")
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
    monkeypatch.setattr("hydra_run.subprocess.run", fake_run)
    monkeypatch.setattr("hydra_run.socket.socket", lambda *args: FakeSocket())
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "slurm-tmp"))
    execute_stage(plan)
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd[0].endswith("run_mineru_full_extraction_gpu.sh")
    assert cmd[1:3] == ["--no-default-caller-args", "--"]
    caller = cmd[3:]
    for option, value in (("--split", "train+test"), ("--split-index", str(Path("/tmp/custom-splits.json").resolve())),
                          ("--limit", "20"), ("--max-in-flight", "2"), ("--retries", "5"),
                          ("--start-page-id", "1"), ("--end-page-id", "3")):
        assert caller[caller.index(option) + 1] == value
    assert "--retry-incomplete-only" in caller
    assert caller[caller.index("--api-url") + 1] == "http://127.0.0.1:43123"
    assert kwargs["env"]["MINERU_API_MAX_CONCURRENT_REQUESTS"] == "4"
    assert kwargs["env"]["TMPDIR"].endswith("/tmp")
    assert kwargs["check"] is True


class FakeSocket:
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def bind(self, address): pass
    def getsockname(self): return ("127.0.0.1", 43123)


def test_mineru_dry_run_does_not_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = config("stage=extract_mineru", "launcher=slurm_a100", "dry_run=true")
    monkeypatch.setattr("hydra_run.subprocess.run", lambda *args, **kwargs: pytest.fail("submitted"))
    launch(cfg, ROOT, hydra_run_dir=tmp_path / "dry")
    output = json.loads(capsys.readouterr().out)
    assert output["submission"]["hydra_launcher"] == "submitit_slurm"
    assert output["plan"]["execution"]["jobs"] == 1
    assert not (tmp_path / "dry").exists()


def test_mineru_submitit_worker_executes_once_in_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config("stage=extract_mineru", "launcher=slurm_a100", "split=train")
    executed = []
    monkeypatch.setattr("hydra_run.execute_stage", lambda plan: executed.append(plan))
    monkeypatch.setattr("hydra_run.subprocess.run", lambda *args, **kwargs: pytest.fail("nested submission"))
    run_dir = tmp_path / "submitit-job"
    launch(cfg, ROOT, native_slurm=True, hydra_run_dir=run_dir)
    assert len(executed) == 1
    assert executed[0]["execution"]["jobs"] == 1
    assert json.loads((run_dir / "plan.json").read_text())["split"] == "train"


@pytest.mark.parametrize("stage, log_name", [
    ("reduce_and_compact", "preprocessing.log"),
    ("describe_figures", "describe_figures.log"),
    ("describe_textual_clues", "describe_textual_clues.log"),
    ("generate_queries", "generate_queries.log"),
    ("judge_queries", "judge_queries.log"),
])
def test_local_stage_captures_stdout_and_stderr(
    stage: str, log_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    def fake_execute(plan):
        print("stage output")
        print("stage warning", file=sys.stderr)

    monkeypatch.setattr("hydra_run.execute_stage", fake_execute)
    run_dir = tmp_path / stage
    launch(config(f"stage={stage}"), ROOT, hydra_run_dir=run_dir)
    log = (run_dir / log_name).read_text()
    assert f"{stage} job submitted" in log
    assert f"{stage} worker started" in log
    assert "stage output" in log
    assert "stage warning" in log
    captured = capsys.readouterr()
    assert "stage output" in captured.out
    assert "stage warning" in captured.err


def test_worker_stage_captures_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "worker"
    plan, _ = build_plan(config("stage=generate_queries"), ROOT, run_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr("hydra_run.execute_stage", lambda plan: print("worker output"))
    assert execute_worker(plan_path) == 0
    log = (run_dir / "generate_queries.log").read_text()
    assert "generate_queries worker started" in log
    assert "worker output" in log


def test_stage_failure_is_recorded_in_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(plan):
        raise RuntimeError("generation failed")

    monkeypatch.setattr("hydra_run.execute_stage", fail)
    run_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="generation failed"):
        launch(config("stage=generate_queries"), ROOT, hydra_run_dir=run_dir)
    log = (run_dir / "generate_queries.log").read_text()
    assert "RuntimeError: generation failed" in log


def test_stage_log_is_written_before_stage_finishes(tmp_path: Path) -> None:
    log_path = tmp_path / "generate_queries.log"
    with stage_log(tmp_path, "generate_queries"):
        print("first query complete")
        assert "first query complete" in log_path.read_text()


def test_wrappers_forward_hydra_overrides(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.json"
    stub = bin_dir / "uv"
    stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CAPTURE": str(capture)}
    wrappers = {
        "scripts/clue_generation/describe_all_figures.sh": "describe_figures",
        "scripts/clue_generation/describe_all_textual_clues.sh": "describe_textual_clues",
        "scripts/query_generation/generate_queries.sh": "generate_queries",
        "scripts/query_generation/judge_train_queries.sh": "judge_queries",
        "scripts/preprocessing/reduce_and_compact_preprocessed.sh": "reduce_and_compact",
    }
    for wrapper, stage in wrappers.items():
        subprocess.run([str(ROOT / wrapper), "split=test", "stage.limit=3"], check=True, env=env)
        assert json.loads(capture.read_text()) == [
            "run", "--no-sync", "dataset-generation", f"stage={stage}", "split=test", "stage.limit=3",
        ]
    subprocess.run([str(ROOT / "scripts/extraction/run_mineru_full_extraction.sh"),
                    "-m", "stage=extract_mineru", "split=train", "launcher=slurm_a100"],
                   check=True, env=env)
    assert json.loads(capture.read_text()) == [
        "run", "--no-sync", "dataset-generation", "-m", "stage=extract_mineru",
        "split=train", "launcher=slurm_a100", "stage=extract_mineru"]
    subprocess.run([str(ROOT / "scripts/extraction/run_mineru_full_extraction.sh"),
                    "--split", "train", "--limit", "3"], check=True, env=env)
    assert json.loads(capture.read_text()) == [
        "run", "--no-sync", "dataset-generation", "extract-mineru-pdfs",
        "--input-dir", "data/pdf_datasets", "--output-dir", "data/preprocessed",
        "--split", "train", "--limit", "3"]
