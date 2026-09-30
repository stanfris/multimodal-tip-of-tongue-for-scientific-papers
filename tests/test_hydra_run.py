"""Hydra composition and the stage execution boundary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from hydra_run import build_plan, job_script, scheduler_command


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
