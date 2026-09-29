"""Check the scheduler boundary without submitting jobs or loading models."""

from pathlib import Path
import json
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from common.managed_settings import DEFAULT_SETTINGS_PATH, load_managed_settings
from hydra_run import build_plan, execute_worker, job_script, launch, scheduler_command, worker_command


ROOT = Path(__file__).resolve().parents[1]


def config(profile: str, task: str = "describe_figures") -> dict:
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        composed = compose(config_name="config", overrides=[
            f"launcher={profile}", f"task={task}",
            f"launcher.base_dir={ROOT / 'data'}",
            f"launcher.run_dir={ROOT / 'runs'}",
            f"launcher.python={ROOT / '.venv/bin/python'}",
        ])
    return OmegaConf.to_container(composed, resolve=True)


def hydra_launcher(profile: str) -> dict:
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        composed = compose(config_name="config", overrides=[f"launcher={profile}"], return_hydra_config=True)
    return OmegaConf.to_container(composed.hydra.launcher, resolve=False)


def test_switching_profiles_resolves_paths_and_device_settings() -> None:
    for profile in ("local_gpu", "slurm_a100", "slurm_cpu", "pbs_rt_hg", "pbs_rt_hc"):
        plan, settings = build_plan(config(profile), ROOT)
        assert settings is not None
        assert settings["dataset"]["root"] == str(ROOT / "data")
        assert Path(settings["visual_descriptions"]["prompt"]["template"]).is_absolute()
        assert settings["visual_descriptions"]["generation"]["device_map"] == plan["launcher"]["device_map"]
        assert settings["textual_descriptions"]["generation"]["dtype"] == plan["launcher"]["dtype"]
        assert settings["visual_query"]["model"]["attn_implementation"] == plan["launcher"]["attn_implementation"]
        assert settings["visual_query"]["judgement"]["device_map"] == plan["launcher"]["device_map"]
        plan["settings_snapshot"] = "/tmp/settings.yaml"
        assert worker_command(plan)[3:5] == ["describe-figures", "--settings"]


def test_scheduler_requests_match_profiles() -> None:
    for profile in ("pbs_rt_hg", "pbs_rt_hc"):
        plan, _ = build_plan(config(profile), ROOT)
        command = scheduler_command(plan, Path("/tmp/job.sh"), Path("/tmp/logs"))
        joined = " ".join(command)
        assert ("ngpus=1" in joined) == bool(plan["launcher"]["gpus"])
        assert command[0] == "qsub"
        assert plan["launcher"]["queue"] in command
        assert "-m hydra_run --worker" in job_script(Path("/tmp/plan.json"), plan)


def test_slurm_profiles_use_native_submitit_launcher() -> None:
    for profile, gpus, partition in (("slurm_a100", 1, "gpu_a100"), ("slurm_cpu", 0, "rome")):
        launcher = hydra_launcher(profile)
        application = config(profile)["launcher"]
        assert launcher["_target_"].endswith(".SlurmLauncher")
        assert launcher["gpus_per_node"] == application["gpus"] == gpus
        assert launcher["cpus_per_task"] == application["cpus"] == 16
        assert launcher["partition"] == application["partition"] == partition
        assert launcher["array_parallelism"] == 1000


def test_slurm_requires_multirun_for_submission() -> None:
    cfg = config("slurm_a100")
    cfg["dry_run"] = False
    with pytest.raises(ValueError, match="-m"):
        launch(cfg, ROOT, native_slurm=False)


def test_setup_is_only_in_slurm_launcher_configs() -> None:
    for profile in ("local_gpu", "slurm_a100", "slurm_cpu", "pbs_rt_hg", "pbs_rt_hc"):
        hydra_config = hydra_launcher(profile)
        plan, _ = build_plan(config(profile), ROOT)
        script = job_script(Path("/tmp/plan.json"), plan)
        if profile.startswith("slurm"):
            setup = hydra_config["setup"]
            assert setup[-1] == "source scripts/environment/activate_env.sh main"
            if profile == "slurm_a100":
                assert setup[:3] == ["module purge", "module load 2023", "module load CUDA/12.4.0"]
        else:
            assert hydra_config.get("setup") is None
        assert "module " not in script
        assert "source scripts/environment/activate_env.sh" not in script
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_slurm_setup_activates_venv_for_python_imports() -> None:
    for profile in ("slurm_a100", "slurm_cpu"):
        commands = "\n".join(hydra_launcher(profile)["setup"])
        script = (
            "set -e\nmodule() { :; }\n"
            f"cd {shlex.quote(str(ROOT))}\n{commands}\n"
            "python -c 'import hydra, yaml, os, sys; "
            "assert os.environ[\"VIRTUAL_ENV\"] == sys.prefix'"
        )
        result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_preprocess_uses_profile_root_and_cpus() -> None:
    plan, snapshot = build_plan(config("slurm_cpu", "preprocess"), ROOT)
    assert snapshot is None
    command = worker_command(plan)
    assert command[1:5] == ["--root-dir", str(ROOT / "data"), "--workers", "16"]


def test_legacy_cli_reads_hydra_defaults() -> None:
    settings, path = load_managed_settings(DEFAULT_SETTINGS_PATH)
    assert path == ROOT / DEFAULT_SETTINGS_PATH
    assert settings["dataset"]["root"] == str(ROOT / "data")
    assert settings["visual_query"]["query_sets"]["train"]["collection_id"] == "query_generation_train"


def test_judgement_task_receives_hydra_snapshot() -> None:
    plan, settings = build_plan(config("pbs_rt_hg", "judge_queries"), ROOT)
    assert settings is not None
    plan["settings_snapshot"] = "/tmp/settings.yaml"
    assert worker_command(plan)[1:3] == ["-m", "queries.judge_queries"]


def test_hydra_log_directory_follows_script_folder() -> None:
    folders = {
        "describe_figures": "clue_generation",
        "describe_textual_clues": "clue_generation",
        "generate_queries": "query_generation",
        "judge_queries": "query_generation",
        "preprocess": "preprocessing",
    }
    for task, folder in folders.items():
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
            composed = compose(config_name="config", overrides=[f"task={task}"], return_hydra_config=True)
        for mode in ("run", "sweep"):
            path = str(composed.hydra[mode].dir)
            assert f"/{folder}/{task}/" in path


def test_worker_output_is_written_under_log_root(tmp_path, monkeypatch) -> None:
    plan, _ = build_plan(config("local_gpu", "preprocess"), ROOT)
    plan["log_dir"] = str(tmp_path / "logs" / "preprocessing" / "preprocess" / "run")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr("hydra_run.worker_command", lambda _: [sys.executable, "-c", "print('worker output')"])
    assert execute_worker(plan_path) == 0
    assert (Path(plan["log_dir"]) / "worker.log").read_text().strip() == "worker output"
