"""Hydra composition and the stage execution boundary."""

from __future__ import annotations

import json
import contextlib
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from hydra_run import build_plan, job_script, scheduler_command, mineru_caller_args, normalize_mineru_cuda_visibility, execute_stage, execute_worker, launch, stage_log


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
    assert config("stage=generate_queries")["visual_query"]["selection"]["modes"] == ["visual-and-text"]
    assert config("stage=describe_textual_clues")["model"]["name"] == "Qwen/Qwen3-4B"
    assert config("stage=judge_queries")["visual_query"]["judgement"]["model"] == "google/gemma-3-27b-it"
    assert config("stage=describe_figures", "model=qwen3_text")["model"]["name"] == "Qwen/Qwen3-4B"


def test_launcher_paths_override_dataset_defaults() -> None:
    for profile in ("local_gpu", "slurm_a100", "slurm_cpu", "pbs_h200", "pbs_rt_hg", "pbs_rt_hc"):
        cfg = config(f"launcher={profile}")
        assert cfg["dataset"]["root"] == cfg["launcher"]["dataset_root"]
        assert cfg["dataset"]["processed_root"] == cfg["launcher"]["processed_root"]
        assert cfg["dataset"]["pdf_dir"] == cfg["launcher"]["pdf_dir"]
        assert cfg["dataset"]["split_index"] == f"{cfg['dataset']['root']}/splits/pdf_dataset_split.json"
        assert cfg["launcher"]["cache_root"]
    cfg = config("launcher=slurm_a100", "launcher.dataset_root=/tmp/corpus")
    assert cfg["dataset"]["root"] == "/tmp/corpus"
    assert cfg["dataset"]["processed_root"] == "/tmp/corpus/processed"
    assert cfg["dataset"]["preprocessed"] == "/tmp/corpus/preprocessed"
    assert cfg["dataset"]["pdf_dir"] == "/tmp/corpus/pdf_datasets"
    assert cfg["dataset"]["split_index"] == "/tmp/corpus/splits/pdf_dataset_split.json"
    plan, settings = build_plan(config("launcher=slurm_a100", "stage=describe_textual_clues",
                                       "launcher.dataset_root=/scratch-shared/sfris1"), ROOT)
    assert plan["dataset"]["processed_root"] == "/scratch-shared/sfris1/processed"
    assert settings["dataset"]["preprocessed"] == "/scratch-shared/sfris1/preprocessed"
    assert settings["dataset"]["split_index"] == "/scratch-shared/sfris1/splits/pdf_dataset_split.json"


def test_launchers_default_to_repository_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VTT_DATA_ROOT", raising=False)
    for profile in ("local_gpu", "slurm_a100", "slurm_cpu", "pbs_h200", "pbs_rt_hg", "pbs_rt_hc"):
        plan, settings = build_plan(config(f"launcher={profile}"), tmp_path)
        assert plan["data_root"] == str(tmp_path / "data")
        assert settings["dataset"]["preprocessed"] == str(tmp_path / "data/preprocessed")
        assert settings["dataset"]["split_index"] == str(tmp_path / "data/splits/pdf_dataset_split.json")


@pytest.mark.parametrize("profile", ["local_gpu", "slurm_a100", "pbs_h200"])
def test_launcher_data_root_environment_override_moves_all_mineru_paths(
    profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    monkeypatch.setenv("VTT_DATA_ROOT", str(corpus))
    cfg = config("stage=extract_mineru", f"launcher={profile}")
    plan, _ = build_plan(cfg, ROOT)
    assert plan["data_root"] == str(corpus)
    assert plan["dataset"]["pdf_dir"] == str(corpus / "pdf_datasets")
    assert plan["dataset"]["processed_root"] == str(corpus / "processed")
    assert plan["dataset"]["split_index"] == str(corpus / "splits/pdf_dataset_split.json")
    args = mineru_caller_args(plan, "http://127.0.0.1:8002")
    assert args[args.index("--input-dir") + 1] == str(corpus / "pdf_datasets")
    assert args[args.index("--output-dir") + 1] == str(corpus / "processed")
    assert args[args.index("--split-index") + 1] == str(corpus / "splits/pdf_dataset_split.json")


@pytest.mark.parametrize("profile", ["local_gpu", "slurm_a100", "pbs_h200"])
def test_mineru_full_scan_uses_launcher_paths_without_split_index(profile: str, tmp_path: Path) -> None:
    cfg = config("stage=extract_mineru", f"launcher={profile}", "split=train+test",
                 "stage.all_domain_pdfs=true", "stage.domains=[ACL]",
                 f"launcher.dataset_root={tmp_path / 'corpus'}")
    plan, _ = build_plan(cfg, ROOT)
    args = mineru_caller_args(plan, "http://127.0.0.1:8002")
    assert args[args.index("--input-dir") + 1] == str(tmp_path / "corpus/pdf_datasets")
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "corpus/processed")
    assert "--split-index" not in args
    assert "--all-domain-pdfs" in args
    assert args[args.index("--domains") + 1] == "ACL"


@pytest.mark.parametrize("profile", ["local_gpu", "slurm_a100", "pbs_h200"])
def test_mineru_indexed_scan_uses_launchers_index_and_output(profile: str, tmp_path: Path) -> None:
    cfg = config("stage=extract_mineru", f"launcher={profile}", "split=train",
                 f"launcher.dataset_root={tmp_path / 'corpus'}")
    plan, _ = build_plan(cfg, ROOT)
    args = mineru_caller_args(plan, "http://127.0.0.1:8002")
    assert args[args.index("--split-index") + 1] == str(tmp_path / "corpus/splits/pdf_dataset_split.json")
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "corpus/processed")
    assert "--all-domain-pdfs" not in args


def test_managed_settings_resolves_relative_split_override_once(tmp_path: Path) -> None:
    cfg = config("launcher=local_gpu", "dataset.split_index=data/splits/pdf_dataset_split.json")
    _, settings = build_plan(cfg, tmp_path)
    assert settings["dataset"]["split_index"] == str(tmp_path / "data/splits/pdf_dataset_split.json")


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
    assert f"select=1:ncpus=16:mem={plan['launcher']['memory']}:ngpus=1" in submission
    assert "-m hydra_run --worker" in job_script(Path("/tmp/plan.json"), plan)


def test_pbs_job_preserves_venv_python_symlink(tmp_path: Path) -> None:
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    plan, _ = build_plan(config("launcher=pbs_rt_hg"), tmp_path)
    script = job_script(tmp_path / "plan.json", plan)
    assert f"exec {venv_python} -m hydra_run --worker" in script
    assert f"exec {Path(sys.executable).resolve()} -m hydra_run --worker" not in script


def test_h200_job_disables_flashinfer_sampling_without_nvcc() -> None:
    plan, _ = build_plan(config("launcher=pbs_h200", "model.provider=vllm"), ROOT)
    script = job_script(Path("/tmp/plan.json"), plan)
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in script
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in script


def test_invalid_generation_split_rejected() -> None:
    with pytest.raises(ValueError, match="split=train or split=test"):
        build_plan(config("stage=generate_queries", "split=all"), ROOT)


def test_other_split_is_available_for_mineru() -> None:
    plan, _ = build_plan(config("stage=extract_mineru", "split=other", "launcher=slurm_a100"), ROOT)
    args = mineru_caller_args(plan, "http://127.0.0.1:8002")
    assert args[args.index("--split") + 1] == "other"
    assert "--split-index" in args
    with pytest.raises(ValueError, match="only for extract_mineru"):
        build_plan(config("stage=reduce_and_compact", "split=other"), ROOT)


def test_mineru_default_has_no_page_cutoff() -> None:
    plan, _ = build_plan(config("stage=extract_mineru", "launcher=slurm_a100"), ROOT)
    assert plan["stage"]["end_page_id"] is None
    assert "--end-page-id" not in mineru_caller_args(plan, "http://127.0.0.1:8002")


def test_mineru_rejects_other_split_with_full_folder_scan_before_submission() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        build_plan(config("stage=extract_mineru", "launcher=pbs_h200", "split=other",
                          "stage.all_domain_pdfs=true"), ROOT)


def test_mineru_requires_gpu_and_describes_one_job() -> None:
    for profile in ("slurm_cpu", "pbs_rt_hc"):
        with pytest.raises(ValueError, match="requires a GPU launcher"):
            build_plan(config("stage=extract_mineru", f"launcher={profile}"), ROOT)
    for profile in ("local_gpu", "pbs_h200", "pbs_rt_hg"):
        plan, _ = build_plan(config("stage=extract_mineru", f"launcher={profile}"), ROOT)
        assert plan["execution"]["jobs"] == 1
        assert plan["execution"]["resources"]["gpus"] == 1
    plan, _ = build_plan(config("stage=extract_mineru", "launcher=slurm_a100"), ROOT)
    assert plan["execution"]["jobs"] == 1
    assert plan["execution"]["components"] == ["MinerU server", "extraction caller"]
    assert plan["execution"]["resources"] == {
        "partition": "gpu_a100", "gpus": 1, "cpus": 16, "memory": "32G", "walltime": "03:00:00"}
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base="1.3"):
        composed = compose(config_name="config", overrides=["stage=extract_mineru", "launcher=slurm_a100"],
                           return_hydra_config=True)
    assert "module load CUDA/12.6.0" in list(composed.hydra.launcher.setup)


def test_h200_mineru_defaults_propagate_through_plan_and_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config("stage=extract_mineru", "launcher=pbs_h200")
    plan, _ = build_plan(cfg, ROOT, tmp_path / "run")
    assert plan["stage"]["server_concurrency"] == 32
    assert plan["stage"]["max_in_flight"] == 32
    assert plan["execution"]["server_concurrency"] == 32
    assert plan["execution"]["max_in_flight"] == 32
    assert plan["execution"]["resources"]["queue"] == cfg["launcher"]["queue"]
    assert "--max-in-flight" in mineru_caller_args(plan, "http://127.0.0.1:8002")
    calls = []
    monkeypatch.setattr("hydra_run.subprocess.run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    monkeypatch.setattr("hydra_run.socket.socket", lambda *args: FakeSocket())
    monkeypatch.setenv("TMPDIR", str(tmp_path / "pbs-tmp"))
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    execute_stage(plan)
    cmd, kwargs = calls[0]
    assert cmd.count("--max-in-flight") == 1
    assert cmd[cmd.index("--max-in-flight") + 1] == "32"
    assert kwargs["env"]["MINERU_API_MAX_CONCURRENT_REQUESTS"] == "32"
    assert Path(kwargs["env"]["TMPDIR"]).parent.parent == tmp_path / "pbs-tmp"
    assert Path(kwargs["env"]["TMPDIR"]).name == "tmp"


def test_mineru_pbs_gpu_uuid_is_mapped_before_starting_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = build_plan(config("stage=extract_mineru", "launcher=pbs_h200"), ROOT, tmp_path / "run")
    gpu_uuid = "GPU-765a1d87-07a5-a69c-dcce-56244d200dbe"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"0, GPU-other\n3, {gpu_uuid}\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("hydra_run.subprocess.run", fake_run)
    monkeypatch.setattr("hydra_run.socket.socket", lambda *args: FakeSocket())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "pbs-tmp"))
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    execute_stage(plan)
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert calls[1][1]["env"]["MINERU_API_MAX_CONCURRENT_REQUESTS"] == "32"
    assert calls[1][0][calls[1][0].index("--max-in-flight") + 1] == "32"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu_uuid


def test_mineru_numeric_gpu_visibility_is_left_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hydra_run.subprocess.run", lambda *args, **kwargs: pytest.fail("unexpected nvidia-smi"))
    env = {"CUDA_VISIBLE_DEVICES": "1,0"}
    normalize_mineru_cuda_visibility(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_mineru_unknown_gpu_uuid_fails_before_starting_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-available\n")

    monkeypatch.setattr("hydra_run.subprocess.run", fake_run)
    env = {"CUDA_VISIBLE_DEVICES": "GPU-missing"}
    with pytest.raises(RuntimeError, match="matched 0 GPUs"):
        normalize_mineru_cuda_visibility(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-missing"


def test_h200_mineru_submits_one_pbs_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="123.server\n", stderr="")

    monkeypatch.setattr("hydra_run.subprocess.run", fake_run)
    run_dir = tmp_path / "mineru-job"
    launch(config("stage=extract_mineru", "launcher=pbs_h200"), ROOT, hydra_run_dir=run_dir)
    assert len(commands) == 1
    assert commands[0][0] == "qsub"
    assert "select=1:ncpus=16:mem=160gb:ngpus=1" in commands[0]
    assert "-m hydra_run --worker" in (run_dir / "job.sh").read_text()
    saved = json.loads((run_dir / "plan.json").read_text())
    assert saved["execution"]["jobs"] == 1
    assert saved["execution"]["server_concurrency"] == 32
    assert saved["execution"]["max_in_flight"] == 32


def test_h200_mineru_cli_accepts_full_domain_pbs_launch() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli", "launcher=pbs_h200", "split=train+test",
         "stage.all_domain_pdfs=true", "stage.domains=[ACL]", "dry_run=true",
         "stage=extract_mineru"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    output = json.loads(result.stdout)
    assert output["submission"][0] == "qsub"
    assert output["plan"]["execution"]["jobs"] == 1
    assert output["plan"]["stage"]["all_domain_pdfs"] is True
    assert output["plan"]["stage"]["domains"] == ["ACL"]


def test_mineru_concurrency_hydra_overrides_remain_independent() -> None:
    plan, _ = build_plan(config("stage=extract_mineru", "launcher=pbs_h200",
                                "stage.server_concurrency=24", "stage.max_in_flight=16"), ROOT)
    assert plan["execution"]["server_concurrency"] == 24
    assert plan["execution"]["max_in_flight"] == 16
    args = mineru_caller_args(plan, "http://127.0.0.1:8002")
    assert args[args.index("--max-in-flight") + 1] == "16"


def test_reduce_and_compact_preserves_mineru_output_and_compacts_copy(tmp_path: Path) -> None:
    source = tmp_path / "processed" / "papers" / "ACL" / "ACL_paper"
    content = source / "mineru" / "raw" / "hybrid_auto" / "paper_content_list_v2.json"
    content.parent.mkdir(parents=True)
    content.write_text(json.dumps([[{"type": "text", "text": "Paper text"}]]), encoding="utf-8")
    (source / "markdown.md").write_text("# Paper\n", encoding="utf-8")
    (source / "paper.json").write_text(json.dumps({
        "paper_id": "ACL_paper",
        "source_pdf_relpath": "ACL/paper.pdf",
        "source_paper_dataset": "ACL",
        "figures": [],
        "structured_outputs": {"v2": "mineru/raw/hybrid_auto/paper_content_list_v2.json"},
    }), encoding="utf-8")
    (source / "figures.json").write_text("[]", encoding="utf-8")
    (source / "_SUCCESS").touch()
    pdf = tmp_path / "pdf_datasets" / "ACL" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7\n")

    cfg = config("stage=reduce_and_compact", "launcher=local_gpu",
                 f"launcher.dataset_root={tmp_path}", "dataset.split_index=null")
    plan, _ = build_plan(cfg, ROOT, tmp_path / "run")
    execute_stage(plan)

    destination = tmp_path / "preprocessed" / "ACL" / "ACL_paper"
    assert (source / "mineru").exists()
    assert json.loads((source / "paper.json").read_text(encoding="utf-8"))["structured_outputs"]
    assert (destination / "paper.pdf").read_bytes() == b"%PDF-1.7\n"
    assert (destination / "images").is_dir()
    assert not (destination / "mineru").exists()


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


def test_stage_log_exposes_stdout_file_descriptor(tmp_path: Path) -> None:
    """vLLM's worker uses sys.stdout.fileno() while initializing NCCL."""
    with (tmp_path / "stdout.txt").open("w") as original:
        with contextlib.redirect_stdout(original), stage_log(tmp_path, "describe_figures"):
            assert sys.stdout.fileno() == original.fileno()
            assert os.fstat(sys.stdout.fileno()) == os.fstat(original.fileno())


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
        "--input-dir", "data/pdf_datasets", "--output-dir", "data/processed",
        "--split", "train", "--limit", "3"]
