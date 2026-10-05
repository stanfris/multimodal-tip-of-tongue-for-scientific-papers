from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ENTRYPOINTS = {
    "scripts/environment/sync_mineru_env.sh": "transformers==4.57.3",
    "scripts/pdf_corpus/build_pdf_dataset_split.sh": "python -m pdf_corpus.build_pdf_dataset_split",
    "scripts/pdf_corpus/build_pdf_datasets_folder.sh": "python -m pdf_corpus.build_pdf_datasets_folder",
    "scripts/pdf_corpus/split_arxiv_pdfs_by_domain.sh": "python -m pdf_corpus.split_arxiv_pdfs_by_domain",
    "scripts/pdf_corpus/pdf_page_distribution.sh": "python -m pdf_corpus.pdf_page_distribution",
    "scripts/document_splits/build_document_split.sh": "python -m document_splits.build_document_split",
    "scripts/review/run_review_app.sh": "streamlit run",
    "scripts/extraction/probe_mineru_env.sh": "dataset-generation probe-mineru-env",
    "scripts/extraction/benchmark_mineru_extraction.sh": "dataset-generation benchmark-mineru-pdfs",
    "scripts/preprocessing/parsed_dataset_stats.sh": "dataset-generation parsed-dataset-stats",
    "scripts/reporting/generated_artifact_stats.sh": "dataset-generation stats",
    "scripts/packaging/prepare_huggingface_dataset.sh": "dataset-generation prepare-hf-dataset",
    "scripts/packaging/restore_pdf_datasets_from_huggingface.sh": "python -m dataset_packaging.restore_hf_dataset",
}


def test_operational_python_entrypoints_have_bash_wrappers() -> None:
    for relative_path, command in EXPECTED_ENTRYPOINTS.items():
        wrapper = PROJECT_ROOT / relative_path
        assert wrapper.is_file(), f"Missing Bash wrapper: {relative_path}"
        assert os.access(wrapper, os.X_OK), f"Bash wrapper is not executable: {relative_path}"
        contents = wrapper.read_text(encoding="utf-8")
        assert contents.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in contents
        assert command in contents
        assert '"$@"' in contents


def test_documented_bash_scripts_exist() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    documented_scripts = set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.sh", readme))

    assert documented_scripts
    for relative_path in documented_scripts:
        assert (PROJECT_ROOT / relative_path).is_file(), f"README references missing script: {relative_path}"


def test_huggingface_upload_wrapper_selects_additional_mode(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    fake_uv.chmod(0o755)
    wrapper = PROJECT_ROOT / "scripts/packaging/upload_huggingface_dataset.sh"
    capture = tmp_path / "args.txt"
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "CAPTURE_ARGS": str(capture)}

    subprocess.run(["bash", str(wrapper), "package", "--additional-only", "--data-root", "dataset"], check=True, env=env)
    additional_args = capture.read_text().splitlines()
    assert "--additional-only" in additional_args
    assert "--upload-only" not in additional_args
    assert additional_args[additional_args.index("--output-dir") + 1] == "package"

    subprocess.run(["bash", str(wrapper), "package"], check=True, env=env)
    assert "--upload-only" in capture.read_text().splitlines()


def test_main_and_mineru_environment_switchers_exist() -> None:
    activate = (PROJECT_ROOT / "scripts/environment/activate_env.sh").read_text(encoding="utf-8")
    runner = (PROJECT_ROOT / "scripts/environment/run_in_env.sh").read_text(encoding="utf-8")

    for environment in ("main", "mineru"):
        assert f"{environment})" in activate
        assert f"{environment})" in runner


def test_mineru_environment_includes_flashinfer_build_tool() -> None:
    sync_script = (PROJECT_ROOT / "scripts/environment/sync_mineru_env.sh").read_text(encoding="utf-8")

    assert "ninja>=1.11,<2" in sync_script
    assert "pdftext==0.6.3" in sync_script


def test_mineru_server_launchers_expose_environment_build_tools() -> None:
    launchers = (
        PROJECT_ROOT / "scripts/extraction/start_mineru_router.sh",
        PROJECT_ROOT / "scripts/extraction/run_mineru_full_extraction_gpu.sh",
        PROJECT_ROOT / "scripts/extraction/slurm/run_mineru_full_extraction_a100.job",
    )

    for launcher in launchers:
        script = launcher.read_text(encoding="utf-8")
        assert 'export VIRTUAL_ENV="$MINERU_VENV_DIR"' in script
        assert 'export PATH="$VIRTUAL_ENV/bin:$PATH"' in script
        assert "command -v ninja" in script or 'shutil.which("ninja")' in script
        assert "mineru_check_pdftext_compatibility" in script


def test_mineru_gpu_launcher_rejects_cpu_fallback() -> None:
    launcher = (PROJECT_ROOT / "scripts/extraction/run_mineru_full_extraction_gpu.sh").read_text()
    assert "torch.cuda.is_available()" in launcher
    assert "refusing CPU fallback" in launcher
    assert "--output-dir data/processed" in launcher


def test_mineru_runtime_rejects_non_iterable_pagechars_dependency() -> None:
    launcher = (PROJECT_ROOT / "scripts/extraction/mineru_server_launcher.sh").read_text(
        encoding="utf-8"
    )

    assert 'expected = "0.6.3"' in launcher
    assert "non-iterable PageChars API" in launcher


def test_mineru_server_output_is_logged_but_only_caller_output_is_teed() -> None:
    launcher = (PROJECT_ROOT / "scripts/extraction/mineru_server_launcher.sh").read_text(
        encoding="utf-8"
    )

    assert '>>"$server_log" 2>&1 &' in launcher
    assert '> >(tee -a "$server_log")' not in launcher
    assert 'exec env PYTHONUNBUFFERED=1 "${caller_cmd[@]}"' in launcher
    assert ') 2>&1 | tee -a "$caller_log"' in launcher
    assert "MinerU server exited before readiness with status %s" in launcher


def test_mineru_lifecycle_stops_mock_server_after_caller_success_and_failure(tmp_path: Path) -> None:
    lifecycle = PROJECT_ROOT / "scripts/extraction/mineru_server_launcher.sh"
    for caller_exit in (0, 7):
        server_log = tmp_path / f"server-{caller_exit}.log"
        caller_log = tmp_path / f"caller-{caller_exit}.log"
        script = f'''set -uo pipefail
source "{lifecycle}"
mineru_health_check() {{ return 0; }}
mineru_run_caller_with_server http://127.0.0.1:43123 'sleep 60' "{server_log}" "{caller_log}" 2 bash -c 'exit {caller_exit}'
status=$?
if kill -0 "$MINERU_SERVER_PID" 2>/dev/null; then exit 99; fi
exit "$status"
'''
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert result.returncode == caller_exit, result.stderr
        assert "Starting MinerU server" in server_log.read_text()
        assert "Stopping MinerU server" in server_log.read_text()
        assert "Starting caller" in caller_log.read_text()


def test_mineru_lifecycle_stops_mock_server_when_readiness_fails(tmp_path: Path) -> None:
    lifecycle = PROJECT_ROOT / "scripts/extraction/mineru_server_launcher.sh"
    server_log = tmp_path / "server.log"
    caller_log = tmp_path / "caller.log"
    script = f'''set -uo pipefail
source "{lifecycle}"
mineru_health_check() {{ return 1; }}
mineru_run_caller_with_server http://127.0.0.1:43123 'sleep 60' "{server_log}" "{caller_log}" 1 bash -c 'exit 0'
status=$?
if kill -0 "$MINERU_SERVER_PID" 2>/dev/null; then exit 99; fi
exit "$status"
'''
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 124
    assert "Stopping MinerU server" in server_log.read_text()
    assert not caller_log.exists() or "Starting caller" not in caller_log.read_text()


def test_slurm_mineru_launcher_preserves_log_stream_separation() -> None:
    script = (
        PROJECT_ROOT / "scripts/extraction/slurm/run_mineru_full_extraction_a100.job"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array" not in script
    assert "SLURM_ARRAY_TASK_ID" not in script
    assert "#SBATCH --output=logs/extraction/slurm/mineru_extract_%j.out" in script
    assert "#SBATCH --error=logs/extraction/slurm/mineru_extract_%j.err" in script
    assert 'DEFAULT_ROOT_DIR="$SLURM_SUBMIT_DIR"' in script
    assert "--root-dir)" in script
    assert 'ROOT_DIR="${ROOT_DIR_ARG:-${ROOT_DIR:-$DEFAULT_ROOT_DIR}}"' in script
    assert 'DATASET_DIR="${DATASET_DIR:-/scratch-shared/sfris1}"' in script
    assert 'MINERU_CONCURRENCY="${MINERU_CONCURRENCY:-8}"' in script
    assert "--split)" in script
    assert "--split=*)" in script
    assert 'SPLIT_ARG="$2"' in script
    assert 'EXTRACTION_SCOPE="${SPLIT_ARG:-${SPLIT:-full}}"' in script
    assert "CALLER_CMD+=(--split all --all-domain-pdfs)" in script
    assert '--split "$EXTRACTION_SCOPE"' in script
    assert 'Resume is enabled: completed documents are preserved' in script
    assert 'export MINERU_API_MAX_CONCURRENT_REQUESTS="$MINERU_CONCURRENCY"' in script
    assert '--max-in-flight "$MINERU_CONCURRENCY"' in script
    assert 'source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"' in script
    assert 'SERVER_LOG="${MINERU_SERVER_LOG:-logs/extraction/slurm/${RUN_ID}.server.log}"' in script
    assert 'CALLER_LOG="${MINERU_CALLER_LOG:-logs/extraction/slurm/${RUN_ID}.caller.log}"' in script
    assert 'SERVER_CMD="${MINERU_SERVER_CMD:-$MINERU_VENV_DIR/bin/mineru-router' in script
    assert '--output-dir "${OUTPUT_DIR:-$DATASET_DIR/processed}"' in script
    assert "mineru_run_caller_with_server" in script


def test_slurm_mineru_launcher_builds_indexed_split_command(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts/extraction"
    scripts_dir.mkdir(parents=True)
    launcher = scripts_dir / "mineru_server_launcher.sh"
    launcher.write_text(
        """mineru_python() { printf '%s\\n' \"$ROOT_DIR/.venv-mineru/bin/python\"; }
mineru_check_pdftext_compatibility() { return 0; }
mineru_run_caller_with_server() {
  shift 5
  printf 'CALLER'
  printf ' <%s>' \"$@\"
  printf '\\n'
}
""",
        encoding="utf-8",
    )

    venv_bin = tmp_path / ".venv-mineru/bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nprintf '43123\\n'\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_router = venv_bin / "mineru-router"
    fake_router.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_router.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ninja = fake_bin / "ninja"
    fake_ninja.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_ninja.chmod(0o755)
    bash_env = tmp_path / "bash_env"
    bash_env.write_text("module() { return 0; }\n", encoding="utf-8")

    job = PROJECT_ROOT / "scripts/extraction/slurm/run_mineru_full_extraction_a100.job"
    env = {
        **os.environ,
        "BASH_ENV": str(bash_env),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_ID": "test-job",
        "SLURM_SUBMIT_DIR": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(job), "--root-dir", str(tmp_path), "--split", "train+test", "--domains", "ACL"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "processes the indexed train+test split" in result.stdout
    assert "<--split> <train+test>" in result.stdout
    assert "<--domains> <ACL>" in result.stdout
    assert "<--all-domain-pdfs>" not in result.stdout


def test_slurm_preprocessing_uses_hydra_submitit_profile() -> None:
    profile = (PROJECT_ROOT / "config/launcher/slurm_cpu.yaml").read_text(encoding="utf-8")
    assert "override /hydra/launcher: submitit_slurm" in profile
    assert "partition: rome" in profile
    assert "cpus_per_task: 16" in profile
    assert "mem_gb: 28" in profile
    assert "gpus_per_node: 0" in profile
