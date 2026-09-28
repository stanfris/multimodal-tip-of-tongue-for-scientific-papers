from __future__ import annotations

import os
import re
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
        PROJECT_ROOT / "scripts/extraction/slurm/run_mineru_full_extraction_a100.sbatch",
    )

    for launcher in launchers:
        script = launcher.read_text(encoding="utf-8")
        assert 'export VIRTUAL_ENV="$MINERU_VENV_DIR"' in script
        assert 'export PATH="$VIRTUAL_ENV/bin:$PATH"' in script
        assert "command -v ninja" in script or 'shutil.which("ninja")' in script
        assert "mineru_check_pdftext_compatibility" in script


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
    assert '"${caller_cmd[@]}" 2>&1 | tee -a "$caller_log"' in launcher


def test_slurm_mineru_launcher_preserves_log_stream_separation() -> None:
    script = (
        PROJECT_ROOT / "scripts/extraction/slurm/run_mineru_full_extraction_a100.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --output=mineru_extract_%A_%a.out" in script
    assert "#SBATCH --error=mineru_extract_%A_%a.err" in script
    assert 'source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"' in script
    assert 'SERVER_LOG="${MINERU_SERVER_LOG:-logs/mineru/slurm/${RUN_ID}.server.log}"' in script
    assert 'CALLER_LOG="${MINERU_CALLER_LOG:-logs/mineru/slurm/${RUN_ID}.caller.log}"' in script
    assert "mineru_run_caller_with_server" in script
