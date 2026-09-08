#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/00_sync_env.sh"
"$ROOT_DIR/scripts/01_build_acl_subset.sh"
"$ROOT_DIR/scripts/02_download_acl_pdfs.sh"
"$ROOT_DIR/scripts/09_run_mineru_full_extraction.sh"
"$ROOT_DIR/scripts/03_build_canonical.sh"

if [[ "${RUN_VISUAL:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/04_describe_all_figures.sh"
fi

if [[ "${RUN_TEXTUAL:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/05_describe_all_textual_clues.sh"
fi

if [[ "${RUN_QUERIES:-1}" == "1" ]]; then
  "$ROOT_DIR/scripts/06_generate_queries.sh"
fi
