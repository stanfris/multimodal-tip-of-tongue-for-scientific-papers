#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${1:-huggingface_dataset}"
if (( $# > 0 )); then
  shift
fi

HF_REPO_ID="${HF_REPO_ID:-kasys/open-source-scientific-documents}"
HF_CLI="${HF_CLI:-hf}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

uv run python -m dataset_packaging.prepare_hf_dataset \
  --output-dir "$DATASET_DIR" \
  --repo-id "$HF_REPO_ID" \
  --hf-cli "$HF_CLI" \
  --upload-only \
  "$@"
