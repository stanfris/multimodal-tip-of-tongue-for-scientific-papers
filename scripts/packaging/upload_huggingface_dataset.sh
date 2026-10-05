#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${1:-huggingface_dataset}"
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  shift
else
  DATASET_DIR="huggingface_dataset"
fi

HF_REPO_ID="${HF_REPO_ID:-kasys/open-source-scientific-documents}"
HF_CLI="${HF_CLI:-hf}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

COMMAND=(uv run python -m dataset_packaging.prepare_hf_dataset
  --output-dir "$DATASET_DIR"
  --repo-id "$HF_REPO_ID"
  --hf-cli "$HF_CLI")
ADDITIONAL_ONLY=0
for arg in "$@"; do
  if [[ "$arg" == --additional-only ]]; then
    ADDITIONAL_ONLY=1
    break
  fi
done

if (( ADDITIONAL_ONLY == 0 )); then
  COMMAND+=(--upload-only)
fi
COMMAND+=("$@")
"${COMMAND[@]}"
