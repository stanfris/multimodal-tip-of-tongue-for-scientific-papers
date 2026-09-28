#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if (( $# == 0 )); then
  echo "Usage: $0 DESTINATION_DIR [hf download options...]" >&2
  echo "Set HF_REPO_ID to download a different dataset repository." >&2
  exit 2
fi

DESTINATION_DIR="$1"
shift

HF_REPO_ID="${HF_REPO_ID:-kasys/open-source-scientific-documents}"
HF_CLI="${HF_CLI:-uv run hf}"

if ! command -v "${HF_CLI_CMD[0]}" >/dev/null 2>&1; then
  echo "Hugging Face CLI not found: $HF_CLI" >&2
  echo "Install it or set HF_CLI to the path of the hf executable." >&2
  exit 127
fi

mkdir -p "$DESTINATION_DIR"

"${HF_CLI_CMD[@]}" download "$HF_REPO_ID" \
  --repo-type dataset \
  --local-dir "$DESTINATION_DIR" \
  "$@"
