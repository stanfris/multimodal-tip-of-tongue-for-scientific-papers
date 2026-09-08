#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-data/acl_subset}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"

uv run python scripts/build_acl_subset.py \
  --output-dir "$OUTPUT_DIR" \
  --download-pdfs \
  --max-workers "$MAX_WORKERS" \
  --sleep-seconds "$SLEEP_SECONDS"
