#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-data/acl_subset}"

uv run python scripts/build_acl_subset.py \
  --output-dir "$OUTPUT_DIR" \
  --metadata-only
