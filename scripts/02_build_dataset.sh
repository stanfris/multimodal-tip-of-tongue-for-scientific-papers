#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
SPLIT="${SPLIT:-train}"
PAPER_SPLIT="${PAPER_SPLIT:-$SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/processed/acl_fig_markdown/$SPLIT}"
FORMAT="${FORMAT:-parquet}"

uv run dataset-generation build \
  --data-dir "$DATA_DIR" \
  --split "$SPLIT" \
  --paper-split "$PAPER_SPLIT" \
  --output "$OUTPUT_DIR" \
  --format "$FORMAT"
