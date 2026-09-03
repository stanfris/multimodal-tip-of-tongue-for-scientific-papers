#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
SPLIT="${SPLIT:-train}"
DATASET_DIR="${DATASET_DIR:-$DATA_DIR/processed/acl_fig_markdown/$SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/canonical}"

uv run dataset-generation build-canonical \
  --data-dir "$DATA_DIR" \
  --dataset "$DATASET_DIR" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR"
