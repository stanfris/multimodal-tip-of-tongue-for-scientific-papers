#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
INPUT_DIR="${INPUT_DIR:-$DATA_DIR/processed/mineru_pdf_extraction/papers}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/canonical}"

uv run dataset-generation build-canonical \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"
