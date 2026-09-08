#!/usr/bin/env bash
set -euo pipefail

PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/preprocessed}"
PDF_DIR="${PDF_DIR:-data/acl_subset/pdfs}"
MAX_PAGES="${MAX_PAGES:-10}"
REPORT="${REPORT:-data/preprocessed_reduction_report.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$(dirname "$0")/.."

"$PYTHON_BIN" scripts/reduce_preprocessed_collection.py \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --max-pages "$MAX_PAGES" \
  --report "$REPORT"

"$PYTHON_BIN" scripts/compact_preprocessed_collection.py \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --pdf-dir "$PDF_DIR"
