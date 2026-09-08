#!/usr/bin/env bash
set -euo pipefail

PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/preprocessed/papers}"
PDF_DIR="${PDF_DIR:-data/acl_subset/pdfs}"
MAX_PAGES="${MAX_PAGES:-10}"
REPORT="${REPORT:-data/preprocessed_reduction_report.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
FAIL_FAST="${FAIL_FAST:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "$(dirname "$0")/.."

extra_args=()
if [[ "$FAIL_FAST" == "1" ]]; then
  extra_args+=(--fail-fast)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"$PYTHON_BIN" scripts/reduce_preprocessed_collection.py \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --max-pages "$MAX_PAGES" \
  --report "$REPORT" \
  --workers "$WORKERS" \
  --progress-every "$PROGRESS_EVERY" \
  "${extra_args[@]}"

"$PYTHON_BIN" scripts/compact_preprocessed_collection.py \
  --preprocessed-dir "$PREPROCESSED_DIR" \
  --pdf-dir "$PDF_DIR" \
  --workers "$WORKERS" \
  --progress-every "$PROGRESS_EVERY" \
  "${extra_args[@]}"
