#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run dataset-generation extract-mineru-pdfs \
  --input-dir data/pdf_datasets \
  --output-dir data/preprocessed \
  "$@"
