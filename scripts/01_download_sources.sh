#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
SPLIT="${SPLIT:-train}"
PAPER_SPLIT="${PAPER_SPLIT:-$SPLIT}"

uv run dataset-generation download-sources \
  --data-dir "$DATA_DIR" \
  --split "$SPLIT" \
  --paper-split "$PAPER_SPLIT"
