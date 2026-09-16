#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python -m dataset_generation.document_downloads.build_arxiv_open_reuse \
  --category-prefix eess. \
  --target-per-domain 30000 \
  --output-dir data/arxiv_open_reuse_engineering \
  "$@"
