#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python -m document_downloads.build_arxiv_open_reuse \
  --category-prefix physics. \
  --target-per-domain 30000 \
  --output-dir data/arxiv_open_reuse_physics \
  "$@"
