#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python -m document_downloads.build_arxiv_open_reuse \
  --target-for-prefix physics.=30000 \
  --target-for-prefix eess.=60000 \
  "$@"
