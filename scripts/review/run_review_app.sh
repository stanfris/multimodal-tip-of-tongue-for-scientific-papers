#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

exec uv run streamlit run --server.address 127.0.0.1 src/review/review_app.py "$@"
