#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run python scripts/reduce_preprocessed_collection.py --preprocessed-dir data/preprocessed/papers
uv run python scripts/compact_preprocessed_collection.py --preprocessed-dir data/preprocessed/papers
