#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python -m preprocessing.reduce_preprocessed_collection --preprocessed-dir data/preprocessed/papers
uv run python -m preprocessing.compact_preprocessed_collection --preprocessed-dir data/preprocessed/papers
