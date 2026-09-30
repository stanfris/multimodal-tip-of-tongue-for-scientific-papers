#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" exec uv run --no-sync dataset-generation stage=describe_textual_clues "$@"
