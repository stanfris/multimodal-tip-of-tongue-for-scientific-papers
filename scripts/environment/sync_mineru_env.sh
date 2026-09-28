#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
MINERU_PYTHON_VERSION="${MINERU_PYTHON_VERSION:-3.11}"

if [[ ! -x "$MINERU_VENV/bin/python" ]]; then
  uv venv "$MINERU_VENV" --python "$MINERU_PYTHON_VERSION"
fi

uv pip install \
  --python "$MINERU_VENV/bin/python" \
  --upgrade \
  'mineru[all]>=3.4,<4' \
  'transformers>=4.57.3,<5' \
  "$@"
