#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
MINERU_PYTHON_VERSION="${MINERU_PYTHON_VERSION:-3.11}"

case "$MINERU_VENV" in
  .venv-mineru|*/.venv-mineru) ;;
  *)
    echo "Refusing to clear unexpected MinerU environment path: $MINERU_VENV" >&2
    echo "MINERU_VENV must end with .venv-mineru" >&2
    exit 2
    ;;
esac

uv venv --clear "$MINERU_VENV" --python "$MINERU_PYTHON_VERSION"

uv pip install \
  --python "$MINERU_VENV/bin/python" \
  'mineru[all]==3.4.0' \
  'transformers==4.57.3' \
  'ninja>=1.11,<2' \
  "$@"
