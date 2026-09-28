#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
if [[ ! -x "$MINERU_VENV/bin/mineru-router" ]]; then
  printf 'Dedicated MinerU environment not found. Run scripts/environment/sync_mineru_env.sh first.\n' >&2
  exit 2
fi

exec "$MINERU_VENV/bin/mineru-router" "$@"
