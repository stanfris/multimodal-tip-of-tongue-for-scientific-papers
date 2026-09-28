#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"

MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
if [[ ! -x "$MINERU_VENV/bin/mineru-router" ]]; then
  printf 'Dedicated MinerU environment not found. Run scripts/environment/sync_mineru_env.sh first.\n' >&2
  exit 2
fi

if [[ "$MINERU_VENV" = /* ]]; then
  MINERU_VENV_DIR="$MINERU_VENV"
else
  MINERU_VENV_DIR="$ROOT_DIR/$MINERU_VENV"
fi
export VIRTUAL_ENV="$MINERU_VENV_DIR"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if ! command -v ninja >/dev/null 2>&1; then
  printf 'MinerU requires ninja for vLLM/FlashInfer JIT compilation.\n' >&2
  printf 'Repair the environment with: scripts/environment/sync_mineru_env.sh\n' >&2
  exit 2
fi

if ! mineru_check_pdftext_compatibility "$MINERU_VENV/bin/python"; then
  printf 'Repair the environment with: scripts/environment/sync_mineru_env.sh\n' >&2
  exit 2
fi

exec "$MINERU_VENV/bin/mineru-router" "$@"
