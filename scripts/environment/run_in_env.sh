#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if (( $# < 2 )); then
  echo "Usage: $0 {main|mineru} COMMAND [ARG ...]" >&2
  exit 2
fi

ENVIRONMENT="$1"
shift

case "$ENVIRONMENT" in
  main)
    VENV_DIR="$ROOT_DIR/.venv"
    SETUP_SCRIPT="scripts/environment/sync_env.sh"
    ;;
  mineru)
    VENV_DIR="$ROOT_DIR/.venv-mineru"
    SETUP_SCRIPT="scripts/environment/sync_mineru_env.sh"
    ;;
  *)
    echo "Unknown environment: $ENVIRONMENT (expected main or mineru)" >&2
    exit 2
    ;;
esac

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Environment is not initialized: $VENV_DIR" >&2
  echo "Create it with: $SETUP_SCRIPT" >&2
  exit 2
fi

cd "$ROOT_DIR"
exec env \
  VIRTUAL_ENV="$VENV_DIR" \
  PATH="$VENV_DIR/bin:$PATH" \
  "$@"
