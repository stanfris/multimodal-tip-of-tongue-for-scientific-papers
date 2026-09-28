#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script so the environment remains active:" >&2
  echo "  source scripts/environment/activate_env.sh {main|mineru}" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT="${1:-}"

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
    echo "Usage: source scripts/environment/activate_env.sh {main|mineru}" >&2
    return 2
    ;;
esac

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Environment is not initialized: $VENV_DIR" >&2
  echo "Create it with: $SETUP_SCRIPT" >&2
  return 2
fi

if declare -F deactivate >/dev/null 2>&1; then
  deactivate
fi

source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
hash -r
printf 'Active environment: %s (%s)\n' "$ENVIRONMENT" "$VENV_DIR"
