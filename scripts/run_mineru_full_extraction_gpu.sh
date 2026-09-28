#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/mineru_server_launcher.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mineru_full_extraction_gpu.sh [launcher options] [--] [extract-mineru-pdfs args]

Starts one MinerU router/API server in the background, waits for /health, runs
the existing extract-mineru-pdfs caller against 127.0.0.1, then shuts the server
down and exits with the caller's status.

Launcher options:
  --port PORT                 MinerU server port (default: $MINERU_PORT or 8002)
  --startup-timeout SECONDS   Readiness timeout (default: $MINERU_STARTUP_TIMEOUT or 600)
  --server-log PATH           Server log path
  --caller-log PATH           Caller log path
  --server-cmd COMMAND        Full server command (default uses mineru-router)
  -h, --help                  Show this help

Any other arguments are forwarded to:
  uv run --no-sync dataset-generation extract-mineru-pdfs
EOF
}

MINERU_PORT="${MINERU_PORT:-8002}"
MINERU_STARTUP_TIMEOUT="${MINERU_STARTUP_TIMEOUT:-600}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ).$$}"
SERVER_LOG="${MINERU_SERVER_LOG:-logs/mineru/local/${RUN_ID}.server.log}"
CALLER_LOG="${MINERU_CALLER_LOG:-logs/mineru/local/${RUN_ID}.caller.log}"
SERVER_CMD="${MINERU_SERVER_CMD:-}"
CALLER_ARGS=()

while (($#)); do
  case "$1" in
    --port)
      MINERU_PORT="$2"
      shift 2
      ;;
    --startup-timeout)
      MINERU_STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --server-log)
      SERVER_LOG="$2"
      shift 2
      ;;
    --caller-log)
      CALLER_LOG="$2"
      shift 2
      ;;
    --server-cmd)
      SERVER_CMD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      CALLER_ARGS+=("$@")
      break
      ;;
    *)
      CALLER_ARGS+=("$1")
      shift
      ;;
  esac
done

API_URL="http://127.0.0.1:${MINERU_PORT}"
if [[ -z "$SERVER_CMD" ]]; then
  SERVER_CMD="uv run --no-sync mineru-router --host 127.0.0.1 --port ${MINERU_PORT}"
fi

CALLER_CMD=(
  uv run --no-sync dataset-generation extract-mineru-pdfs
  --input-dir data/pdf_datasets
  --split-index data/splits/pdf_dataset_split.json
  --split all
  --output-dir data/preprocessed
  --api-url "$API_URL"
  "${CALLER_ARGS[@]}"
)

printf 'MinerU server log: %s\n' "$SERVER_LOG"
printf 'MinerU caller log: %s\n' "$CALLER_LOG"
mineru_run_caller_with_server \
  "$API_URL" \
  "$SERVER_CMD" \
  "$SERVER_LOG" \
  "$CALLER_LOG" \
  "$MINERU_STARTUP_TIMEOUT" \
  "${CALLER_CMD[@]}"
