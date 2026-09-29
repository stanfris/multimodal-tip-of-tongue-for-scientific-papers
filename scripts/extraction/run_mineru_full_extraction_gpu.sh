#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/extraction/run_mineru_full_extraction_gpu.sh [launcher options] [--] [extract-mineru-pdfs args]

Starts one MinerU router/API server in the background, waits for /health, runs
the existing extract-mineru-pdfs caller against 127.0.0.1, then shuts the server
down and exits with the caller's status. Server output is written only to the
server log. Extraction progress is shown live and appended to the caller log.

Launcher options:
  --port PORT                 MinerU server port (default: $MINERU_PORT or 8002)
  --startup-timeout SECONDS   Readiness timeout (default: $MINERU_STARTUP_TIMEOUT or 600)
  --server-log PATH           Server log path
  --caller-log PATH           Caller log path
  --server-cmd COMMAND        Full server command (default uses mineru-router)
  -h, --help                  Show this help

Any other arguments are forwarded to:
  .venv-mineru/bin/python -m extraction
EOF
}

MINERU_PORT="${MINERU_PORT:-8002}"
MINERU_STARTUP_TIMEOUT="${MINERU_STARTUP_TIMEOUT:-600}"
MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ).$$}"
SERVER_LOG="${MINERU_SERVER_LOG:-logs/extraction/local/${RUN_ID}.server.log}"
CALLER_LOG="${MINERU_CALLER_LOG:-logs/extraction/local/${RUN_ID}.caller.log}"
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

if [[ ! -x "$MINERU_VENV/bin/python" ]]; then
  printf 'Dedicated MinerU environment not found: %s\n' "$MINERU_VENV" >&2
  printf 'Create it with: scripts/environment/sync_mineru_env.sh\n' >&2
  exit 2
fi

if [[ "$MINERU_VENV" = /* ]]; then
  MINERU_VENV_DIR="$MINERU_VENV"
else
  MINERU_VENV_DIR="$ROOT_DIR/$MINERU_VENV"
fi
export VIRTUAL_ENV="$MINERU_VENV_DIR"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if ! mineru_check_pdftext_compatibility "$MINERU_VENV/bin/python"; then
  printf 'Repair the environment with: scripts/environment/sync_mineru_env.sh\n' >&2
  exit 2
fi

if ! "$MINERU_VENV/bin/python" - <<'PY'
import shutil
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    mineru_version = version("mineru")
    transformers_version = version("transformers")
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2Config

    config = PPDocLayoutV2Config()
except (ImportError, PackageNotFoundError) as exc:
    raise SystemExit(f"MinerU runtime dependency is missing: {exc}") from exc

if Version(mineru_version) != Version("3.4.0"):
    raise SystemExit(f"Expected MinerU 3.4.0, found {mineru_version}")
if Version(transformers_version) != Version("4.57.3"):
    raise SystemExit(f"Expected Transformers 4.57.3, found {transformers_version}")
if not hasattr(config, "reading_order_config"):
    raise SystemExit(
        "Incompatible MinerU runtime: PPDocLayoutV2Config lacks reading_order_config"
    )
if shutil.which("ninja") is None:
    raise SystemExit("Missing ninja executable required by vLLM/FlashInfer JIT compilation")

print(
    f"MinerU runtime OK: mineru={mineru_version} "
    f"transformers={transformers_version} pdftext=0.6.3"
)
PY
then
  cat >&2 <<'EOF'
MinerU cannot start with the current environment.
Repair the pinned MinerU 3.x runtime, then rerun this command:
  scripts/environment/sync_mineru_env.sh
EOF
  exit 2
fi

API_URL="http://127.0.0.1:${MINERU_PORT}"
if [[ -z "$SERVER_CMD" ]]; then
  SERVER_CMD="$MINERU_VENV/bin/mineru-router --host 127.0.0.1 --port ${MINERU_PORT}"
fi

CALLER_CMD=(
  env "PYTHONPATH=$ROOT_DIR/src" "$MINERU_VENV/bin/python" -m extraction
  --input-dir data/pdf_datasets
  --split-index data/splits/pdf_dataset_split.json
  --split train+test
  --output-dir data/processed
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
