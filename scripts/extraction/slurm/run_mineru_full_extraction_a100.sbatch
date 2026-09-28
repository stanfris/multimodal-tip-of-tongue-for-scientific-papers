#!/usr/bin/env bash
#SBATCH --job-name=mineru_extract
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=scripts/extraction/slurm/mineru_extract_%j.out
#SBATCH --error=scripts/extraction/slurm/mineru_extract_%j.err

set -euo pipefail

ROOT_DIR_ARG=""
CALLER_ARGS=()
while (($#)); do
  case "$1" in
    --root-dir)
      if (($# < 2)); then
        echo "--root-dir requires a path." >&2
        exit 2
      fi
      ROOT_DIR_ARG="$2"
      shift 2
      ;;
    --root-dir=*)
      ROOT_DIR_ARG="${1#*=}"
      shift
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

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  DEFAULT_ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  DEFAULT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
ROOT_DIR="${ROOT_DIR_ARG:-${ROOT_DIR:-$DEFAULT_ROOT_DIR}}"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh" ]]; then
  echo "Repository root not found at: $ROOT_DIR" >&2
  echo "Submit this job from the repository root or set ROOT_DIR explicitly." >&2
  exit 2
fi

source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"

choose_free_port() {
  local python_bin
  python_bin="$(mineru_python)"
  "$python_bin" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="$JOB_ID"
DATASET_DIR="${DATASET_DIR:-/scratch-shared/sfris1}"
MINERU_PORT="${MINERU_PORT:-$(choose_free_port)}"
MINERU_STARTUP_TIMEOUT="${MINERU_STARTUP_TIMEOUT:-600}"
MINERU_CONCURRENCY="${MINERU_CONCURRENCY:-8}"
API_URL="http://127.0.0.1:${MINERU_PORT}"
MINERU_VENV="${MINERU_VENV:-.venv-mineru}"
export MINERU_API_MAX_CONCURRENT_REQUESTS="$MINERU_CONCURRENCY"

if [[ ! -x "$MINERU_VENV/bin/python" ]] || [[ ! -x "$MINERU_VENV/bin/mineru-router" ]]; then
  echo "Dedicated MinerU environment not found: $MINERU_VENV" >&2
  echo "Create it with scripts/environment/sync_mineru_env.sh before submitting." >&2
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
  echo "MinerU requires ninja for vLLM/FlashInfer JIT compilation." >&2
  echo "Repair the environment with scripts/environment/sync_mineru_env.sh before submitting." >&2
  exit 2
fi

if ! mineru_check_pdftext_compatibility "$MINERU_VENV/bin/python"; then
  echo "Repair the environment with scripts/environment/sync_mineru_env.sh before submitting." >&2
  exit 2
fi

SLURM_TMP_BASE="${SLURM_TMPDIR:-/tmp/${USER:-mineru}/mineru_${RUN_ID}}"
mkdir -p "$SLURM_TMP_BASE" scripts/extraction/slurm
export TMPDIR="$SLURM_TMP_BASE/tmp"
mkdir -p "$TMPDIR"

SERVER_LOG="${MINERU_SERVER_LOG:-scripts/extraction/slurm/${RUN_ID}.server.log}"
CALLER_LOG="${MINERU_CALLER_LOG:-scripts/extraction/slurm/${RUN_ID}.caller.log}"
SERVER_CMD="${MINERU_SERVER_CMD:-$MINERU_VENV_DIR/bin/mineru-router --host 127.0.0.1 --port ${MINERU_PORT}}"

CALLER_CMD=(
  env "PYTHONPATH=$ROOT_DIR/src" "$MINERU_VENV/bin/python" -m extraction
  --input-dir "${INPUT_DIR:-$DATASET_DIR/pdf_datasets}"
  --split-index "${SPLIT_INDEX:-$DATASET_DIR/data/splits/pdf_dataset_split.json}"
  --split "${SPLIT:-train+test}"
  --output-dir "${OUTPUT_DIR:-$DATASET_DIR/processed}"
  --api-url "$API_URL"
  --max-in-flight "$MINERU_CONCURRENCY"
  "${CALLER_ARGS[@]}"
)

printf 'Job %s processes the full %s split from %s.\n' "$JOB_ID" "${SPLIT:-train+test}" "${INPUT_DIR:-$DATASET_DIR/pdf_datasets}"
printf 'MinerU server and caller concurrency: %s\n' "$MINERU_CONCURRENCY"
printf 'MinerU server log: %s\n' "$SERVER_LOG"
printf 'MinerU caller log: %s\n' "$CALLER_LOG"
printf 'Temporary path: %s\n' "$SLURM_TMP_BASE"

mineru_run_caller_with_server \
  "$API_URL" \
  "$SERVER_CMD" \
  "$SERVER_LOG" \
  "$CALLER_LOG" \
  "$MINERU_STARTUP_TIMEOUT" \
  "${CALLER_CMD[@]}"
