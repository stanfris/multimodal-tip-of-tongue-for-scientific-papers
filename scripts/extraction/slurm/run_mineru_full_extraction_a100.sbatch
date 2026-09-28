#!/usr/bin/env bash
#SBATCH --job-name=mineru_extract
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-0%8
#SBATCH --output=mineru_extract_%A_%a.out
#SBATCH --error=mineru_extract_%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/extraction/mineru_server_launcher.sh"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script is intended to run under sbatch as an array job." >&2
  exit 2
fi

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

TASK_ID="$SLURM_ARRAY_TASK_ID"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_ID="${JOB_ID}_${TASK_ID}"
PDFS_PER_TASK="${PDFS_PER_TASK:-1}"
START_INDEX="${START_INDEX:-$((TASK_ID * PDFS_PER_TASK))}"
END_INDEX="${END_INDEX:-$((START_INDEX + PDFS_PER_TASK))}"
MINERU_PORT="${MINERU_PORT:-$(choose_free_port)}"
MINERU_STARTUP_TIMEOUT="${MINERU_STARTUP_TIMEOUT:-600}"
API_URL="http://127.0.0.1:${MINERU_PORT}"

SLURM_TMP_BASE="${SLURM_TMPDIR:-/tmp/${USER:-mineru}/mineru_${RUN_ID}}"
mkdir -p "$SLURM_TMP_BASE" logs/mineru/slurm logs/slurm
export TMPDIR="$SLURM_TMP_BASE/tmp"
mkdir -p "$TMPDIR"

SERVER_LOG="${MINERU_SERVER_LOG:-logs/mineru/slurm/${RUN_ID}.server.log}"
CALLER_LOG="${MINERU_CALLER_LOG:-logs/mineru/slurm/${RUN_ID}.caller.log}"
SERVER_CMD="${MINERU_SERVER_CMD:-uv run --no-sync mineru-router --host 127.0.0.1 --port ${MINERU_PORT}}"

CALLER_CMD=(
  uv run --no-sync dataset-generation extract-mineru-pdfs
  --input-dir "${INPUT_DIR:-data/pdf_datasets}"
  --split-index "${SPLIT_INDEX:-data/splits/pdf_dataset_split.json}"
  --split "${SPLIT:-train+test}"
  --output-dir "${OUTPUT_DIR:-data/preprocessed}"
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --api-url "$API_URL"
  "$@"
)

printf 'Job %s task %s processes discovered PDF indices [%s, %s).\n' "$JOB_ID" "$TASK_ID" "$START_INDEX" "$END_INDEX"
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
