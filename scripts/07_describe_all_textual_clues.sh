#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
DATASET_DIR="${DATASET_DIR:-$DATA_DIR/preprocessed}"
CLUES_DIR="${CLUES_DIR:-$DATA_DIR/clues}"
SPLIT="${SPLIT:-train}"
TEXT_BACKEND="${TEXT_BACKEND:-transformers}"
TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen3-4B}"
MAX_TOKENS="${TEXT_MAX_TOKENS:-600}"
MAX_MARKDOWN_CHARS="${MAX_MARKDOWN_CHARS:-12000}"
TEMPERATURE="${TEXT_TEMPERATURE:-0.0}"
BATCH_SIZE="${TEXT_BATCH_SIZE:-8}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
LIMIT="${LIMIT:-}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  --backend "$TEXT_BACKEND"
  --model "$TEXT_MODEL"
  --data-dir "$DATA_DIR"
  --dataset "$DATASET_DIR"
  --clues-dir "$CLUES_DIR"
  --split "$SPLIT"
  --all
  --max-tokens "$MAX_TOKENS"
  --max-markdown-chars "$MAX_MARKDOWN_CHARS"
  --temperature "$TEMPERATURE"
  --batch-size "$BATCH_SIZE"
  --start-index "$START_INDEX"
  --device-map "$DEVICE_MAP"
  --dtype "$DTYPE"
)

if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  args+=(--attn-implementation "$ATTN_IMPLEMENTATION")
fi
if [[ -n "$END_INDEX" ]]; then
  args+=(--end-index "$END_INDEX")
fi
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

uv run dataset-generation describe-textual-clues "${args[@]}"
