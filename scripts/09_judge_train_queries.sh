#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-data}"
DATASET_DIR="${DATASET_DIR:-$DATA_DIR/preprocessed}"
SPLIT="${SPLIT:-train}"
SPLIT_INDEX="${SPLIT_INDEX:-$DATA_DIR/splits/document_split.json}"
COLLECTION_ID="${COLLECTION_ID:-query_generation_${SPLIT}}"
INPUT_DIR="${INPUT_DIR:-$DATA_DIR/query_collections/$COLLECTION_ID}"
JUDGEMENT_MODEL="${JUDGEMENT_MODEL:-google/gemma-3-27b-it}"

if [[ ! -f "$SPLIT_INDEX" ]]; then
  echo "Split index not found: $SPLIT_INDEX" >&2
  echo "Build it with: uv run python scripts/build_document_split.py --dataset $DATASET_DIR --output $SPLIT_INDEX" >&2
  exit 1
fi

args=(
  --dataset "$DATASET_DIR"
  --split-index "$SPLIT_INDEX"
  --split "$SPLIT"
  --input-dir "$INPUT_DIR"
  --model "$JUDGEMENT_MODEL"
)

if [[ -n "${MODE:-}" ]]; then
  args+=(--mode "$MODE")
else
  args+=(--mode visual-only --mode visual-and-text)
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ -n "${JUDGEMENT_MAX_TOKENS:-}" ]]; then
  args+=(--max-tokens "$JUDGEMENT_MAX_TOKENS")
fi
if [[ -n "${JUDGEMENT_TEMPERATURE:-}" ]]; then
  args+=(--temperature "$JUDGEMENT_TEMPERATURE")
fi
if [[ -n "${JUDGEMENT_DEVICE_MAP:-}" ]]; then
  args+=(--device-map "$JUDGEMENT_DEVICE_MAP")
fi
if [[ -n "${JUDGEMENT_DTYPE:-}" ]]; then
  args+=(--dtype "$JUDGEMENT_DTYPE")
fi
if [[ -n "${JUDGEMENT_ATTN_IMPLEMENTATION:-}" ]]; then
  args+=(--attn-implementation "$JUDGEMENT_ATTN_IMPLEMENTATION")
fi

uv run python scripts/judge_queries.py "${args[@]}"
