#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${QUERY_CONFIG:-configs/runs/query_generation_default.yaml}"
DATA_DIR="${DATA_DIR:-data}"
DATASET_DIR="${DATASET_DIR:-$DATA_DIR/preprocessed}"
CLUES_DIR="${CLUES_DIR:-$DATA_DIR/clues}"
SPLIT="${SPLIT:-train}"
SPLIT_INDEX="${SPLIT_INDEX:-$DATA_DIR/splits/document_split.json}"
COLLECTION_ID="${COLLECTION_ID:-query_generation_${SPLIT}_judged}"
JUDGEMENT_MODEL="${JUDGEMENT_MODEL:-google/gemma-3-27b-it}"

if [[ ! -f "$SPLIT_INDEX" ]]; then
  echo "Split index not found: $SPLIT_INDEX" >&2
  echo "Build it with: uv run python scripts/build_document_split.py --dataset $DATASET_DIR --output $SPLIT_INDEX" >&2
  exit 1
fi

args=(
  --config "$CONFIG"
  --dataset "$DATASET_DIR"
  --clues-dir "$CLUES_DIR"
  --split-index "$SPLIT_INDEX"
  --split "$SPLIT"
  --collection-id "$COLLECTION_ID"
  --judge-queries
)

if [[ -n "${MODE:-}" ]]; then
  args+=(--mode "$MODE")
else
  args+=(--mode visual-only --mode visual-and-text)
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "$START_INDEX")
fi
if [[ -n "${END_INDEX:-}" ]]; then
  args+=(--end-index "$END_INDEX")
fi
if [[ -n "${QUERY_MODEL_PROVIDER:-}" ]]; then
  args+=(--model-provider "$QUERY_MODEL_PROVIDER")
fi
if [[ -n "${QUERY_MODEL:-}" ]]; then
  args+=(--model "$QUERY_MODEL")
fi
if [[ -n "${QUERY_MAX_TOKENS:-}" ]]; then
  args+=(--max-tokens "$QUERY_MAX_TOKENS")
fi
if [[ -n "${QUERY_TEMPERATURE:-}" ]]; then
  args+=(--temperature "$QUERY_TEMPERATURE")
fi
args+=(--judgement-model "$JUDGEMENT_MODEL")
if [[ -n "${JUDGEMENT_MAX_TOKENS:-}" ]]; then
  args+=(--judgement-max-tokens "$JUDGEMENT_MAX_TOKENS")
fi
if [[ -n "${JUDGEMENT_TEMPERATURE:-}" ]]; then
  args+=(--judgement-temperature "$JUDGEMENT_TEMPERATURE")
fi
if [[ -n "${JUDGEMENT_DEVICE_MAP:-}" ]]; then
  args+=(--judgement-device-map "$JUDGEMENT_DEVICE_MAP")
fi
if [[ -n "${JUDGEMENT_DTYPE:-}" ]]; then
  args+=(--judgement-dtype "$JUDGEMENT_DTYPE")
fi
if [[ -n "${JUDGEMENT_ATTN_IMPLEMENTATION:-}" ]]; then
  args+=(--judgement-attn-implementation "$JUDGEMENT_ATTN_IMPLEMENTATION")
fi
if [[ -n "${COMPONENT_BUDGET:-}" ]]; then
  args+=(--component-budget "$COMPONENT_BUDGET")
fi
if [[ -n "${VISUAL_COMPONENT_BUDGET:-}" ]]; then
  args+=(--visual-component-budget "$VISUAL_COMPONENT_BUDGET")
fi
if [[ -n "${TEXTUAL_COMPONENT_BUDGET:-}" ]]; then
  args+=(--textual-component-budget "$TEXTUAL_COMPONENT_BUDGET")
fi
if [[ "${ALLOW_PARTIAL_COMPONENTS:-0}" == "1" ]]; then
  args+=(--allow-partial-components)
fi

uv run dataset-generation generate-queries "${args[@]}"
