#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${QUERY_CONFIG:-configs/runs/query_generation_default.yaml}"

args=(--config "$CONFIG")

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
if [[ -n "${MODE:-}" ]]; then
  args+=(--mode "$MODE")
fi
if [[ -n "${COLLECTION_ID:-}" ]]; then
  args+=(--collection-id "$COLLECTION_ID")
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
if [[ -n "${COMPONENT_BUDGET:-}" ]]; then
  args+=(--component-budget "$COMPONENT_BUDGET")
fi
if [[ -n "${VISUAL_COMPONENT_BUDGET:-}" ]]; then
  args+=(--visual-component-budget "$VISUAL_COMPONENT_BUDGET")
fi
if [[ -n "${TEXTUAL_COMPONENT_BUDGET:-}" ]]; then
  args+=(--textual-component-budget "$TEXTUAL_COMPONENT_BUDGET")
fi
if [[ -n "${MAX_TEXT_COMPONENTS:-}" ]]; then
  args+=(--max-text-components "$MAX_TEXT_COMPONENTS")
fi
if [[ "${ALLOW_PARTIAL_COMPONENTS:-0}" == "1" ]]; then
  args+=(--allow-partial-components)
fi

uv run dataset-generation generate-queries "${args[@]}"
