#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CORPUS_ROOT=""
PREPROCESSED_DIR=""
PDF_DIR=""
FORWARDED_ARGS=()
COMPACT_FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-dir)
            CORPUS_ROOT="${2:?--root-dir requires a path}"
            shift 2
            ;;
        --root-dir=*)
            CORPUS_ROOT="${1#*=}"
            shift
            ;;
        --preprocessed-dir)
            PREPROCESSED_DIR="${2:?--preprocessed-dir requires a path}"
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --preprocessed-dir=*)
            PREPROCESSED_DIR="${1#*=}"
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        --pdf-dir)
            PDF_DIR="${2:?--pdf-dir requires a path}"
            shift 2
            ;;
        --pdf-dir=*)
            PDF_DIR="${1#*=}"
            shift
            ;;
        --dry-run|--fail-fast)
            FORWARDED_ARGS+=("$1")
            COMPACT_FORWARD_ARGS+=("$1")
            shift
            ;;
        --workers|--progress-every)
            FORWARDED_ARGS+=("$1" "${2:?$1 requires a value}")
            COMPACT_FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --workers=*|--progress-every=*)
            FORWARDED_ARGS+=("$1")
            COMPACT_FORWARD_ARGS+=("$1")
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$PREPROCESSED_DIR" ]]; then
    if [[ -n "$CORPUS_ROOT" ]]; then
        PREPROCESSED_DIR="$CORPUS_ROOT/processed"
    else
        PREPROCESSED_DIR="data/preprocessed"
    fi
    FORWARDED_ARGS=(--preprocessed-dir "$PREPROCESSED_DIR" "${FORWARDED_ARGS[@]}")
fi

COMPACT_ARGS=(--preprocessed-dir "$PREPROCESSED_DIR" "${COMPACT_FORWARD_ARGS[@]}")
if [[ -n "$CORPUS_ROOT" ]]; then
    COMPACT_ARGS+=(--root-dir "$CORPUS_ROOT")
fi
if [[ -n "$PDF_DIR" ]]; then
    COMPACT_ARGS+=(--pdf-dir "$PDF_DIR")
fi

uv run python -m preprocessing.reduce_preprocessed_collection "${FORWARDED_ARGS[@]}"
uv run python -m preprocessing.compact_preprocessed_collection "${COMPACT_ARGS[@]}"
