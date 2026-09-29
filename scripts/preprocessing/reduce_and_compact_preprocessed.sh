#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CORPUS_ROOT=""
PREPROCESSED_DIR=""
PREPROCESSED_DIR_EXPLICIT=false
PDF_DIR=""
SPLIT_INDEX=""
SPLIT="train+test"
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
            PREPROCESSED_DIR_EXPLICIT=true
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --preprocessed-dir=*)
            PREPROCESSED_DIR="${1#*=}"
            PREPROCESSED_DIR_EXPLICIT=true
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
        --split-index)
            SPLIT_INDEX="${2:?--split-index requires a path}"
            shift 2
            ;;
        --split-index=*)
            SPLIT_INDEX="${1#*=}"
            shift
            ;;
        --split)
            SPLIT="${2:?--split requires a value}"
            shift 2
            ;;
        --split=*)
            SPLIT="${1#*=}"
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

if [[ -n "$CORPUS_ROOT" ]]; then
    [[ -n "$PDF_DIR" ]] || PDF_DIR="$CORPUS_ROOT/pdf_datasets"
    if [[ "$PREPROCESSED_DIR_EXPLICIT" == false && -z "$SPLIT_INDEX" && -f "$CORPUS_ROOT/data/splits/pdf_dataset_split.json" ]]; then
        SPLIT_INDEX="$CORPUS_ROOT/data/splits/pdf_dataset_split.json"
    fi
fi

if [[ -n "$SPLIT_INDEX" ]]; then
    FORWARDED_ARGS+=(--split-index "$SPLIT_INDEX" --split "$SPLIT")
    COMPACT_FORWARD_ARGS+=(--split-index "$SPLIT_INDEX" --split "$SPLIT")
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
