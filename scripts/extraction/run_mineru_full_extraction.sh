#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "${1:-}" in
  --input-dir|--input-dir=*|--output-dir|--output-dir=*|--split|--split=*|\
  --split-index|--split-index=*|--api-url|--api-url=*|--domains|--domains=*|\
  --limit|--limit=*|--max-in-flight|--max-in-flight=*|--all-domain-pdfs|\
  --retry-incomplete-only|--help|-h)
    UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" exec uv run --no-sync dataset-generation extract-mineru-pdfs \
      --input-dir data/pdf_datasets --output-dir data/processed "$@"
    ;;
esac

UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" exec uv run --no-sync dataset-generation "$@" stage=extract_mineru
