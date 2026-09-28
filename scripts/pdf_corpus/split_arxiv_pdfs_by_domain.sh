#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

exec uv run python -m pdf_corpus.split_arxiv_pdfs_by_domain "$@"
