#!/usr/bin/env bash
set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DATASET="${REPO_ROOT}/data/canonical/papers.jsonl"
OUTPUT_DIR="${REPO_ROOT}/data/canonical/pdfs"

echo "Downloading ACL Anthology PDFs..."
echo "Input: ${INPUT_DATASET}"
echo "Output Directory: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

"${REPO_ROOT}/.venv/bin/python" -m dataset_generation.download_acl_pdfs \
    --input "${INPUT_DATASET}" \
    --output-dir "${OUTPUT_DIR}"
