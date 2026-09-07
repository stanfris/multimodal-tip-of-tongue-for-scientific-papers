#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/acl_subset/pdfs}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/mineru_pdf_extraction}"
API_URL="${API_URL:-http://127.0.0.1:8002}"
BACKEND="${BACKEND:-hybrid-engine}"
EFFORT="${EFFORT:-medium}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
RESULT_TIMEOUT="${RESULT_TIMEOUT:-3600}"
RETRIES="${RETRIES:-2}"
MIN_MARKDOWN_CHARS="${MIN_MARKDOWN_CHARS:-200}"
START_PAGE_ID="${START_PAGE_ID:-0}"
END_PAGE_ID="${END_PAGE_ID:-9}"

mkdir -p "$OUTPUT_DIR/logs"
LOG_PATH="$OUTPUT_DIR/logs/full_run_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "input_dir=$INPUT_DIR"
  echo "output_dir=$OUTPUT_DIR"
  echo "api_url=$API_URL"
  echo "backend=$BACKEND"
  echo "effort=$EFFORT"
  echo "max_in_flight=$MAX_IN_FLIGHT"
  echo "start_page_id=$START_PAGE_ID"
  echo "end_page_id=$END_PAGE_ID"

  uv run dataset-generation extract-mineru-pdfs \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --api-url "$API_URL" \
    --backend "$BACKEND" \
    --effort "$EFFORT" \
    --max-in-flight "$MAX_IN_FLIGHT" \
    --poll-interval "$POLL_INTERVAL" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --result-timeout "$RESULT_TIMEOUT" \
    --retries "$RETRIES" \
    --min-markdown-chars "$MIN_MARKDOWN_CHARS" \
    --start-page-id "$START_PAGE_ID" \
    --end-page-id "$END_PAGE_ID"

  echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG_PATH"
