# ACL Set Dataset Generation

This repository builds a persisted research dataset from a curated ACL
Anthology subset. The current path starts from official ACL Anthology XML
metadata, downloads the selected PDFs, extracts markdown and figures with
MinerU, and writes a canonical paper-level dataset for clue and query
generation.

## Repository Layout

```text
src/dataset_generation/
  schema.py      # Output schema validation
  acl_subset.py  # ACL Anthology subset metadata and PDF downloads
  canonical.py   # Canonical paper/figure/clue representation
  storage.py     # Artifact writing and reading
  synthetic.py   # Stable interfaces for later test collection generation
  cli.py         # dataset-generation command line interface
```

Generated data is intentionally ignored by Git:

```text
data/acl_subset/                          # curated ACL metadata and PDFs
data/processed/mineru_pdf_extraction/     # extracted PDF markdown and figures
data/canonical/                           # canonical papers, figures, clues, queries
data/query_collections/<id>/              # generated query/eval collections
```

## Installation

Use `uv` from this directory:

```bash
uv sync --dev
```

## Query Review UI

Launch the local Streamlit reviewer for generated tip-of-the-tongue query collections:

```bash
uv run streamlit run review_app.py --server.address 127.0.0.1
```

The reviewer defaults to `data/canonical`, joins generated examples with the
canonical paper, figure, markdown, visual clue, and textual clue artifacts, and
writes manual annotations separately to
`data/reviews/query_review_annotations.jsonl`.

## Bash Scripts

Run the full dataset generation pipeline from bash:

```bash
scripts/build_everything_from_scratch.sh
```

This syncs the environment, builds the ACL subset metadata, downloads PDFs,
runs MinerU extraction, builds the canonical artifact, generates visual/textual
canonical clue files, and generates query collections. Set `RUN_VISUAL=0`,
`RUN_TEXTUAL=0`, or `RUN_QUERIES=0` to skip the expensive model-backed stages.

Or run individual steps:

```bash
scripts/00_sync_env.sh
scripts/01_build_acl_subset.sh
scripts/02_download_acl_pdfs.sh
scripts/08_start_mineru_router.sh
scripts/09_run_mineru_full_extraction.sh
scripts/03_build_canonical.sh
scripts/04_describe_all_figures.sh
scripts/05_describe_all_textual_clues.sh
scripts/06_generate_queries.sh
```

Run `08_start_mineru_router.sh` in a separate terminal before
`09_run_mineru_full_extraction.sh`. `03_build_canonical.sh` builds the
canonical paper-level artifact under `data/canonical/` from
`data/processed/mineru_pdf_extraction/papers/`, which is the default input for
clue and query generation.

The scripts use `DATA_DIR=data` and `SPLIT=train` by default. Override them as
environment variables, for example:

```bash
LIMIT=100 scripts/05_describe_all_textual_clues.sh
```

The clue-generation scripts write to the canonical clue files:

```text
data/canonical/visual_clues.jsonl
data/canonical/textual_clues.jsonl
```

The full-run scripts are single-process and do not launch multi-GPU workers.
They default to `RESUME=1`, write clues incrementally, and record failures under
`data/canonical/`. They use PyTorch SDPA attention by default; set
`ATTN_IMPLEMENTATION=` to let Transformers choose automatically. Useful
overrides:

```bash
TEXT_BATCH_SIZE=16 LIMIT=1000 scripts/05_describe_all_textual_clues.sh
VISUAL_BATCH_SIZE=1 START_INDEX=5000 scripts/04_describe_all_figures.sh
ATTN_IMPLEMENTATION= scripts/05_describe_all_textual_clues.sh
```

## ACL Anthology Subset

Build the curated recent full-paper ACL Anthology subset from official
Anthology XML metadata:

```bash
python scripts/build_acl_subset.py --metadata-only
```

This writes `data/acl_subset/papers.jsonl` plus build metadata. The target
volumes are explicitly configured in `dataset_generation.acl_subset.TARGET_VOLUMES`
and are not selected by fuzzy venue/title matching.

Download the corresponding PDFs after writing metadata:

```bash
python scripts/build_acl_subset.py --download-pdfs
```

PDF downloads run concurrently and show completed/total, skipped, failed, rate,
and ETA. Tune concurrency with `--max-workers`; the default is `16`.

## MinerU PDF Extraction

The full-paper PDF extraction path uses MinerU 3.x through a persistent
`mineru-api` or `mineru-router` service. This avoids paying model startup cost
once per PDF and keeps corpus-level resume state in this repository instead of
depending on MinerU's in-process task IDs.

First inspect the machine:

```bash
uv run dataset-generation probe-mineru-env
```

On the DGX, start a persistent router. Limit GPUs with `CUDA_VISIBLE_DEVICES`
when needed; do not hardcode GPU IDs in the extraction command.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/08_start_mineru_router.sh
```

The script defaults to `uv run mineru-router --local-gpus auto`, preloads VLM
workers, uses `MINERU_API_MAX_CONCURRENT_REQUESTS=1` per worker, and exposes the
MinerU API at `http://127.0.0.1:8002`. It writes MinerU service output under
`data/mineru_api_output` unless `MINERU_API_OUTPUT_ROOT` is overridden.

Run extraction against the downloaded ACL subset PDFs:

```bash
MAX_IN_FLIGHT=4 START_PAGE_ID=0 END_PAGE_ID=9 scripts/09_run_mineru_full_extraction.sh
```

By default the full-run script parses only page IDs `0..9`, i.e. the first 10
pages. Override `START_PAGE_ID` and `END_PAGE_ID` for a different slice, or use
`END_PAGE_ID=99999` to request the whole PDF.

Completed papers are skipped on restart. Each successful paper has:

```text
data/processed/mineru_pdf_extraction/papers/<paper_id>/
  _SUCCESS
  markdown.md
  paper.json
  figures.json
  mineru/                  # MinerU Markdown, content_list JSON, images
```

`paper.json` records the source PDF, MinerU configuration/version, Markdown
path, page count if available, and normalized `image`/`chart` figure records
with page, bounding box, caption, footnote, and image paths. Failed PDFs are
appended to `data/processed/mineru_pdf_extraction/failures.jsonl` and do not
stop the batch. Full-run logs are written under
`data/processed/mineru_pdf_extraction/logs/`, and the latest run configuration
and summary are saved as `run_config.json` and `last_run_summary.json`.

Before a full run, benchmark medium-effort throughput on a small sample:

```bash
uv run dataset-generation benchmark-mineru-pdfs \
  --input-dir data/acl_subset/pdfs \
  --sample-size 20 \
  --api-url http://127.0.0.1:8002 \
  --max-in-flight-values 1 2 4 8
```

For this project, use `hybrid-engine --effort medium`: current MinerU
documentation states that hybrid medium is the default fast path and skips
expensive image/chart analysis, while still returning extracted image/chart
blocks when available. The benchmark writes a `benchmark_report.json` with a
recommended `max_in_flight` starting point.

The extraction client requests only production outputs: Markdown, images, and
content list JSON. It explicitly skips the original PDF, middle JSON, and model
output in MinerU responses.

## Figure Descriptions

Generate Qwen-VL visual descriptions from the canonical ACL subset artifact:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

This reads figure image references from `data/canonical/papers.jsonl` and writes
canonical visual clues to `data/canonical/visual_clues.jsonl`. You can still
pass explicit local images for ad hoc checks:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --image /path/to/figure.png \
  --num-samples 1
```

Use `--output-dir` only when intentionally writing a separate interpretation
artifact instead of the canonical clue sidecar.

## Textual Clues

Generate semantic memory cues from canonical paper markdown:

```bash
uv run dataset-generation describe-textual-clues \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

This writes canonical textual clues to `data/canonical/textual_clues.jsonl`.

Print stored coverage statistics:

```bash
uv run dataset-generation stats --dataset data/canonical
```

## Artifact Structure

Canonical builds write:

```text
data/canonical/
  papers.jsonl
  markdown/
  images/
  pdfs/
  build_report.json
  textual_clues.jsonl
  visual_clues.jsonl
  queries.jsonl
```

The base dataset stays compact and canonical: paper metadata, markdown
references, figure image references, source PDF references, and source
provenance. Model-generated descriptions are sidecars keyed by canonical
`paper_id` and `figure_id`:

```text
data/canonical/textual_clues.jsonl
data/canonical/visual_clues.jsonl
```

Prompt templates live in `prompts/` and are referenced by ID/version from
sidecar metadata. Query generation joins canonical papers with clue sidecars and
writes query collections under `data/query_collections/<collection_id>/`.

## Output Schema

Each `papers.jsonl` row contains:

| Field | Description |
| --- | --- |
| `paper_id` | ACL Anthology paper ID |
| `markdown_relpath` | Relative path to canonical markdown |
| `markdown_sha256` | SHA-256 of the canonical markdown file |
| `source` | ACL subset, PDF, and extraction provenance |
| `figures` | Canonical figures with `figure_id`, `image_relpath`, `image_sha256`, and metadata |
| `queries` | Reserved query grouping fields |
- Try fallback suffixes in order: `.0`, `00`, `000`, `0`.

## Synthetic Test Collections

`dataset_generation.synthetic` defines stable interfaces for later retrieval/evaluation dataset generation:

- `TestCollectionExample`
- `SyntheticCollectionConfig`
- `generate_synthetic_queries(records, config)`
- `write_test_collection(collection, output_dir)`

The current implementation is deterministic and template-based. It deliberately does not hard-code an LLM provider; future model-backed generation should be added behind an adapter interface.

## Tests

Run fixture-based tests without downloading full datasets:

```bash
uv run pytest
```
