# ACL Set Dataset Generation

This repository builds a persisted research dataset from a curated ACL
Anthology subset. The current path starts from official ACL Anthology XML
metadata, downloads the selected PDFs, extracts markdown and figures with
MinerU, then generates clue and query artifacts directly from the preprocessed
paper directories.

## Repository Layout

```text
src/dataset_generation/
  schema.py      # Output schema validation
  acl_subset.py  # ACL Anthology subset metadata and PDF downloads
  preprocessed.py # Direct reader for extracted paper directories and clues
  storage.py     # Artifact writing and reading
  synthetic.py   # Stable interfaces for later test collection generation
  cli.py         # dataset-generation command line interface
```

Generated data is intentionally ignored by Git:

```text
data/acl_subset/                          # curated ACL metadata and PDFs
data/processed/mineru_pdf_extraction/     # extracted PDF markdown and figures
data/clues/                               # per-paper textual and visual clues
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

The reviewer can inspect generated query collections and writes manual
annotations separately to `data/reviews/query_review_annotations.jsonl`.

## Bash Scripts

Run the dataset generation stages individually:

```bash
scripts/00_sync_env.sh
scripts/01_build_acl_subset.sh
scripts/02_download_acl_pdfs.sh
scripts/03_start_mineru_router.sh
scripts/04_run_mineru_full_extraction.sh
scripts/05_reduce_and_compact_preprocessed.sh
scripts/06_describe_all_figures.sh
scripts/07_describe_all_textual_clues.sh
scripts/08_generate_queries.sh
```

Run `03_start_mineru_router.sh` in a separate terminal before
`04_run_mineru_full_extraction.sh`. The clue and query stages read directly
from `data/preprocessed` or `data/preprocessed/papers`.

The scripts use `DATA_DIR=data` and `SPLIT=train` by default. Override them as
environment variables, for example:

```bash
LIMIT=100 scripts/07_describe_all_textual_clues.sh
```

Build a fixed-seed stratified document split before clue/query generation:

```bash
uv run python scripts/build_document_split.py \
  --dataset data/preprocessed \
  --output data/splits/document_split_seed42.json
```

Then run a specific set by passing the split index and `SPLIT=train` or
`SPLIT=test`:

```bash
SPLIT_INDEX=data/splits/document_split_seed42.json SPLIT=train scripts/06_describe_all_figures.sh
SPLIT_INDEX=data/splits/document_split_seed42.json SPLIT=train scripts/07_describe_all_textual_clues.sh
SPLIT_INDEX=data/splits/document_split_seed42.json SPLIT=train COLLECTION_ID=query_generation_train scripts/08_generate_queries.sh
SPLIT_INDEX=data/splits/document_split_seed42.json SPLIT=test COLLECTION_ID=query_generation_test scripts/08_generate_queries.sh
```

The split index stores ordered paper IDs. `START_INDEX`, `END_INDEX`, `LIMIT`,
`RESUME`, and `OVERWRITE` apply after the split is selected, so restarts and
partial reruns operate on the chosen train/test set instead of the whole corpus.

The clue-generation scripts write per-paper clue files:

```text
data/clues/<paper_id>/base/textual_clues.jsonl
data/clues/<paper_id>/images/<figure_id>.jsonl
```

The full-run scripts are single-process and do not launch multi-GPU workers.
They default to `RESUME=1`, write clues incrementally, and record failures under
`data/clues/`. They use PyTorch SDPA attention by default; set
`ATTN_IMPLEMENTATION=` to let Transformers choose automatically. Useful
overrides:

```bash
TEXT_BATCH_SIZE=16 LIMIT=1000 scripts/07_describe_all_textual_clues.sh
VISUAL_BATCH_SIZE=1 START_INDEX=5000 scripts/06_describe_all_figures.sh
ATTN_IMPLEMENTATION= scripts/07_describe_all_textual_clues.sh
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
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/03_start_mineru_router.sh
```

The script defaults to `uv run mineru-router --local-gpus auto`, preloads VLM
workers, uses `MINERU_API_MAX_CONCURRENT_REQUESTS=1` per worker, and exposes the
MinerU API at `http://127.0.0.1:8002`. It writes MinerU service output under
`data/mineru_api_output` unless `MINERU_API_OUTPUT_ROOT` is overridden.

Run extraction against the downloaded ACL subset PDFs:

```bash
MAX_IN_FLIGHT=4 START_PAGE_ID=0 END_PAGE_ID=9 scripts/04_run_mineru_full_extraction.sh
```

By default the full-run script parses only page IDs `0..9`, i.e. the first 10
pages. Override `START_PAGE_ID` and `END_PAGE_ID` for a different slice, or use
`END_PAGE_ID=99999` to request the whole PDF.

Completed papers are skipped on restart. Each successful paper has:

```text
data/preprocessed/papers/<paper_id>/
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

Generate Qwen-VL visual descriptions from preprocessed ACL subset papers:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

This reads figure image references from
`data/preprocessed` and writes visual clues to
`data/clues/<paper_id>/images/<figure_id>.jsonl`. You can still pass explicit
local images for ad hoc checks:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --image /path/to/figure.png \
  --num-samples 1
```

Use `--clues-dir` to write to a different clue directory.

## Textual Clues

Generate semantic memory cues from preprocessed paper markdown:

```bash
uv run dataset-generation describe-textual-clues \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

This writes textual clues to `data/clues/<paper_id>/base/textual_clues.jsonl`.

Print stored coverage statistics:

```bash
uv run dataset-generation stats --dataset data/query_collections/query_generation_default/visual_only
```

## Artifact Structure

Preprocessing and clue generation write:

```text
data/preprocessed/papers/<paper_id>/
  markdown.md
  paper.json
  figures.json
  mineru/

data/clues/<paper_id>/
  base/textual_clues.jsonl
  images/<figure_id>.jsonl

data/clues/queries.jsonl
```

Prompt templates live in `prompts/` and are referenced by ID/version from
sidecar metadata. Query generation joins preprocessed papers with per-paper
clue files and writes query collections under
`data/query_collections/<collection_id>/`.

## Output Schema

Each preprocessed paper directory must contain:

| Field | Description |
| --- | --- |
| `paper.json` | Paper metadata, including `paper_id` and extracted figures |
| `markdown.md` | Extracted full-paper markdown |
| `figures.json` | Extracted figure metadata |
| `mineru/` | MinerU output files and image assets |

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
