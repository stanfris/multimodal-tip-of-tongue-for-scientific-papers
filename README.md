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
scripts/09_judge_train_queries.sh
```

Run `03_start_mineru_router.sh` in a separate terminal before
`04_run_mineru_full_extraction.sh`. The clue and query stages read directly
from `data/preprocessed` or `data/preprocessed/papers`.

The bash scripts are intentionally thin wrappers for the standard workflow.
Use the Python CLIs directly for ad hoc runs.

Build a fixed-seed stratified document split before clue/query generation.
The default split is 1,000 train documents and 200 test documents:

```bash
uv run python scripts/build_document_split.py \
  --dataset data/preprocessed \
  --output data/splits/document_split.json
```

Figure descriptions, textual clues, query generation, and query judgement are managed by
`configs/settings.yaml`. That file contains the overall dataset pointer,
dataset paths, split index, stage-specific prompt/model/generation settings,
and the standardized train/test query-set definitions. Runtime overrides for
managed query settings are intentionally not supported.

Generate the training set by default, or explicitly select the test set:

```bash
scripts/06_describe_all_figures.sh
scripts/07_describe_all_textual_clues.sh
scripts/08_generate_queries.sh
scripts/09_judge_train_queries.sh
scripts/06_describe_all_figures.sh --set test
scripts/07_describe_all_textual_clues.sh --set test
scripts/08_generate_queries.sh --set test
scripts/09_judge_train_queries.sh --set test
```

The split index stores ordered paper IDs. For standard full-run behavior, edit
`visual_descriptions`, `textual_descriptions`, `visual_query`, or
`visual_query.judgement` in `configs/settings.yaml`.

The clue-generation scripts write per-paper clue files:

```text
data/clues/<paper_id>/base/textual_clues.jsonl
data/clues/<paper_id>/images/<figure_id>.jsonl
```

The full-run scripts are single-process and do not launch multi-GPU workers.
They resume by default, write clues incrementally, and record failures under
`data/clues/`.

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

On the DGX, start a persistent router:

```bash
scripts/03_start_mineru_router.sh
```

Use `uv run mineru-router --help` directly for custom router settings.

Run extraction against the downloaded ACL subset PDFs:

```bash
scripts/04_run_mineru_full_extraction.sh
```

Use `uv run dataset-generation extract-mineru-pdfs --help` directly for custom
extraction settings.

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
scripts/06_describe_all_figures.sh
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
scripts/07_describe_all_textual_clues.sh
```

This writes textual clues to `data/clues/<paper_id>/base/textual_clues.jsonl`.

Print stored coverage statistics:

```bash
uv run dataset-generation stats --dataset data/query_collections/query_generation_train/visual_only
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
