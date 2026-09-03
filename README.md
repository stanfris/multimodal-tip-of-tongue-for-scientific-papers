# ACL Figure Markdown Dataset Generation

This repository builds a persisted research dataset that combines figure records from
[`citeseerx/ACL-fig`](https://huggingface.co/datasets/citeseerx/ACL-fig) with matching paper markdown from
[`KRLabsOrg/acl-anthology-md`](https://huggingface.co/datasets/KRLabsOrg/acl-anthology-md), config `fulltext`.

The matcher extracts ACL Anthology IDs from ACL-fig image filenames, normalizes IDs, applies known proceedings-volume fallback suffixes, and enriches each figure record with the corresponding paper markdown when available.

## Repository Layout

```text
src/dataset_generation/
  sources.py     # Hugging Face dataset loading and ACL-fig record extraction
  matching.py    # ID normalization, filename extraction, matching logic
  schema.py      # Output schema validation
  build.py       # Reproducible build orchestration
  storage.py     # Artifact writing and reading
  synthetic.py   # Stable interfaces for later test collection generation
  cli.py         # dataset-generation command line interface
```

Generated data is intentionally ignored by Git:

```text
data/raw/hf/                              # local Hugging Face source cache
data/interim/interpretations/<run_id>/    # generated visual/textual sidecars
data/processed/acl_fig_markdown/<split>/  # compact joined base dataset
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

This syncs the environment, downloads source datasets, builds the processed and
canonical artifacts, generates visual/textual canonical clue files, and
generates query collections. Set `RUN_VISUAL=0`, `RUN_TEXTUAL=0`, or
`RUN_QUERIES=0` to skip the expensive model-backed stages.

Or run individual steps:

```bash
scripts/00_sync_env.sh
scripts/01_download_sources.sh
scripts/02_build_dataset.sh
scripts/03_build_canonical.sh
scripts/04_describe_all_figures.sh
scripts/05_describe_all_textual_clues.sh
scripts/06_generate_queries.sh
```

`02_build_dataset.sh` writes the processed figure-row artifact under
`data/processed/acl_fig_markdown/<split>/`. `03_build_canonical.sh` then builds
the canonical paper-level artifact under `data/canonical/`, which is the default
input for clue and query generation.

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

## Build The Dataset

Download the source datasets once into the local data folder:

```bash
uv run dataset-generation download-sources --data-dir data --split train --paper-split train
```

Build the default train split from local sources when present and write a Parquet artifact:

```bash
uv run dataset-generation build --data-dir data
```

Build a specific split with an explicit output format:

```bash
uv run dataset-generation build \
  --split train \
  --output data/processed/acl_fig_markdown/train \
  --format parquet
```

Run a small debug build without scanning the full markdown stream:

```bash
uv run dataset-generation build \
  --split train \
  --limit 100 \
  --paper-limit 1000 \
  --output data/processed/debug_acl_fig_markdown \
  --format jsonl
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
MAX_IN_FLIGHT=4 scripts/09_run_mineru_full_extraction.sh
```

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

## Managed Runs

Config-driven runs keep dataset, parser, prompt, model, and output settings in one YAML file:

```bash
uv run dataset-generation run --config configs/runs/acl_fig_markdown_smoke.yaml
```

Example configs live under `configs/runs/`. Relative paths in a run config are resolved from the config file location, so prompt templates and outputs remain portable when the project moves. A managed run writes the normal dataset artifact plus provenance files:

```text
data/processed/acl_fig_markdown_smoke/
  data.jsonl
  metadata.json
  stats.json
  resolved_config.yaml
  resolved_config.json
  prompt.txt
```

The first manager implementation uses a parser registry and a model-adapter registry. The default `acl_fig_markdown` parser wraps the existing build pipeline, and the initial `"null"` model adapter validates config without making model calls.

## Figure Descriptions

Generate Qwen-VL visual descriptions from the processed dataset artifact:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

This reads image references from `data/processed/acl_fig_markdown/<split>/data.*`,
so visual sidecars use the same `record_id` values as textual sidecars. You can
still pass explicit local images for ad hoc checks:

```bash
uv run dataset-generation describe-figures \
  --backend transformers \
  --image /path/to/figure.png \
  --num-samples 1
```

Outputs are written under `data/interim/interpretations/qwen3_vl_figure_description/` by default.

## Textual Clues

Generate semantic memory cues from the matched paper markdown in the same processed dataset artifact:

```bash
uv run dataset-generation describe-textual-clues \
  --backend transformers \
  --data-dir data \
  --split train \
  --num-samples 1
```

Outputs are written under `data/interim/interpretations/qwen3_textual_clue_description/` by default with `kind=textual`.

Print stored coverage statistics:

```bash
uv run dataset-generation stats --dataset data/processed/acl_fig_markdown/train
```

## Artifact Structure

Each build writes:

```text
data/processed/acl_fig_markdown/train/
  data.parquet      # or data.jsonl / data.csv
  images/           # ACL-Fig images keyed by record_id
  metadata.json     # build timestamp, source names/configs/splits, seed, build config
  stats.json        # match coverage statistics
```

Rows include the matched paper `markdown` plus `image_relpath` and `image_path`
references when images are materialized. Use `--no-images` on `dataset-generation build`
to skip writing the image files.

The base dataset should stay compact and canonical: figure/document metadata, stable IDs, labels, and matched markdown. Model-generated descriptions belong in sidecar artifacts keyed by `record_id`:

```text
data/interim/interpretations/<run_id>/
  interpretations.jsonl  # record_id, kind=visual|textual, text, model, prompt_id, prompt_version
  metadata.json          # run-level provenance
```

The current default pipeline writes canonical clue sidecars directly next to
`data/canonical/papers.jsonl`:

```text
data/canonical/textual_clues.jsonl
data/canonical/visual_clues.jsonl
```

Prompt templates live in `prompts/` and are referenced by ID/version from sidecar metadata. Later query generation should join the base dataset with one or more interpretation sidecars, then write query collections under `data/query_collections/<collection_id>/`.

## Output Schema

Required fields:

| Field | Description |
| --- | --- |
| `record_id` | Stable SHA-1 ID derived from split, index, and filename |
| `filename` | ACL-fig image filename/path |
| `extracted_paper_id` | Paper ID extracted from the filename |
| `normalized_paper_id` | Normalized lookup key |
| `resolved_paper_id` | Matched markdown paper ID, including fallback suffix when used |
| `label` | ACL-fig label/class value |
| `markdown` | Matched paper markdown, or empty string when unmatched |
| `match_status` | `matched` or `unmatched` |
| `source_fig_dataset` | Figure source dataset name |
| `source_fig_split` | Figure source split |
| `source_paper_dataset` | Markdown source dataset name |
| `source_paper_config` | Markdown source config |

Additional image metadata fields include `image_width`, `image_height`, and `image_mode` when available.

## Matching Behavior

The matching code preserves the prototype behavior:

- Normalize URLs, trailing `.pdf`, `.dataset`, casing, whitespace, and trailing version suffixes such as `v2`.
- Extract paper IDs from filenames like `2007.sigdial-1.12.pdf-Figure4.png`.
- Try direct normalized-ID matches first.
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
