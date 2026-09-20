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
  document_downloads/ # Source-document metadata builders and PDF download commands
  preprocessed.py # Direct reader for extracted paper directories and clues
  storage.py     # Artifact writing and reading
  synthetic.py   # Stable interfaces for later test collection generation
  cli.py         # dataset-generation command line interface
```

Generated data is intentionally ignored by Git:

```text
data/acl_subset/                          # curated ACL metadata and PDFs
data/arxiv_open_reuse/                    # arXiv CC BY/CC0 metadata, report, PDFs
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
.venv/bin/python -m streamlit run review_app.py --server.address 127.0.0.1
```

The reviewer can inspect generated query collections and writes manual
annotations separately to `data/reviews/query_review_annotations.jsonl`.

## Bash Scripts

Run the dataset generation stages individually:

```bash
scripts/00_sync_env.sh
scripts/document_downloads/build_acl_subset.sh
scripts/document_downloads/download_acl_pdfs.sh
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

All source-document PDF downloads are centralized under
`src/dataset_generation/document_downloads/` and exposed through:

```bash
uv run dataset-generation download-documents --help
```

Thin shell wrappers for each document corpus live under
`scripts/document_downloads/`.

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
managed query settings are intentionally not supported, except for the
`--limit` cap on total queries per mode, including existing queries when resuming.

Generate the training set by default, or explicitly select the test set:

```bash
scripts/06_describe_all_figures.sh
scripts/07_describe_all_textual_clues.sh
scripts/08_generate_queries.sh
scripts/09_judge_train_queries.sh
scripts/06_describe_all_figures.sh --set test
scripts/07_describe_all_textual_clues.sh --set test
scripts/08_generate_queries.sh --set test
scripts/08_generate_queries.sh --set test --limit 3
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
scripts/document_downloads/build_acl_subset.sh
```

This writes `data/acl_subset/papers.jsonl` plus build metadata. The target
volumes are explicitly configured in
`dataset_generation.document_downloads.acl_subset.TARGET_VOLUMES`
and are not selected by fuzzy venue/title matching.

Download the corresponding PDFs after writing metadata:

```bash
uv run dataset-generation download-documents acl
# or
scripts/document_downloads/download_acl_pdfs.sh
```

PDF downloads run concurrently and show completed/total, skipped, failed, rate,
and ETA. Tune concurrency with `--max-workers`; the default is `16`.

## PMC OA Strict Biology/Medical Subset

Build a PubMed Central Open Access Subset dataset with strict article-version
license gating. By default this scans publication years 2024, 2025, and 2026:

```bash
uv run dataset-generation build-pmc-oa-subset \
  --metadata-only \
  --email you@example.org
# or
scripts/document_downloads/build_pmc_oa_subset.sh --email you@example.org
```

The discovery query defaults to:

```text
("cc by license"[filter] OR "cc0 license"[filter]) AND open_access[filter] AND has_pdf[filter] NOT pmc embargo[filter]
```

The builder uses only official NCBI/PMC infrastructure: NCBI E-Utilities for
PMC discovery and PubMed metadata, and the PMC Article Datasets AWS Open Data
bucket for article-version JSON metadata and PDF acquisition. It does not scrape
article pages. Each candidate must pass the AWS JSON `license_code` check as
exactly `cc by` or `cc0` before its PDF URL is ever queued for download.
Long runs print progress for E-Utilities date-range discovery, AWS
article-version license checks, PubMed enrichment batches, accepted domain
counts, and PDF downloads. AWS article-version checks are parallelized
independently from PDF downloads; tune them with `--aws-metadata-workers`
(default `64`). PubMed enrichment now runs only for current candidate batches
until the default target of `30,000` Biology and `30,000` Medical/Clinical
records is met; tune each enrichment batch with `--pubmed-batch-size` (default
`200`) and tune PDF downloads with `--max-workers` (default `16`).

After auditing `data/pmc_oa_strict/report.json` or `report.md`, download the
eligible PDFs:

```bash
uv run dataset-generation download-documents pmc-oa --max-workers 32
# or
scripts/document_downloads/download_pmc_oa_pdfs.sh --max-workers 32
```

Useful smoke test:

```bash
uv run dataset-generation build-pmc-oa-subset \
  --metadata-only \
  --start-year 2024 \
  --end-year 2024 \
  --max-records 100 \
  --email you@example.org
```

Outputs are written under `data/pmc_oa_strict/`:

```text
papers.jsonl     # full metadata manifest with PMCID, PMID, DOI, subjects, license, PDF source/path
discovered_pmcids.jsonl # incrementally saved E-Utilities discovery results
discovery_ranges.jsonl  # completed/partial date-range checkpoints
aws_metadata.jsonl      # cached PMC AWS license/PDF eligibility decisions
pubmed_metadata.jsonl   # cached PubMed XML-derived subject metadata
manifest.csv     # compact tabular manifest
report.json      # counts by license, domain, year, and subject
report.md        # human-readable summary
pdfs/            # optional Biology and Medical_Clinical_Research PDF folders
```

Rerunning the same command resumes from these sidecars: completed discovery
ranges are skipped, confirmed AWS metadata decisions are reused, failed AWS
metadata checks are retried, cached PubMed metadata is reused, and existing
valid PDFs are skipped.

The final report explicitly states whether the strict CC BY/CC0 policy produced
at least 20,000 Biology papers and 20,000 Medical/Clinical Research papers.

## arXiv Open-Reuse Physics and Engineering Corpus

Build a scientific-document corpus from the Cornell arXiv metadata snapshot on
Kaggle while filtering licenses before any PDF retrieval. The pipeline streams
`arxiv-metadata-oai-snapshot.json` line by line, filters by arXiv category
prefixes and optional year bounds, and only selects records whose paper-level
`license` field normalizes to the configured allowlist. It does not use the
standard paginated arXiv API or the live OAI-PMH endpoint for initial bulk
collection.

Install and authenticate the Kaggle CLI before the first run:

```bash
pip install kaggle
# Create an API token at https://www.kaggle.com/settings/account, then either:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
# or set KAGGLE_USERNAME and KAGGLE_KEY in your environment.
```

The command reuses an existing snapshot when `--snapshot-path` points to one or
when `data/arxiv_open_reuse/arxiv-metadata-oai-snapshot.json` already exists.
If the snapshot is missing, `--download-snapshot` is enabled by default and runs:

```bash
kaggle datasets download \
  -d Cornell-University/arxiv \
  -f arxiv-metadata-oai-snapshot.json \
  -p data/arxiv_open_reuse \
  --unzip
```

PDFs are retrieved from the public Google Cloud bucket linked by the official
Cornell/Kaggle arXiv dataset, not from requester-pays S3. The inspected bucket
layout stores individual PDFs, so the downloader requests only exact selected
objects. New-style IDs use paths like
`arxiv/arxiv/pdf/2401/2401.00003v3.pdf`; old-style IDs use paths like
`arxiv/hep-th/pdf/9901/9901001v1.pdf`. Selected IDs that are not present in
the Kaggle/GCS PDF bucket are written to `pdfs/kaggle_unavailable_ids.jsonl`.

Run the combined discovery and PDF download pass:

```bash
uv run dataset-generation build-arxiv-open-reuse \
  --snapshot-path data/arxiv_open_reuse/arxiv-metadata-oai-snapshot.json \
  --target-for-prefix physics.=30000 \
  --target-for-prefix eess.=60000 \
  --category-prefix physics. \
  --category-prefix eess. \
  --max-workers 32
# or
scripts/document_downloads/build_arxiv_open_reuse.sh \
  --max-workers 32
```

This targets 30,000 Physics records and 60,000 Engineering records in one
combined manifest under `data/arxiv_open_reuse/`. To run Physics and
Engineering in parallel instead, use the split scripts in separate
terminals. They write to separate output directories, so their selected
manifests and PDF manifests do not collide:

```bash
scripts/document_downloads/build_arxiv_open_reuse_physics.sh \
  --max-workers 32
scripts/document_downloads/build_arxiv_open_reuse_engineering.sh \
  --max-workers 32
```

Use `--metadata-only` when you want to audit eligibility before downloading.
Tune selection with repeated `--category-prefix`, `--start-year`,
`--end-year`, `--target-count`, `--target-per-domain`, or repeated
`--target-for-prefix PREFIX=COUNT`, and repeated
`--allowed-license`. By default the category prefixes are `physics.` and
`eess.`, the CLI target is 20,000 selected records, and the bundled shell
scripts target `physics.=30000` and `eess.=60000`. The license allowlist is strictly
`CC BY 4.0`. Equivalent license
URLs are normalized across HTTP/HTTPS and trailing slashes; missing licenses,
unknown licenses, non-CC licenses, arXiv's default non-exclusive distribution
license, and Creative Commons licenses outside the allowlist are rejected before
any PDF is queued. Selection is deterministic in snapshot order; `--selection-seed`
is recorded for reproducibility but no sampling is used.

The old OAI-PMH helpers remain available for legacy incremental use via
`--metadata-source oai`, but OAI is no longer required for this initial bulk
collection path.

PDF downloads are separate from selection and intentionally conservative. Tune
parallel PDF downloads with `--max-workers`, network retry delay with
`--request-delay-seconds`, timeouts with `--timeout-seconds`, and retry count
with `--max-retries`. The default downloader uses up to 32 workers and no
per-request delay. Downloads are resumable: valid existing PDFs are skipped,
partial downloads use `.part` files, and every completed download is validated
for a PDF header before being accepted.

You can also download only eligible PDFs later, resuming existing metadata and
skipping valid PDFs already present:

```bash
uv run dataset-generation download-documents arxiv-open-reuse --max-workers 32
# or
scripts/document_downloads/download_arxiv_open_reuse_pdfs.sh --max-workers 32
```

The default target is 20,000 selected records. The default category prefixes
are `physics.` and `eess.`. The default license whitelist is `CC BY 4.0` only.
Outputs are written
under `data/arxiv_open_reuse/`:

```text
selected_arxiv_documents.jsonl
eligible_records.jsonl      # compatibility copy of selected records
report.json                 # counts by domain, category, year, and license
pdfs/*.pdf                  # downloaded only after eligibility is known
pdfs/download_manifest.jsonl
pdfs/download_failures.jsonl
pdfs/kaggle_unavailable_ids.jsonl
arxiv-metadata-oai-snapshot.json
```

To materialize separate Physics and Engineering PDF folders from the combined
manifest, run:

```bash
uv run python scripts/split_arxiv_pdfs_by_domain.py
```

This copies valid source PDFs from `data/arxiv_open_reuse/pdfs/` into
`data/arxiv_open_reuse/pdfs_by_domain/physics/` and
`data/arxiv_open_reuse/pdfs_by_domain/engineering/` using the manifest's
`selection_domain`. Add `--move` only if you want to move files out of the flat
PDF directory instead of copying them.

To build one consolidated five-folder PDF dataset, first make sure arXiv has
already been split into `pdfs_by_domain/`, then run:

```bash
uv run python scripts/build_pdf_datasets_folder.py --dry-run
uv run python scripts/build_pdf_datasets_folder.py
```

This creates `data/pdf_datasets/ACL`, `data/pdf_datasets/Physics`,
`data/pdf_datasets/Engineering`, `data/pdf_datasets/Biology`, and
`data/pdf_datasets/Medicine`. Use `--clean` to rebuild the destination from
scratch, or `--move` if you want to move files instead of copying them.

Package the five-folder corpus for Hugging Face Datasets as one shared
retrieval corpus:

```bash
uv run python prepare_hf_dataset.py \
  --input-root data/pdf_datasets \
  --output-dir huggingface_dataset \
  --shard-size-gb 1 \
  --prepare-only
```

The command writes:

```text
huggingface_dataset/
  README.md
  metadata.parquet
  duplicates.parquet
  preparation_report.json
  data/<SOURCE>/shard-00000.tar
```

Each uncompressed TAR shard uses a WebDataset-compatible pair per document:
`DOCUMENT_ID.pdf` and `DOCUMENT_ID.json`. `metadata.parquet` has one row per
PDF with the stable `document_id`, source, original relative path, shard path,
member path, byte size, SHA-256 checksum, document-level license when present,
title, year, and DOI. Exact duplicate files are preserved and recorded in
`duplicates.parquet`.

The packager is resumable. Completed shards are written through temporary files,
atomically renamed, and skipped on rerun when their shard manifest and TAR
members validate. Use `--force-rebuild` to rebuild completed shards.

Validate an existing prepared folder without uploading:

```bash
uv run python prepare_hf_dataset.py \
  --output-dir huggingface_dataset \
  --validate-only
```

Upload only after validation succeeds:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
uv run python prepare_hf_dataset.py \
  --output-dir huggingface_dataset \
  --repo-id kasys/open-source-scientific-documents \
  --upload-only
```

Before upload, validation scans the source folders, checks TAR members, and
recomputes SHA-256 for every source PDF. The command logs each stage, periodic
file and byte counts, and elapsed time. Once `starting hf upload` appears,
transfer progress is reported by the Hugging Face CLI.

If the prepared dataset has already been validated and has not changed, skip
the local validation pass on upload:

```bash
HF_XET_HIGH_PERFORMANCE=1 uv run python prepare_hf_dataset.py \
  --output-dir huggingface_dataset \
  --upload-only --skip-validation
```

The upload path uses the current Hugging Face CLI (`hf upload`) and the existing
authenticated session or `HF_TOKEN`; it does not print tokens or create manual
Git commits. The same command is also available through:

```bash
uv run dataset-generation prepare-hf-dataset --help
```

To inspect a single packaged PDF by `document_id`:

```python
import tarfile

import pandas as pd

metadata = pd.read_parquet("huggingface_dataset/metadata.parquet")
row = metadata.loc[metadata.document_id == "ACL_..."].iloc[0]

with tarfile.open("huggingface_dataset/" + row.shard, "r") as tar:
    pdf_bytes = tar.extractfile(row.member_path).read()
```

Create a random fixed-seed train/test split from those five PDF folders with
1,000 train PDFs and 100 test PDFs per dataset:

```bash
uv run python scripts/build_pdf_dataset_split.py \
  --input-dir data/pdf_datasets \
  --output data/splits/pdf_dataset_split.json
```

The selector/downloader uses `.part` files for atomic PDF writes, PDF header
validation before marking success, and stable filenames derived from arXiv
identifiers. Interrupted runs retain partial manifests and skip selected
metadata and valid PDFs already present. When running inside tmux, snapshot
scan and PDF phases also post short `tmux display-message` status updates so
pane status bars keep moving during long runs.

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
