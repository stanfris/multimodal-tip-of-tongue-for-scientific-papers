# Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers

This repository is part of the paper `Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers`. Here, we provide all of the code necessary to generate all components of our dataset, in order to provide a basis for future work in generating queries and documents. We also provide instructions to simply download our dataset, such that you can use it directly. 

## Repository Layout
This repository contains code for each step, and source files, scripts, and results are all organized by these steps. The core steps include the downloading of the raw documents, extraction of figures and text, the generation of clues, generation of queries, and the judging of queries. 


```text
src/
  cli.py                 # dataset-generation command line interface
  common/                # Shared JSONL, settings, validation, and model helpers
  document_downloads/    # Source metadata builders and PDF download commands
  pdf_corpus/            # PDF corpus assembly, splitting, and statistics
  document_splits/       # Canonical split filtering and legacy index compatibility
  extraction/            # MinerU extraction logic
  preprocessing/         # Preprocessed paper readers and cleanup tools
  clues/                 # Textual and visual clue generation
  queries/               # Query generation, judgement, and test collections
  dataset_packaging/     # Dataset schema, storage, and Hugging Face packaging
  review/                # Streamlit query review application

scripts/
  environment/           # Environment setup
  document_downloads/    # Document corpus workflow wrappers
  pdf_corpus/            # Corpus assembly, domain splitting, and statistics
  document_splits/       # Legacy extracted-document split tooling
  extraction/            # Local and Slurm extraction wrappers
  preprocessing/         # Preprocessing workflow wrappers
  clue_generation/       # Clue-generation wrappers
  query_generation/      # Query-generation and judgement wrappers
  review/                # Query review application launcher
  reporting/             # Generated-artifact reporting wrappers
  packaging/             # Hugging Face upload/download wrappers

config/                   # Hydra launcher, stage, split, model, and dataset groups
logs/                     # Runtime logs grouped by script folder
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

The repository uses two intentionally separate environments because the main
pipeline and MinerU 3.x require incompatible Transformers versions:

```bash
scripts/environment/sync_env.sh
scripts/environment/sync_mineru_env.sh
```

`.venv` is the main project environment. `.venv-mineru` is a disposable,
isolated MinerU 3.4.0 GPU environment pinned to Transformers 4.57.3. Re-running
its sync script clears and rebuilds it, so incompatible packages cannot linger.
The MinerU environment also includes Ninja for vLLM/FlashInfer CUDA extension
compilation.
Standard wrappers select the correct environment
automatically. To switch an interactive shell explicitly:

```bash
source scripts/environment/activate_env.sh main
source scripts/environment/activate_env.sh mineru
```

To run one command without changing the current shell:

```bash
scripts/environment/run_in_env.sh main python --version
scripts/environment/run_in_env.sh mineru mineru --version
```

## Hydra Configuration and Launching

`dataset-generation` is the Hydra entrypoint for clue generation, query generation,
judgement, MinerU extraction callers, and preprocessing reduction/compaction. Its resolved config is the
source of truth. The config groups are:

| Group | Purpose |
| --- | --- |
| `config/dataset/` | Dataset identity, roots, input and output locations |
| `config/stage/` | Stage selection and stage-specific limits/workers |
| `config/model/` | Reusable model identifiers and loading defaults |
| `config/split/` | `train`, `test`, or `all` for reduction/compaction |
| `config/launcher/` | Machine profile: resources, data/output/cache paths, and scheduler |
| `config/visual_descriptions/`, `textual_descriptions/`, `visual_query/` | Existing prompt and generation settings |

Use Hydra overrides directly. The stage scripts below select a stage and
forward the rest of the command unchanged:

```bash
scripts/query_generation/generate_queries.sh split=test stage.limit=3
scripts/clue_generation/describe_all_figures.sh launcher=local_gpu stage.limit=3
scripts/query_generation/generate_queries.sh model=noop split=test stage.limit=3
```

`model=noop` generates template queries without loading a model. For a real
model, choose `model=phi4`, `model=qwen3_text`, `model=qwen3_vl`, or
`model=gemma3_judge`, as appropriate for the stage. Stage selection chooses a
sensible default model; an explicit `model=...` override takes precedence.
Inspect the full composition with `uv run --no-sync dataset-generation
stage=generate_queries split=test --cfg job --resolve`.

Local runs use `launcher=local_gpu` by default. A small preprocessing smoke run
that does not mutate data is:

```bash
scripts/preprocessing/reduce_and_compact_preprocessed.sh split=all \
  stage.dry_run=true dataset.processed_root=data/preprocessed \
  dataset.pdf_dir=data/acl_subset/pdfs stage.workers=1
```

The `extract_mineru` stage calls an already running MinerU API; the server
start/readiness/shutdown lifecycle remains in the extraction scripts.

The launcher profile supplies filesystem locations as well as execution
resources. `launcher=slurm_a100` and `launcher=slurm_cpu` point at Snellius
scratch locations and select Hydra Submitit. Use `-m` to submit through Slurm:

```bash
uv run --no-sync dataset-generation -m stage=describe_figures launcher=slurm_a100 split=train
```

`launcher=pbs_rt_hg` and `launcher=pbs_rt_hc` use PBS, while
`launcher=local_gpu` runs locally. Override machine paths with values such as
`launcher.dataset_root=/scratch/project`, `launcher.processed_root=/scratch/project/processed`,
`launcher.cache_root=/scratch/cache`, or `dataset.query_output=/scratch/queries`.
The environment variables `VTT_DATA_ROOT`, `VTT_CACHE_ROOT`, `VTT_LOG_ROOT`, `HF_HOME`, and `VTT_PYTHON` remain optional machine defaults.
Set `dry_run=true` to inspect the resolved plan without executing or submitting.

Hydra writes `.hydra/config.yaml`, `.hydra/hydra.yaml`, and
`.hydra/overrides.yaml` in its run directory. The pipeline also writes
`plan.json` and, for generation stages, `settings.yaml` as the adapter
snapshot used by existing generation code. MinerU server orchestration has
its own process lifecycle and remains in the extraction scripts.

## Query Review UI

Launch the local Streamlit reviewer for generated tip-of-the-tongue query collections:

```bash
scripts/review/run_review_app.sh
```

The reviewer can inspect generated query collections and writes manual
annotations separately to `data/reviews/query_review_annotations.jsonl`.

## Bash Scripts

Run the dataset generation stages individually:

```bash
scripts/environment/sync_env.sh
scripts/environment/sync_mineru_env.sh
scripts/document_downloads/build_acl_subset.sh
scripts/document_downloads/download_acl_pdfs.sh
scripts/pdf_corpus/build_pdf_datasets_folder.sh
scripts/pdf_corpus/build_pdf_dataset_split.sh
scripts/extraction/start_mineru_router.sh
scripts/extraction/run_mineru_full_extraction.sh
scripts/preprocessing/reduce_and_compact_preprocessed.sh
scripts/clue_generation/describe_all_figures.sh
scripts/clue_generation/describe_all_textual_clues.sh
scripts/query_generation/generate_queries.sh
scripts/query_generation/judge_train_queries.sh
```

Run `start_mineru_router.sh` in a separate terminal before
`run_mineru_full_extraction.sh`. The clue and query stages read directly
from `data/preprocessed` or `data/preprocessed/papers`.

For an external corpus, set the paths through Hydra:

```bash
scripts/preprocessing/reduce_and_compact_preprocessed.sh split=all \
  launcher.dataset_root=/path/to/corpus-root \
  dataset.processed_root=/path/to/corpus-root/processed \
  dataset.pdf_dir=/path/to/corpus-root/pdf_datasets
```

The preprocessing stage prints discovered-paper totals and progress records.
Use `stage.progress_every=1` for per-paper updates or `0` to disable them.
The clue and query scripts accept the same `stage`, `split`, `model`, `launcher`,
`dataset`, and generation-section Hydra overrides described above.

Source-document download and MinerU utilities still expose their dedicated
operational commands; for example `uv run dataset-generation download-documents
--help`.

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
`document_downloads.acl_subset.TARGET_VOLUMES`
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
scripts/pdf_corpus/split_arxiv_pdfs_by_domain.sh
```

This copies valid source PDFs from `data/arxiv_open_reuse/pdfs/` into
`data/arxiv_open_reuse/pdfs_by_domain/physics/` and
`data/arxiv_open_reuse/pdfs_by_domain/engineering/` using the manifest's
`selection_domain`. Add `--move` only if you want to move files out of the flat
PDF directory instead of copying them.

To build one consolidated five-folder PDF dataset, first make sure arXiv has
already been split into `pdfs_by_domain/`, then run:

```bash
scripts/pdf_corpus/build_pdf_datasets_folder.sh --dry-run
scripts/pdf_corpus/build_pdf_datasets_folder.sh
```

This creates `data/pdf_datasets/ACL`, `data/pdf_datasets/Physics`,
`data/pdf_datasets/Engineering`, `data/pdf_datasets/Biology`, and
`data/pdf_datasets/Medicine`. Use `--clean` to rebuild the destination from
scratch, or `--move` if you want to move files instead of copying them.

Package the five-folder corpus for Hugging Face Datasets as one shared
retrieval corpus:

```bash
scripts/packaging/prepare_huggingface_dataset.sh \
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
scripts/packaging/prepare_huggingface_dataset.sh \
  --output-dir huggingface_dataset \
  --validate-only
```

Upload only after validation succeeds:

```bash
scripts/packaging/upload_huggingface_dataset.sh huggingface_dataset
```

The upload script targets `kasys/open-source-scientific-documents` by default.
Set `HF_REPO_ID=owner/dataset-name` to use a different dataset repository. Any
arguments after the dataset directory are forwarded to `dataset_packaging.prepare_hf_dataset`.

Before upload, validation scans the source folders, checks TAR members, and
recomputes SHA-256 for every source PDF. The command logs each stage, periodic
file and byte counts, and elapsed time. Once `starting hf upload` appears,
transfer progress is reported by the Hugging Face CLI.

If the prepared dataset has already been validated and has not changed, skip
the local validation pass on upload:

```bash
HF_XET_HIGH_PERFORMANCE=1 scripts/packaging/prepare_huggingface_dataset.sh \
  --output-dir huggingface_dataset \
  --upload-only --skip-validation
```

The upload path uses the current Hugging Face CLI (`hf upload`) and the existing
authenticated session or `HF_TOKEN`; it does not print tokens or create manual
Git commits. Inspect the packaging options through the same wrapper:

```bash
scripts/packaging/prepare_huggingface_dataset.sh --help
```

Download the complete dataset repository into a specific local directory:

```bash
scripts/packaging/download_huggingface_dataset.sh data/downloaded_hf_dataset
```

Restore the downloaded shards to the original `pdf_datasets` layout. The target
directory will contain source folders such as `ACL`, `Physics`, and `Medicine`:

```bash
scripts/packaging/restore_pdf_datasets_from_huggingface.sh \
  /scratch-shared/sfris1 \
  data/pdf_datasets
```

The restore command checks every PDF's recorded size and SHA-256 checksum, and
will not replace existing PDFs unless `--overwrite` is supplied. For faster
restoration, process independent TAR shards concurrently with `--workers`.
Use `--skip-checksum` only for trusted archives; size checks still apply:

```bash
scripts/packaging/restore_pdf_datasets_from_huggingface.sh \
  /scratch-shared/sfris1 \
  data/pdf_datasets \
  --workers 4 --skip-checksum
```

The downloader uses the same default repository. Override it with
`HF_REPO_ID=owner/dataset-name`, or set `HF_CLI` when the `hf` executable is not
on `PATH`. For the project virtual environment, run:

```bash
HF_CLI="$PWD/.venv/bin/hf" scripts/packaging/download_huggingface_dataset.sh /scratch-shared/sfris1
```

Additional arguments are forwarded to `hf download`, for example:

```bash
scripts/packaging/download_huggingface_dataset.sh data/metadata-only \
  --include "*.parquet" "README.md"
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

Create a random fixed-seed train/test split from those five PDF folders. By
default, each dataset group contributes 2,200 train PDFs and 110 test PDFs, for
11,000 train PDFs and 550 test PDFs in total:

```bash
scripts/pdf_corpus/build_pdf_dataset_split.sh \
  --input-dir data/pdf_datasets \
  --output data/splits/pdf_dataset_split.json
```

For the scratch-shared corpus, both paths can be selected with its root:

```bash
scripts/pdf_corpus/build_pdf_dataset_split.sh --root-dir /scratch-shared/sfris1
```

This reads the PDF subfolders beneath `/scratch-shared/sfris1/pdf_datasets`
and writes `/scratch-shared/sfris1/data/splits/pdf_dataset_split.json`.

The split contains source documents, not a guaranteed number of generated
queries: downstream generation can emit one or more queries per selected PDF.
Use `--train-size-per-dataset` and `--test-size-per-dataset` to override these
document counts.

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
scripts/extraction/probe_mineru_env.sh
```

On the DGX, start a persistent router:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/extraction/start_mineru_router.sh \
  --host 127.0.0.1 \
  --port 8002
```

The wrapper activates `.venv-mineru` for the router and its vLLM/FlashInfer
subprocesses. This is required so JIT build tools such as `ninja` remain on
`PATH`. The environment also pins `pdftext==0.6.3`: MinerU 3.4.0 is not
compatible with the non-iterable `PageChars` API introduced in pdftext 0.7.
Pass `--help` to the wrapper for custom router settings.

With the router running, use the Hydra extraction caller for a small local run:

```bash
uv run --no-sync dataset-generation stage=extract_mineru split=train \
  stage.limit=20 stage.max_in_flight=2
```

The `extract_mineru` stage uses `dataset.pdf_dir` as input and
`dataset.processed_root` as output. Its default API URL is
`http://127.0.0.1:8002`; override it with `stage.api_url=...` if the router uses
another port. Use `split=test` for the test split. To parse all discovered PDFs,
use `split=all stage.all_domain_pdfs=true`. The router must already be running.

The older `scripts/extraction/run_mineru_full_extraction.sh` remains available
as an operational caller with argparse options. By default, it reads every PDF under
`data/pdf_datasets/{ACL,Physics,Engineering,Biology,Medicine}`. A selected
train/test scope reads `data/splits/pdf_dataset_split.json`, so split membership
and source dataset names are recorded in each extracted `paper.json`. For a
small smoke run, pass normal CLI overrides through the script:

```bash
scripts/extraction/run_mineru_full_extraction.sh --split train --limit 20 --max-in-flight 2
```

Process only the PDFs listed in both the train and test portions of the
canonical split index:

```bash
scripts/extraction/run_mineru_full_extraction.sh --split train+test
```

Restrict that split-index run to one or more domain groups:

```bash
scripts/extraction/run_mineru_full_extraction.sh \
  --split train+test \
  --domains ACL Biology
```

To process every PDF in selected domain folders, including documents outside
the train/test split, bypass the split index explicitly:

```bash
scripts/extraction/run_mineru_full_extraction.sh \
  --domains Engineering Physics \
  --all-domain-pdfs
```

A run with exactly one `--domains` value writes beneath that domain name. For
example, `--domains Engineering` writes papers, failure records, and run
metadata under `data/processed/Engineering/`.

To run the router and extraction caller together on one GPU machine, use the
portable launcher. It starts one local MinerU server, waits for the real
`/health` endpoint on `127.0.0.1`, runs the existing extraction caller, then
shuts the server down and exits with the caller's status:

```bash
scripts/extraction/run_mineru_full_extraction_gpu.sh --split train --limit 20 --max-in-flight 2
```

The launcher defaults to port `8002` and writes separate server/caller logs.
Server output stays in the server log, while extraction progress is both shown
live and appended to the caller log under `logs/extraction/local/`. Override
launcher settings before normal caller arguments:

```bash
scripts/extraction/run_mineru_full_extraction_gpu.sh \
  --port 8012 \
  --startup-timeout 900 \
  -- \
  --input-dir data/pdf_datasets \
  --output-dir data/processed \
  --split all
```

Before submitting PDFs, the launcher verifies that `.venv-mineru` has MinerU
3.4.0 and a Transformers 4.57.3 PP-DocLayoutV2 configuration with reading-order
support. Create or repair that isolated environment with:

```bash
scripts/environment/sync_mineru_env.sh
```

By default the server command is:

```bash
.venv-mineru/bin/mineru-router --host 127.0.0.1 --port <PORT>
```

If a local MinerU install exposes a different router flag shape, pass the exact
server command with `--server-cmd` or `MINERU_SERVER_CMD`. The caller remains
the repository CLI and preserves the current `hybrid-engine`, page range, and
output format defaults unless you explicitly override them with caller args.

On a Slurm cluster with A100 nodes, submit one server plus one caller as a
single job:

```bash
sbatch scripts/extraction/slurm/run_mineru_full_extraction_a100.job
```

When submitting from outside the repository root, set the checkout explicitly:

```bash
repo_root=/path/to/repo
sbatch \
  --output="$repo_root/logs/extraction/slurm/mineru_extract_%j.out" \
  --error="$repo_root/logs/extraction/slurm/mineru_extract_%j.err" \
  "$repo_root/scripts/extraction/slurm/run_mineru_full_extraction_a100.job" \
  --root-dir "$repo_root"
```

The script uses one node, one A100 GPU, 16 CPUs, 32G memory, and a three-hour
time limit. It chooses a localhost port and writes Slurm stdout/stderr plus the
server/caller logs to `logs/extraction/slurm/`. It uses `$SLURM_TMPDIR` for
temporary files when available. Live extraction progress is written to both
the caller log and `mineru_extract_<job>.out`; MinerU server output remains only
in the server log. Input and split-index paths default to locations beneath
`/scratch-shared/sfris1`, and extracted papers are written to
`/scratch-shared/sfris1/processed`. Additional arguments after the script path
are forwarded to `extract-mineru-pdfs`; `--root-dir` sets the repository
checkout, and environment variables such as
`DATASET_DIR`, `INPUT_DIR`, `SPLIT_INDEX`, `SPLIT`, `OUTPUT_DIR`,
`MINERU_PORT`, and `MINERU_STARTUP_TIMEOUT` override the defaults. The Slurm
launcher sets both MinerU's server request limit and the extraction caller's
in-flight limit from `MINERU_CONCURRENCY`, which defaults to `8`.

The A100 job processes every discovered PDF by default. Pass `--split train`,
`--split test`, or `--split train+test` after the `.job` path to use the
corresponding entries from the split index. Override its location with
`--split-index PATH`. Every launch reports its selected scope and resume
behavior before starting the MinerU server.

Completed papers are skipped on restart. Each successful paper has:

```text
data/preprocessed/papers/<subset>/<paper_id>/
  _SUCCESS
  markdown.md
  paper.json
  figures.json
  mineru/                  # MinerU Markdown, content_list JSON, images
```

`paper_id` is derived from the PDF path relative to the input root, so repeated
filenames across datasets do not collide. `paper.json` records the source PDF,
relative source path, source dataset, split, MinerU configuration/version,
Markdown path, page count if available, and normalized `image`/`chart` figure
records with page, bounding box, caption, footnote, and image paths. Failed PDFs are
appended to `<output-dir>/failures.jsonl` and do not stop the batch. The latest
run configuration and summary are saved as `run_config.json` and
`last_run_summary.json`. Each run also rewrites
`<output-dir>/incomplete_documents.json` with the current complement of the
successful outputs: selected PDFs that are still pending, partial, or failed.
At startup, `resume_report.json` records how many selected documents are being
preserved as complete, queued as incomplete, and retried after prior failures.
Completed paper directories are never replaced. An incomplete directory is
replaced only after its retry has produced and validated a complete result.
Use `--retry-incomplete-only` to process only the PDFs named by that manifest or
the failure log; completed papers are still skipped.

Extraction and automatic postprocessing preserve the complete document; they
do not impose a page-count limit. Page-count filtering is confined to the
explicit `--prune` mode of `pdf_corpus.pdf_page_distribution`.

Before a full run, benchmark medium-effort throughput on a small sample:

```bash
scripts/extraction/benchmark_mineru_extraction.sh \
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

Compaction writes the same `incomplete_documents.json` shape under the
preprocessed root. It records split-index PDFs with no completed extraction,
paper directories missing required compaction inputs, and compaction failures;
failures are also appended to `compaction_failures.jsonl`.

## Figure Descriptions

Generate Qwen-VL visual descriptions from preprocessed papers:

```bash
scripts/clue_generation/describe_all_figures.sh
```

This reads figure image references from
`data/preprocessed` and writes visual clues to
`data/clues/<paper_id>/images/<figure_id>.jsonl`. For a small figure run, use `stage.limit=3` and choose an appropriate
`model` and `launcher` profile. Output locations come from `dataset.root` and
`dataset.clues_dir`.

## Textual Clues

Generate semantic memory cues from preprocessed paper markdown:

```bash
scripts/clue_generation/describe_all_textual_clues.sh
```

This writes textual clues to `data/clues/<paper_id>/base/textual_clues.jsonl`.

Print stored coverage statistics:

```bash
scripts/reporting/generated_artifact_stats.sh \
  --dataset data/query_collections/query_generation_train/visual_only
```

## Artifact Structure

Preprocessing and clue generation write:

```text
data/preprocessed/papers/<subset>/<paper_id>/
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

`queries.synthetic` defines stable interfaces for later retrieval/evaluation dataset generation:

- `TestCollectionExample`
- `SyntheticCollectionConfig`
- `generate_synthetic_queries(records, config)`
- `write_test_collection(collection, output_dir)`

The current implementation is deterministic and template-based. It deliberately does not hard-code an LLM provider; future model-backed generation should be added behind an adapter interface.

## Tests

Run fixture-based tests without downloading full datasets:

```bash
.venv/bin/python -m pytest
```
