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
data/preprocessed/                       # extracted and compacted papers
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

## Run the pipeline

Run these commands from the repository root. Install `uv` and create the main
Python 3.11 environment before any wrapper that calls `uv run`. The separate
MinerU environment is needed on the GPU execution node for extraction. `uv
run --no-sync` in the Hydra wrappers uses the already installed `.venv`; it
does not install missing packages. Environment setup changes `.venv` or
`.venv-mineru` and prints to the terminal; it creates no pipeline log.

```bash
scripts/environment/sync_env.sh
scripts/environment/sync_mineru_env.sh
scripts/environment/run_in_env.sh main python --version
scripts/environment/run_in_env.sh mineru mineru --version
```

The second sync **clears and rebuilds** `.venv-mineru`. The main environment
contains the CLI and Streamlit; MinerU 3.4.0 uses its own pinned environment.
On Slurm, the repository and both environments must be visible from the
allocated node. Set `VTT_DATA_ROOT`, `VTT_LOG_ROOT`, `VTT_CACHE_ROOT`, and
`HF_HOME` as needed, or override the corresponding `launcher.*` paths.

### Corpus and canonical PDF split

The ACL metadata builder and the PMC metadata builder accept `--metadata-only`
through their wrappers. PMC requires an NCBI contact email. arXiv requires an
authenticated Kaggle CLI or an existing metadata snapshot. These are network
operations; `--help` only inspects arguments and creates no corpus files.

```bash
scripts/document_downloads/build_acl_subset.sh
scripts/document_downloads/download_acl_pdfs.sh --max-workers 16
scripts/document_downloads/build_pmc_oa_subset.sh --email you@example.org
scripts/document_downloads/download_pmc_oa_pdfs.sh --max-workers 16
scripts/document_downloads/build_arxiv_open_reuse.sh --metadata-only --snapshot-path data/arxiv_open_reuse/arxiv-metadata-oai-snapshot.json
scripts/document_downloads/download_arxiv_open_reuse_pdfs.sh --max-workers 32
scripts/pdf_corpus/split_arxiv_pdfs_by_domain.sh --dry-run
scripts/pdf_corpus/split_arxiv_pdfs_by_domain.sh
scripts/pdf_corpus/build_pdf_datasets_folder.sh --dry-run
scripts/pdf_corpus/build_pdf_datasets_folder.sh
scripts/pdf_corpus/build_pdf_dataset_split.sh --input-dir data/pdf_datasets --output data/splits/pdf_dataset_split.json
```

The arXiv metadata command needs the snapshot at the shown path; omit
`--snapshot-path` to let the builder download it with Kaggle credentials.
Download results and reports live under `data/acl_subset/`,
`data/pmc_oa_strict/`, and `data/arxiv_open_reuse/`. Assembly writes
`data/arxiv_open_reuse/pdfs_by_domain/` and `data/pdf_datasets/`. The last
command requires the five assembled PDF folders and writes
`data/splits/pdf_dataset_split.json` (2,200 train and 110 test PDFs per
folder by default). These utilities report progress to the terminal; they do
not use Hydra run logs. Inspect options with, for example,
`scripts/pdf_corpus/build_pdf_dataset_split.sh --help`. The old
`scripts/document_splits/build_document_split.sh` is a post-extraction legacy
split and is not the canonical input to extraction.

### Hydra stages and argument order

#### Shared inference backends and ABCI H200

The generative stages use `src/inference`: each stage submits `GenerationRequest`
windows and receives ordered `GenerationResult` values. `transformers` performs
padded model batches, `vllm` submits the whole window to the offline engine for
continuous batching, and `mlx` retains the local Apple Silicon path. Backend
packages are imported only when selected. Existing clue and query output paths,
JSONL rows, and resume keys are unchanged.

Select the backend with `model.provider=transformers`, `model.provider=mlx`, or
`model.provider=vllm`. The model choice is independent: `model=qwen3_vl`,
`model=qwen3_text`, `model=phi4`, and `model=gemma3_judge` still work. Install
vLLM in the H200 worker environment before submitting vLLM runs, for example
with `uv pip install --python .venv/bin/python vllm` after the normal environment
sync. The local test environment does not need it.

The `pbs_h200` launcher requests one GPU. Its starting `max_num_seqs` values
are 64 for Qwen3-VL, 256 for Qwen3 text, 128 for Phi-4, and 64 for Gemma 3
27B. It uses BF16 weights, automatic KV-cache dtype, prefix caching, and a
16,384-token batch budget. These are benchmark priors, not measured optima.
The figure visual budget defaults to 768 tokens on this launcher.
The PBS job sets `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM's sampler does not
require a CUDA compiler on H200 nodes. If a CUDA toolkit with `nvcc` is available,
override it with `launcher.environment.VLLM_USE_FLASHINFER_SAMPLER=1` to benchmark
FlashInfer sampling.

```bash
scripts/clue_generation/describe_all_figures.sh split=train launcher=pbs_h200 model=qwen3_vl model.provider=vllm
scripts/clue_generation/describe_all_textual_clues.sh split=train launcher=pbs_h200 model=qwen3_text model.provider=vllm
scripts/query_generation/generate_queries.sh split=train launcher=pbs_h200 model=phi4 model.provider=vllm
scripts/query_generation/judge_train_queries.sh split=train launcher=pbs_h200 model=gemma3_judge model.provider=vllm
```

Append `dry_run=true` to inspect the resolved plan and settings without
submitting. Runtime overrides use, for example,
`launcher.vllm.max_num_seqs=32 launcher.vllm.max_num_batched_tokens=8192`.
For figures append `visual_descriptions.image.max_visual_tokens=512` (or 768 or
1024). `visual_descriptions.generation.batch_size`,
`textual_descriptions.generation.batch_size`, `visual_query.model.batch_size`,
and `visual_query.judgement.batch_size` control how many requests enter a real
inference call. The H200 preset keeps tensor parallel size at one, including
Gemma; override `launcher.vllm.tensor_parallel_size` only when using a different
GPU allocation.

Run representative benchmarks without writing clues or queries:

```bash
PYTHONPATH=src python -m inference.benchmark visual --provider vllm --dataset data/preprocessed
PYTHONPATH=src python -m inference.benchmark text --provider transformers --dataset data/preprocessed
```

The benchmark grids cover 32/64/128 sequences and 512/768/1024 visual tokens
for figures, 128/256/512 sequences for Qwen3 text, 64/128/256 for Phi-4,
32/64/128 for Gemma, and Transformers batch sizes 1/4/8. It prints JSON lines
with elapsed time, throughput, failures, and configuration. Model weights and a
compatible GPU are needed for real vLLM measurements.

`src/cli.py` routes operational subcommands to argparse and all other calls
to Hydra. Stage wrappers embed their own `stage=...`; do not add a second stage
override. `split=train` or `split=test` selects a canonical split for model
stages. Reduction and extraction also accept `split=train+test`. MinerU
extraction also accepts `split=other` to process PDFs absent from both indexed
splits. The file
`config/split/all.yaml` currently also resolves to `train+test`, so use the
explicit name. Launchers default to the repository's `data/` directory.
Set `VTT_DATA_ROOT` or override `launcher.dataset_root` to move the corpus;
PDFs, the split index, processed output, preprocessing output, and caches
follow that root. For example:

```bash
launcher.dataset_root=/scratch/project launcher.log_root=/scratch/project/logs
```

Append those path overrides to a Hydra command below, replacing the example
paths. Individual `launcher.pdf_dir`, `launcher.processed_root`, and
`launcher.split_index` overrides remain available. `dataset.query_output`
separately controls query output. Indexed extraction requires the split index;
full-folder MinerU extraction with `stage.all_domain_pdfs=true` ignores it.

The reduction, figure, textual, query, and judge wrappers put `stage=...`
**before** user arguments. For their Slurm calls, put overrides before `-m`,
as shown below. The MinerU wrapper instead appends `stage=extract_mineru`
**after** user arguments: its working form puts `-m` first, followed by
`split=` and `launcher=`. It also switches to the direct argparse extraction
client when its first argument begins with a recognized `--` option; that
client requires an already running MinerU server. `-m` submits through Hydra
Submitit; `dry_run=true` prints a plan without creating a run or submitting.
`--cfg job --resolve` inspects composition only. For MinerU, use the
entrypoint directly for `--cfg` because its wrapper appends `stage=` after
all supplied arguments, which Hydra rejects when `--cfg` is present.

```bash
scripts/preprocessing/reduce_and_compact_preprocessed.sh split=train launcher=slurm_cpu -m
scripts/preprocessing/reduce_and_compact_preprocessed.sh split=train launcher=slurm_cpu dry_run=true
scripts/preprocessing/reduce_and_compact_preprocessed.sh split=train launcher=local_gpu dataset.split_index=data/splits/pdf_dataset_split.json stage.dry_run=true
scripts/clue_generation/describe_all_figures.sh split=train launcher=slurm_a100 model=qwen3_vl -m
scripts/clue_generation/describe_all_figures.sh split=train launcher=slurm_a100 model=qwen3_vl dry_run=true
scripts/clue_generation/describe_all_textual_clues.sh split=train launcher=slurm_a100 model=qwen3_text -m
scripts/clue_generation/describe_all_textual_clues.sh split=train launcher=slurm_a100 model=qwen3_text dry_run=true
scripts/query_generation/generate_queries.sh split=train launcher=slurm_a100 model=phi4 -m
scripts/query_generation/generate_queries.sh split=train launcher=slurm_a100 model=phi4 dry_run=true
scripts/query_generation/judge_train_queries.sh split=train launcher=slurm_a100 model=gemma3_judge -m
scripts/query_generation/judge_train_queries.sh split=train launcher=slurm_a100 model=gemma3_judge dry_run=true
```

To run a model stage locally, replace `launcher=slurm_a100 -m` with
`launcher=local_gpu`; a CUDA GPU and model weights are then needed on that
machine. The CPU reduction can run with `launcher=local_gpu` despite the
profile name. `stage.dry_run=true` inspects reduction's paper work without
mutating it, but does execute its readers; `dry_run=true` only prints the
Hydra plan. Use `stage.limit=3` for a small figure, textual, or query run.
`model=noop` is supported for template query generation. The judge model is
selected through `model=gemma3_judge` and the stage maps it to
`visual_query.judgement.model`.

Reduction copies completed papers from `dataset.processed_root` to
`dataset.preprocessed`, then compacts the copies there. MinerU's original files
remain in `processed`. The stage also reads `dataset.pdf_dir`
and the optional `dataset.split_index`; it writes `preprocessed_analysis_report.json`
beside the preprocessed root, and
`incomplete_documents.json` and `compaction_failures.jsonl` within it.
For a lean domain export containing only `markdown.md` and files under `images/`,
use `stage.output_format=markdown_images` and `stage.domains=[DOMAIN]` with
`dataset.split_index=null`. This reads completed papers from
`data/processed/papers/DOMAIN/<paper_id>/` and stages them atomically at
`data/preprocessed/DOMAIN/<paper_id>/`. Existing lean destinations are skipped.
This mode does not need source PDFs or write JSON metadata into paper directories.
On ABCI, use `launcher=pbs_rt_hc launcher.walltime=05:00:00` and set
`VTT_PBS_PROJECT` to your ABCI group when it differs from the configured default.
Figure and textual stages read those papers and write
`<dataset.root>/clues/<domain>/<paper_id>/images/<figure_id>.jsonl` and
`<dataset.root>/clues/<domain>/<paper_id>/base/textual_clues.jsonl`. Query generation
writes `<dataset.query_output>/<domain>/query_generation_train/` (or
`query_generation_test/`) and `<dataset.root>/clues/<domain>/queries.jsonl`;
judging writes beside the queries in each domain collection. Each Hydra run writes
`plan.json`, `.hydra/`, and for model stages `settings.yaml` in
`<launcher.log_root>/<stage family>/<stage>/<timestamp>/` (Slurm multiruns
add `/0`). Reduction writes `preprocessing.log` there; figure, textual, query,
and judge stages write `describe_figures.log`, `describe_textual_clues.log`,
`generate_queries.log`, and `judge_queries.log`, respectively. These logs
capture stage stdout and stderr while retaining console output. Inspect the
resolved run path in the printed plan. A `dry_run=true` command prints only the
plan; any empty `.log` file Hydra creates for that command has no stage output.
During an executing run, the stage log is created immediately and receives a
flushed line after each completed clue batch or query.

### MinerU extraction

The Hydra wrapper runs the MinerU server and extraction caller inside one GPU
allocation. `launcher=pbs_h200` submits one PBS job; `launcher=slurm_a100 -m`
uses Hydra Submitit. The login node needs no visible CUDA device to submit.
The dedicated `.venv-mineru` must be available on the worker node. The direct
GPU script requires a visible CUDA device where it runs. On PBS workers that
provide a GPU UUID in `CUDA_VISIBLE_DEVICES`, the Hydra worker resolves that
UUID through `nvidia-smi` before starting MinerU. This keeps the job on its
assigned GPU while supplying the numeric index required by the pinned vLLM
runtime. An unresolved UUID stops the worker before it submits papers.
The H200 launcher also disables optional vLLM DeepGEMM kernels because the
dedicated MinerU environment does not install DeepGEMM; vLLM's supported
fallback retains the BF16 model and extraction settings.
Its 16-CPU PBS allocation is unchanged. Each ONNX operation defaults to four
intra-op threads and one inter-op thread; OMP, MKL, OpenBLAS, and NumExpr
default to four threads. Set `launcher.environment.MINERU_INTRA_OP_NUM_THREADS=2`
(or `4` or `8`) on the command line to compare thread counts. These settings
are inherited by both the MinerU server and extraction caller.

```bash
scripts/extraction/run_mineru_full_extraction.sh split=train launcher=pbs_h200
scripts/extraction/run_mineru_full_extraction.sh split=train launcher=pbs_h200 stage.server_concurrency=24 stage.max_in_flight=24
scripts/extraction/run_mineru_full_extraction.sh split=train launcher=pbs_h200 dry_run=true
scripts/extraction/run_mineru_full_extraction.sh -m split=train launcher=slurm_a100
scripts/extraction/run_mineru_full_extraction.sh -m split=other launcher=slurm_a100
scripts/extraction/run_mineru_full_extraction.sh -m split=train+test launcher=slurm_a100 stage.all_domain_pdfs=true 'stage.domains=[ACL]'
```

Set `dataset.pdf_dir`, `dataset.processed_root`, and `dataset.split_index`
for a nondefault corpus. The default server concurrency and client in-flight
count are both 32. Override them independently with
`stage.server_concurrency=N` and `stage.max_in_flight=N`; use `stage.limit=20`
or `stage.retry_incomplete_only=true` for a smaller or retry run. Indexed
splits require the canonical PDF split JSON. For `split=other`, extraction scans
`dataset.pdf_dir`, excludes every path in the index's train and test lists,
and marks selected papers as `other`. Paper output is
`<dataset.processed_root>/papers/<subset>/<paper_id>/`. The files `run_config.json`,
`last_run_summary.json`, `resume_report.json`, `incomplete_documents.json`,
and `failures.jsonl` remain at the processed root for runs without a single
domain filter. A single-domain run writes those files under
`<dataset.processed_root>/_runs/<domain>/`, so separate domain jobs do not
replace each other's reports or retry manifests. Paper output remains under
`<dataset.processed_root>/papers/`. The Hydra run directory
contains `plan.json` with both concurrency settings, plus `mineru.server.log`
and `mineru.caller.log` from the allocated node. `run_config.json` also records
the server and client concurrency used by the extraction caller.
The allocated worker also writes `system_info.log`, `gpu_usage.csv`,
`cpu_usage.log`, `process_usage.log`, and `performance_summary.json` to the
Hydra run directory. GPU and CPU samples are taken about every three seconds;
process diagnostics about every 15 seconds. The CPU log is JSON Lines with
one record per allocated logical CPU and an aggregate record. It uses
`mpstat` when available and `/proc/stat` otherwise. An active core has more
than 50% utilization. Missing monitoring tools do not stop extraction.
Set `stage.monitor_usage=false` to skip GPU, CPU, and process sampling and
the derived performance summary. MinerU server and caller logs remain enabled.
For a full-folder scan, the caller uses the selected launcher's PDF and
processed paths but does not open its split-index path. Use `dry_run=true` to
check these resolved paths before submitting a scheduler job.

MinerU parses through the end of each selected PDF by default. Set
`stage.end_page_id=N` (or `--end-page-id N` with the direct CLI) to stop at a
specific zero-based page. Completed papers are skipped on subsequent runs.
With a MinerU server running, benchmark client in-flight counts with
`scripts/extraction/benchmark_mineru_extraction.sh --max-in-flight-values 4 8 16 24 32`.
The report includes papers/sec and pages/sec and recommends only a run that
completed every sampled paper without extraction failures.

Use the job ID from Submitit to inspect the scheduler and logs, or cancel a
still running job. Interrupting the local waiting process with Ctrl-C may
leave the Slurm job running.

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
find logs/extraction/extract_mineru -type f \( -name '*.out' -o -name '*.err' -o -name 'mineru.*.log' \)
scancel JOB_ID
```

A runnable Slurm alternative in this checkout is the separate job script.
It starts MinerU inside the GPU allocation and uses the dedicated MinerU
environment. From the repository root, after creating the split index and
`.venv-mineru`, submit:

```bash
INPUT_DIR="$PWD/data/pdf_datasets" OUTPUT_DIR="$PWD/data/preprocessed" sbatch \
  scripts/extraction/slurm/run_mineru_full_extraction_a100.job \
  --root-dir "$PWD" --split train \
  --split-index "$PWD/data/splits/pdf_dataset_split.json"
```

This script uses
`logs/extraction/slurm/mineru_extract_<job-id>.out` and `.err`, plus
`<job-id>.server.log` and `.caller.log`. Check completion with `sacct`
and the caller log as above. This path is separate from the broken Hydra worker.

### Packaging and review

The packager needs the five PDF source folders and the main environment.
Its `--dry-run` discovers the planned shards without writing. Prepared files
and validation reports are under `huggingface_dataset/`; progress goes to the
terminal. Upload requires Hugging Face authentication and is a separate
network operation. The download wrapper uses `uv run hf` by default; set
`HF_CLI` to the path of an `hf` executable to use it directly.

```bash
scripts/packaging/prepare_huggingface_dataset.sh --input-root data/pdf_datasets --output-dir huggingface_dataset --dry-run
scripts/packaging/prepare_huggingface_dataset.sh --input-root data/pdf_datasets --output-dir huggingface_dataset --prepare-only
scripts/packaging/prepare_huggingface_dataset.sh --input-root data/pdf_datasets --output-dir huggingface_dataset --validate-only
HF_REPO_ID=kasys/open-source-scientific-documents scripts/packaging/upload_huggingface_dataset.sh huggingface_dataset
scripts/packaging/download_huggingface_dataset.sh data/downloaded_hf_dataset
scripts/packaging/restore_pdf_datasets_from_huggingface.sh data/downloaded_hf_dataset data/pdf_datasets --workers 4
```

Once query collections exist under `data/query_collections/`, launch the
local Streamlit reviewer. It listens on `127.0.0.1`; inspect collections in
the browser and find annotations at
`data/reviews/query_review_annotations.jsonl`. Streamlit writes runtime
messages to the terminal.

```bash
scripts/review/run_review_app.sh --server.port 8501
```

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

To add processed data to an existing Hugging Face dataset without uploading or
changing the PDF shards, metadata, or dataset card, run:

```bash
cd /path/to/this/repository
qsub -P "${VTT_PBS_PROJECT:?Set your ABCI group}" \
  -v "VTT_DATA_ROOT=${VTT_DATA_ROOT:?Set the actual dataset root}" \
  scripts/packaging/upload_additional_huggingface.pbs
```

The PBS job requests one `rt_HC` node for three hours and uses eight upload
workers with at most 100,000 files per TAR shard. It uploads the four additional
folders only. Use `-v "VTT_DATA_ROOT=...,HF_REPO_ID=..."` to target another dataset
repository. Authentication must already be available to `hf` on the compute
node (or pass `HF_TOKEN` to PBS through your site's approved secret mechanism).

The data root must contain `preprocessed/`, `clues/`, `query_collections/`, and
`splits/`. Preprocessed and clues must each contain domain folders such as
`ACL/` and `Biology/`. The command packages their contents as uncompressed TAR
shards under `preprocessed/<DOMAIN>/` and `clues/<DOMAIN>/`, preserving paths
within each domain. It uploads `query_collections/` and `splits/` as ordinary
folders, without TAR packaging. Set `VTT_DATA_ROOT` to the dataset root visible
on the compute node. The command scans each domain incrementally, starts building
TAR shards as soon as it has enough files, and overlaps scanning, packaging, and
uploads. It uses four concurrent workers by default; tune this with
`--additional-workers N` according to available network bandwidth and temporary
disk space. Each worker keeps at most one TAR shard in `PBS_LOCALDIR` on ABCI
(otherwise the system temporary directory) and removes it after upload. Peak
temporary space is roughly `N` times the shard size. The command does not modify
the source dataset or the existing `data/` PDF corpus. Adjust shard size with
`--additional-shard-size-gb` (default: 1). TAR shards remain uncompressed;
compression can slow packaging and many source files are already compressed.
For a new upload with many small files, `--additional-max-files-per-shard 10000`
can start uploads before a shard reaches its byte target. Leave it unset when
resuming an existing upload, since changing shard boundaries changes TAR contents.
Each shard is sent through `hf upload`'s folder path so an unchanged TAR already
committed to the Hub is skipped on rerun. The local TAR is still rebuilt to
check its content against the remote copy.
On PBS nodes, uploads use `PBS_LOCALDIR/hf-xet-cache` for Xet scratch data unless
`HF_XET_CACHE` is already set; this also applies to PDF package uploads.
The wrapper uses the existing `.venv` without syncing packages.

For PDF uploads, validation scans the source folders, checks TAR members, and
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
uv run hf download kasys/open-source-scientific-documents --repo-type dataset --local-dir data/downloaded_hf_dataset
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

For a different repository, change the repository ID in the `hf download`
command. For a scratch destination, run:

```bash
uv run hf download kasys/open-source-scientific-documents --repo-type dataset --local-dir /scratch-shared/sfris1
```

Additional arguments are forwarded to `hf download`, for example:

```bash
uv run hf download kasys/open-source-scientific-documents --repo-type dataset --local-dir data/metadata-only \
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
and writes `/scratch-shared/sfris1/splits/pdf_dataset_split.json`.

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

## Extracted and generated artifacts

The stage commands, their prerequisites, and their logs are in [Run the
pipeline](#run-the-pipeline). MinerU success writes
`data/processed/papers/<subset>/<paper_id>/` for local defaults (or the
configured `dataset.processed_root`), with `_SUCCESS`, `markdown.md`,
`paper.json`, `figures.json`, and `mineru/`. Completed papers are skipped on
resume; failed or pending PDFs are recorded in `incomplete_documents.json`.
The reduction and compaction stage copies completed papers to
`data/preprocessed/<subset>/<paper_id>/` for local defaults (or the configured
`dataset.preprocessed`). Figure and textual clue stages
write per-paper JSONL under `data/clues/` with local defaults. Query generation
joins both clue types and writes query collections under
`<dataset.query_output>/<collection_id>/`.

Run `reduce_and_compact` after extraction before generating clues or queries;
it prepares and compacts the separate copy automatically. Its dry-run mode
reports work without copying papers. MinerU can still resume from `processed`.

The operational `extract-mineru-pdfs` CLI remains available for a server that
is **already running**, for example inspect its flags with
`scripts/extraction/run_mineru_full_extraction.sh --help`. This wrapper's
argparse branch is selected only when its first argument is a recognized
`--` flag; it is distinct from the Hydra Slurm workflow.

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
