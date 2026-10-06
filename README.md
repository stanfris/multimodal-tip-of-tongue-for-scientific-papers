# Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers

This repository is part of the paper `Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers`. Here, we provide all of the code necessary to generate all components of our dataset, in order to provide a basis for future work in generating queries and documents. We also provide instructions to only download our dataset, hosted on [HuggingFace](https://huggingface.co/datasets/kasys/open-source-scientific-documents), such that you can use it directly.

Our instructions first explain how to download the full dataset. Further sections explain repository structure, and how to run key elements of the query generation pipeline.
## Downloading the Dataset from Huggingface

Below, we show how to download our data through scripts we provide, which is an intended method for practitioners to build upon our work. If you simply wish download our full datset, you can also apply the Hugging Face cli:
```bash
hf download kasys/open-source-scientific-documents \
  --repo-type dataset \
  --local-dir open-source-scientific-documents
```
This does not handle extraction, workers, or partial components however, we provide these below. The scripts shown below handle each of these components and should be run on a machine with sufficient storage space (at least 500GB of free space). Download selected components into the normal `data/` layout. The downloader
keeps the filtered Hub snapshot and transfer state under
`data/.hf_dataset_download/kasys--open-source-scientific-documents/`,
so rerunning a command resumes downloads and skips matching restored files.
From the repository root, run the script with no arguments to download and restore everything (PDFs, preprocessed content, clues, query collections, and splits) under the default `data/` directory:

```bash
scripts/packaging/download_huggingface_dataset.sh
```

To download all generated components without PDFs, use `--generated`. This also uses `data/` by default:

```bash
scripts/packaging/download_huggingface_dataset.sh --generated
```

To download only PDFs with four restore workers:

```bash
scripts/packaging/download_huggingface_dataset.sh data --pdfs --workers 4
```

The `data` argument is the destination directory; omit it to use `data/`, or replace it with another path.

`--all` explicitly selects every component. `--pdfs` downloads `metadata.parquet` and PDF shards, then restores PDFs under
`data/pdf_datasets/<SOURCE>/` with size and SHA-256 verification. `--generated`
selects parsed content, clues, query collections, and splits. Parsed and clue
TARs extract under `data/preprocessed/<DOMAIN>/` and `data/clues/<DOMAIN>/`;
query and split files copy under their matching directories. Select individual
components with `--preprocessed`, `--clues`, `--query-collections`, and
`--splits`; flags can be combined. `--overwrite` replaces existing files that
differ. `--help` lists the options. Set `HF_REPO_ID` or pass `--repo-id` for
another dataset repository. `duplicates.parquet`, `preparation_report.json`,
and shard manifests are not required.

The standalone PDF restore command remains available for an existing Hub
snapshot. Its target contains source folders such as `ACL`, `Physics`, and
`Medicine`:

```bash
scripts/packaging/restore_pdf_datasets_from_huggingface.sh \
  data/.hf_dataset_download/kasys--open-source-scientific-documents \
  data/pdf_datasets
```

The restore command checks every PDF's recorded size and SHA-256 checksum, and
will not replace existing PDFs unless `--overwrite` is supplied. For faster
restoration, process independent TAR shards concurrently with `--workers`.
Use `--skip-checksum` only for trusted archives; size checks still apply:

```bash
scripts/packaging/restore_pdf_datasets_from_huggingface.sh \
  data/.hf_dataset_download/kasys--open-source-scientific-documents \
  data/pdf_datasets \
  --workers 4 --skip-checksum
```

Our code provides the way in which our dataset is uploaded, which may be useful for reproducibility and transparency purposes. We provide some examples to upload our data, but this is not an integral part of the pipeline for dataset creation.


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
  dataset_packaging/     # Hugging Face packaging and restoration
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
pipeline and MinerU 3.x require incompatible Transformers versions. Install [`uv`](https://docs.astral.sh/uv/), then create and activate the environment:

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
authenticated Kaggle CLI or an existing metadata snapshot.

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
`data/splits/pdf_dataset_split.json`. Rerunning the split command preserves an existing index
when membership is unchanged and refuses to replace a different split unless
`--overwrite` is supplied.

### MinerU extraction

We apply MinerU for extraction of figures, with a configuration that allows for the running of full extraction on a single GPU at once, with optional configurations to parse a single dataset at a time. For settings or help to apply/tune MinerU, please refer to the MinerU documentation. Below, we show commands to run extraction for the pdf_dataset in this repository, one just for the train set, and another which applies all pdfs for the ACL domain set. Apply `--help` to view the available options.

```bash
scripts/extraction/run_mineru_full_extraction.sh split=train launcher=pbs_h200
scripts/extraction/run_mineru_full_extraction.sh split=train+test launcher=slurm_a100 stage.all_domain_pdfs=true 'stage.domains=[ACL]' -m
```

Reduction copies completed papers from `dataset.processed` to
`dataset.preprocessed`, then compacts the copies there, cleaning up files from MinerU processes, as well as excluding equations, which are not applied in our work for visual components. Note that MinerU's original files
remain in `processed`. 
```bash
scripts/preprocessing/reduce_and_compact_preprocessed.sh
```


### Clue and query Generation

Clue generation and query generation are performed using `transformers` and `vllm` backend. `transformers` performs padded model batches, and `vllm` submits the whole window to the offline engine for continuous batching. Backend packages are imported only when selected. Existing clue and query output paths, JSONL rows, and resume keys are unchanged.

Select the backend with `model.provider=transformers` or `model.provider=vllm`. Ensure that VLLM and Cuda are available in your environment before appying it.
The model choices are set to the defaults applied in our work. Query generation can be run on compute clusters, using for example slurm or pbs, which can be found in the launcher configuration. Please adjust these for your cluster. Below, we show several examples of how to run the pipeline on a compute cluster.

```bash
scripts/clue_generation/describe_all_figures.sh split=train launcher=pbs_h200 model=qwen3_vl model.provider=vllm
scripts/clue_generation/describe_all_textual_clues.sh split=train launcher=pbs_h200 model=qwen3_text model.provider=vllm
scripts/query_generation/generate_queries.sh split=train launcher=pbs_h200 model=phi4 model.provider=vllm
scripts/query_generation/judge_train_queries.sh split=train launcher=pbs_h200 model=gemma3_judge model.provider=vllm
```


### Annotation

The `annotations/` folder provides the annotations and the web-pages used to perform an analysis of tip-of-the-tongue queries found on the internet. The notebook `annotations/analysis.ipynb` provides the code which was used to compute percentages and annotator agreement. 



### License
This repository uses the [MIT License](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers/blob/main/LICENSE). 