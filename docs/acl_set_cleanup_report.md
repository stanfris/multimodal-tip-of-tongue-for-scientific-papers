# ACL Set Cleanup Report

## Deprecated Path To Remove

The codebase still exposes an older dataset path built from `citeseerx/ACL-fig`
plus `KRLabsOrg/acl-anthology-md`. That path is deprecated for current work.

Deprecated pieces:

- `dataset-generation download-sources`, which downloads the Hugging Face figure
  and markdown datasets.
- `dataset-generation build`, which joins ACL-fig image rows to markdown rows.
- `dataset-generation run` with the `acl_fig_markdown` parser configs.
- `scripts/01_download_sources.sh` and `scripts/02_build_dataset.sh`.
- Managed run configs under `configs/runs/acl_fig_markdown_*.yaml`.
- Parser/matching modules dedicated to the ACL-fig plus markdown join.

## Current Path To Keep

The current source path should be the curated ACL Anthology subset:

- Build official ACL Anthology metadata into `data/acl_subset/papers.jsonl`.
- Download subset PDFs into `data/acl_subset/pdfs/`.
- Extract markdown and figures from PDFs with MinerU into `data/preprocessed`.
- Generate clues and queries directly from `data/preprocessed` or
  `data/preprocessed/papers`.

## Essential Stages To Preserve

The prompt-backed stages remain essential, but they must consume the
preprocessed extraction output directly:

- `06_describe_all_figures.sh` reads preprocessed paper directories and writes
  `data/clues/<paper_id>/images/<figure_id>.jsonl`.
- `07_describe_all_textual_clues.sh` reads preprocessed paper directories and
  writes `data/clues/<paper_id>/base/textual_clues.jsonl`.
- `08_generate_queries.sh` reads preprocessed paper directories and per-paper
  clue files, then writes query collections plus `data/clues/queries.jsonl`
  when resume is enabled.

## Script Order

The cleaned order should be:

- `00_sync_env.sh`
- `01_build_acl_subset.sh`
- `02_download_acl_pdfs.sh`
- `03_start_mineru_router.sh`
- `04_run_mineru_full_extraction.sh`
- `05_reduce_and_compact_preprocessed.sh`
- `06_describe_all_figures.sh`
- `07_describe_all_textual_clues.sh`
- `08_generate_queries.sh`

## Implementation Notes

There is no intermediate dataset preparation stage. Clue and query generation
read the ACL subset extraction output in place.
