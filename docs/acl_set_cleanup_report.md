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
- Extract markdown and figures from PDFs with MinerU into
  `data/processed/mineru_pdf_extraction/papers/`.
- Build the canonical dataset in `data/canonical/`.

## Essential Stages To Preserve

The prompt-backed stages remain essential, but they must consume the canonical
dataset produced from the ACL subset path:

- `04_describe_all_figures.sh` reads `data/canonical/papers.jsonl` and writes
  `data/canonical/visual_clues.jsonl`.
- `05_describe_all_textual_clues.sh` reads `data/canonical/papers.jsonl` and
  writes `data/canonical/textual_clues.jsonl`.
- `06_generate_queries.sh` reads canonical papers and clue sidecars, then writes
  query collections plus canonical `queries.jsonl` when resume is enabled.

## Script Order

The cleaned order should be:

- `00_sync_env.sh`
- `01_build_acl_subset.sh`
- `02_download_acl_pdfs.sh`
- `03_build_canonical.sh`
- `04_describe_all_figures.sh`
- `05_describe_all_textual_clues.sh`
- `06_generate_queries.sh`

MinerU service and extraction helpers remain available as operational helpers:

- `08_start_mineru_router.sh`
- `09_run_mineru_full_extraction.sh`
- `10_reduce_and_compact_preprocessed.sh`

## Implementation Notes

`03_build_canonical.sh` should no longer default to
`data/processed/acl_fig_markdown/<split>`. It should build canonical papers from
the ACL subset extraction output, preserving paper IDs, markdown, figures, local
image copies, source PDF references, and useful MinerU metadata.
