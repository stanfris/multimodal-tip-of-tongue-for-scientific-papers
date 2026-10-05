---
license: other
language:
- en
tags:
- scientific-papers
- multimodal
- pdf
- tip-of-the-tongue
pretty_name: Open-Source Scientific Documents
---

# Open-Source Scientific Documents

This repository contains the scientific-paper corpus and generated artifacts for **Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers**. It brings together source PDFs, extracted paper content, textual and visual clues, query collections, and split definitions. The [generation code](https://github.com/stanfris/multimodal-tip-of-tongue-for-scientific-papers) documents how these artifacts are made.

The PDFs form a shared retrieval corpus. Query and evaluation collections refer to paper identifiers in that corpus; the repository is not organized as separate train and test copies of every PDF.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/<SOURCE>/shard-*.tar` | Source PDFs, each paired with a JSON sidecar |
| `metadata.parquet` | PDF index: identifiers, original paths, shard members, checksums, and available bibliographic fields |
| `preprocessed/<DOMAIN>/shard-*.tar` | Parsed paper text, figures, and other extraction output, with paths relative to the domain |
| `clues/<DOMAIN>/shard-*.tar` | Generated textual and visual clues, with paths relative to the domain |
| `query_collections/` | Generated query and evaluation collections |
| `splits/` | Split definitions and related indexes |

The source domains are `ACL` (computational linguistics), `Physics` and `Engineering` (arXiv), and `Biology` and `Medicine` (PMC Open Access). Some collections may include additional metadata files at the root of their folder.

## PDF index and shards

The PDF package contains 108,512 entries across 108,512 distinct SHA-256 contents. Counts by source:

- `ACL`: 20,845 PDFs
- `Biology`: 22,000 PDFs
- `Engineering`: 22,000 PDFs
- `Medicine`: 22,000 PDFs
- `Physics`: 21,667 PDFs

Each uncompressed PDF TAR stores pairs named `DOCUMENT_ID.pdf` and `DOCUMENT_ID.json`. The JSON sidecar records the original source and path, SHA-256 checksum, and available title, year, DOI, and license information. `metadata.parquet` indexes the same PDFs and includes `shard` and `member_path`, so a document can be located without scanning every archive. Its columns are:

`document_id`, `source`, `original_filename`, `original_relative_path`, `shard`, `member_path`, `size_bytes`, `sha256`, `license`, `title`, `year`, `doi`

The counts above describe packaged entries, not necessarily distinct papers: exact duplicate PDF bytes may occur across sources. Use the `sha256` column when an experiment requires content deduplication. Bibliographic records can also describe different versions of the same paper, which a byte checksum will not detect.

## Download and use

Install `huggingface_hub` to download selected files. This example retrieves the PDF index and one PDF by its `document_id`:

```python
import pandas as pd
import tarfile
from huggingface_hub import hf_hub_download

repo_id = "kasys/open-source-scientific-documents"
index_path = hf_hub_download(repo_id, "metadata.parquet", repo_type="dataset")
index = pd.read_parquet(index_path)

document_id = "YOUR_DOCUMENT_ID"
row = index.loc[index["document_id"] == document_id].iloc[0]
shard_path = hf_hub_download(repo_id, row["shard"], repo_type="dataset")
with tarfile.open(shard_path, "r") as archive:
    pdf_bytes = archive.extractfile(row["member_path"]).read()
```

To download the entire repository:

```bash
hf download kasys/open-source-scientific-documents --repo-type dataset --local-dir open-source-scientific-documents
```

The `preprocessed/` and `clues/` archives preserve relative paths inside each domain. For example, `preprocessed/ACL/shard-00000.tar` extracts into a local `preprocessed/ACL/` folder. Inspect TAR members before extracting archives from any untrusted source. Query collections and split files are ordinary files that can be downloaded individually.

The [repository's restore script](https://github.com/stanfris/multimodal-tip-of-tongue-for-scientific-papers/blob/main/scripts/packaging/restore_pdf_datasets_from_huggingface.sh) uses `metadata.parquet` to reconstruct the original five-domain PDF layout and verify checksums.

## Provenance and licenses

Source selection is performed by the generation pipeline before packaging. The PDF sources are ACL Anthology, selected open-reuse arXiv records, and selected PMC Open Access records. License information is **per document** where available; there is no single license that covers every PDF or generated artifact. Check the sidecar or `metadata.parquet` entry and the source publication before reuse. Missing license or bibliographic fields are left empty rather than inferred.

## Limitations

Text extraction and generated clues or queries may contain errors. Some papers lack complete bibliographic or license metadata. A SHA-256 match identifies identical files, not all semantically duplicate papers. The TAR archives are uncompressed because many PDF and image files are already compressed.
