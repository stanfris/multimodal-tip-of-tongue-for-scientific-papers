#!/usr/bin/env python3
"""Create a small, query-linked ZIP from the PDF test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


DATASETS = ("ACL", "Biology", "Engineering", "Medicine", "Physics")
DEFAULT_REPO = "kasys/open-source-scientific-documents"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--output", type=Path, default=Path("data/test_sample_20_per_dataset.zip"))
    result.add_argument("--count", type=int, default=20, help="PDFs per dataset (default: 20)")
    result.add_argument("--repo-id", default=DEFAULT_REPO)
    result.add_argument("--offline", action="store_true", help="Use only files already under --data-root")
    return result


def safe_split_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) < 2 or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe PDF split path: {value!r}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Split entry is not a PDF: {value!r}")
    return path


def paper_id(path: PurePosixPath) -> str:
    value = path.with_suffix("").as_posix()
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "unknown"


def matching_paper_id(path: PurePosixPath, queries: dict[str, list[dict]]) -> str | None:
    primary = paper_id(path)
    if primary in queries:
        return primary
    collision = f"{primary}.{hashlib.sha256(path.as_posix().encode()).hexdigest()[:12]}"
    return collision if collision in queries else None


def load_queries(root: Path, dataset: str) -> dict[str, list[dict]]:
    by_paper: dict[str, list[dict]] = defaultdict(list)
    collection = root / "query_collections" / dataset / "query_generation_test"
    for query_file in sorted(collection.glob("*/queries.jsonl")):
        for line_number, line in enumerate(query_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            ids = row.get("relevant_ids") or []
            document_id = str(metadata.get("paper_id") or (ids[0] if ids else ""))
            query = row.get("query")
            if document_id and isinstance(query, str) and query.strip():
                by_paper[document_id].append({
                    "query_id": row.get("query_id"),
                    "mode": query_file.parent.name,
                    "query": query,
                })
            else:
                raise ValueError(f"Invalid query in {query_file}:{line_number}")
    return by_paper


def download_small_files(root: Path, repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["splits/pdf_dataset_split.json", "query_collections/*/query_generation_test/*/queries.jsonl"],
    ))


def metadata_index(repo_id: str) -> dict[tuple[str, str], dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(repo_id, "metadata.parquet", repo_type="dataset")
    rows = pd.read_parquet(metadata_path, columns=[
        "source", "original_relative_path", "shard", "member_path"
    ])
    return {(str(row.source), str(row.original_relative_path)): row._asdict()
            for row in rows.itertuples(index=False)}


def main() -> int:
    args = parser().parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_id):
        raise ValueError("Invalid Hugging Face dataset repository ID")
    local_root = args.data_root.expanduser().resolve()
    split_path = local_root / "splits/pdf_dataset_split.json"
    query_roots = [local_root]
    if not args.offline and (not split_path.is_file() or any(
        not list((local_root / "query_collections" / dataset / "query_generation_test").glob("*/queries.jsonl"))
        for dataset in DATASETS
    )):
        query_roots.append(download_small_files(local_root, args.repo_id))
    split_path = next((root / "splits/pdf_dataset_split.json" for root in query_roots
                       if (root / "splits/pdf_dataset_split.json").is_file()), split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing test split: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_paths = [safe_split_path(value) for value in split["test"]]
    selected = []
    for dataset in DATASETS:
        root = next((root for root in query_roots if list(
            (root / "query_collections" / dataset / "query_generation_test").glob("*/queries.jsonl")
        )), None)
        if root is None:
            raise FileNotFoundError(f"Missing test queries for {dataset}")
        queries = load_queries(root, dataset)
        candidates = [(path, matched, queries[matched]) for path in test_paths
                      if path.parts[0] == dataset
                      if (matched := matching_paper_id(path, queries)) is not None]
        if len(candidates) < args.count:
            raise ValueError(f"{dataset}: only {len(candidates)} test PDFs have queries; need {args.count}")
        selected.extend((dataset, path, matched, query_rows)
                        for path, matched, query_rows in candidates[:args.count])

    remote_index = None
    manifest = []
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for dataset, path, document_id, query_rows in selected:
                local_pdf = local_root / "pdf_datasets" / Path(*path.parts)
                zip_path = f"pdfs/{path.as_posix()}"
                if local_pdf.is_file():
                    archive.write(local_pdf, zip_path)
                else:
                    if args.offline:
                        raise FileNotFoundError(f"Missing local PDF: {local_pdf}")
                    if remote_index is None:
                        remote_index = metadata_index(args.repo_id)
                    entry = remote_index.get((dataset, PurePosixPath(*path.parts[1:]).as_posix()))
                    if entry is None:
                        raise KeyError(f"PDF absent from Hub metadata: {path}")
                    from huggingface_hub import hf_hub_download

                    shard = hf_hub_download(args.repo_id, entry["shard"], repo_type="dataset")
                    with tarfile.open(shard, "r") as tar:
                        member = tar.extractfile(entry["member_path"])
                        if member is None:
                            raise FileNotFoundError(f"Missing {entry['member_path']} in {entry['shard']}")
                        archive.writestr(zip_path, member.read())
                for query_row in query_rows:
                    manifest.append({
                        "dataset": dataset,
                        "document_id": document_id,
                        "pdf": zip_path,
                        **query_row,
                    })
            archive.writestr("queries.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Created {output}: {len(selected)} PDFs, {len(manifest)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
