#!/usr/bin/env python3
"""Publish the dataset card and remove upload-only artifacts from the Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, RepoFile


DEFAULT_REPO = "kasys/open-source-scientific-documents"
DEFAULT_CARD = Path(__file__).resolve().parents[2] / "docs" / "huggingface_dataset_README.md"
ROOT_BUILD_FILES = {"duplicates.parquet", "preparation_report.json"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--apply", action="store_true", help="Publish the card and delete the listed build files.")
    return parser


def obsolete_paths(api: HfApi, repo_id: str) -> list[str]:
    paths = set()
    for entry in api.list_repo_tree(repo_id, recursive=False, repo_type="dataset"):
        if isinstance(entry, RepoFile) and entry.path in ROOT_BUILD_FILES:
            paths.add(entry.path)
    for entry in api.list_repo_tree(repo_id, path_in_repo="data", recursive=True, repo_type="dataset"):
        if isinstance(entry, RepoFile) and entry.path.endswith(".manifest.json"):
            paths.add(entry.path)
    return sorted(paths)


def main() -> None:
    args = build_parser().parse_args()
    card = args.card.expanduser().resolve()
    if not card.is_file():
        raise FileNotFoundError(card)
    api = HfApi()
    deletes = obsolete_paths(api, args.repo_id)
    print(json.dumps({"repo_id": args.repo_id, "card": str(card), "delete_count": len(deletes), "delete_paths": deletes}, indent=2))
    if args.apply:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update dataset card and remove build artifacts",
            operations=[
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
                *(CommitOperationDelete(path_in_repo=path) for path in deletes),
            ],
        )
        print("Published README.md and removed listed build files.")


if __name__ == "__main__":
    main()
