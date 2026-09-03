"""Source dataset loading and ACL-fig record extraction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import Image, load_dataset, load_from_disk
from PIL import Image as PILImage

from dataset_generation.matching import extract_paper_id_from_filename, normalize_id


ACL_FIG_DATASET = "citeseerx/ACL-fig"
ACL_ANTHOLOGY_MD_DATASET = "KRLabsOrg/acl-anthology-md"
ACL_ANTHOLOGY_MD_CONFIG = "fulltext"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROCESSED_DATASET = Path("processed") / "acl_fig_markdown"


def default_processed_dataset_dir(data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / DEFAULT_PROCESSED_DATASET / split


def dataset_cache_path(data_dir: str | Path, dataset_name: str, split: str, config: str | None = None) -> Path:
    dataset_slug = dataset_name.replace("/", "__")
    name = f"{dataset_slug}__{config}__{split}" if config else f"{dataset_slug}__{split}"
    return Path(data_dir) / "raw" / "hf" / name


def download_hf_sources(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    fig_split: str = "train",
    paper_split: str = "train",
) -> dict[str, Path]:
    """Download source HF datasets once and save them under data/raw/hf."""
    fig_path = dataset_cache_path(data_dir, ACL_FIG_DATASET, fig_split)
    paper_path = dataset_cache_path(data_dir, ACL_ANTHOLOGY_MD_DATASET, paper_split, ACL_ANTHOLOGY_MD_CONFIG)

    if not fig_path.exists():
        fig_dataset = load_dataset(ACL_FIG_DATASET, split=fig_split)
        fig_dataset.save_to_disk(fig_path)

    if not paper_path.exists():
        paper_dataset = load_dataset(ACL_ANTHOLOGY_MD_DATASET, ACL_ANTHOLOGY_MD_CONFIG, split=paper_split)
        paper_dataset.save_to_disk(paper_path)

    return {"figure": fig_path, "paper_markdown": paper_path}


def stable_record_id(filename: str, index: int, split: str) -> str:
    key = f"{split}:{index}:{filename}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()


def image_filename(image: Any, item: dict[str, Any]) -> str:
    if isinstance(image, dict):
        return str(image.get("path") or item.get("filename") or "")
    return str(getattr(image, "filename", "") or item.get("filename", ""))


def image_metadata(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if image_bytes:
            with PILImage.open(BytesIO(image_bytes)) as decoded:
                return {
                    "image_width": decoded.width,
                    "image_height": decoded.height,
                    "image_mode": decoded.mode,
                }
        return {"image_width": None, "image_height": None, "image_mode": None}
    return {
        "image_width": getattr(image, "width", None),
        "image_height": getattr(image, "height", None),
        "image_mode": getattr(image, "mode", None),
    }


def extract_acl_fig_record(
    item: dict[str, Any],
    index: int,
    split: str,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    image = item.get("image")
    filename = image_filename(image, item)
    metadata = image_metadata(image)
    extracted_id = extract_paper_id_from_filename(filename)
    normalized_id = normalize_id(extracted_id)
    raw_label = item.get("label")
    label = label_names[raw_label] if label_names and isinstance(raw_label, int) else raw_label

    return {
        "record_id": stable_record_id(filename, index, split),
        "filename": filename,
        "extracted_paper_id": extracted_id,
        "normalized_paper_id": normalized_id,
        "extracted_id": extracted_id,
        "normalized_id": normalized_id,
        "label": label,
        "label_id": raw_label if isinstance(raw_label, int) else None,
        **metadata,
        "source_fig_dataset": ACL_FIG_DATASET,
        "source_fig_split": split,
    }


def load_acl_fig_records(
    split: str = "train",
    limit: int | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    prefer_local: bool = True,
) -> list[dict[str, Any]]:
    local_path = dataset_cache_path(data_dir, ACL_FIG_DATASET, split) if data_dir is not None else None
    if prefer_local and local_path is not None and local_path.exists():
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset(ACL_FIG_DATASET, split=split)
    dataset = dataset.cast_column("image", Image(decode=False))
    label_feature = getattr(dataset.features.get("label"), "names", None)
    iterable = enumerate(dataset)
    if limit is not None:
        iterable = islice(iterable, limit)
    return [
        extract_acl_fig_record(item, index=index, split=split, label_names=label_feature)
        for index, item in iterable
    ]


def materialize_acl_fig_images(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    split: str = "train",
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    prefer_local: bool = True,
    image_dir_name: str = "images",
) -> dict[str, Any]:
    """Write ACL-Fig image files beside an artifact and add row references."""
    if not records:
        return {"image_records": 0, "images_written": 0, "missing_images": 0}

    output_path = Path(output_dir)
    image_dir = output_path / image_dir_name
    image_dir.mkdir(parents=True, exist_ok=True)

    by_record_id = {str(record["record_id"]): record for record in records if record.get("record_id")}
    local_path = dataset_cache_path(data_dir, ACL_FIG_DATASET, split) if data_dir is not None else None
    if prefer_local and local_path is not None and local_path.exists():
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset(ACL_FIG_DATASET, split=split)
    dataset = dataset.cast_column("image", Image(decode=False))

    images_written = 0
    for index, item in enumerate(dataset):
        image = item.get("image")
        filename = image_filename(image, item)
        record_id = stable_record_id(filename, index, split)
        record = by_record_id.get(record_id)
        if record is None:
            continue

        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not image_bytes:
            continue

        suffix = Path(filename).suffix or ".png"
        image_relpath = Path(image_dir_name) / f"{record_id}{suffix}"
        image_path = output_path / image_relpath
        image_path.write_bytes(image_bytes)
        record["image_relpath"] = image_relpath.as_posix()
        record["image_path"] = str(image_path.resolve())
        images_written += 1

        if images_written == len(by_record_id):
            break

    missing_images = len(by_record_id) - images_written
    return {
        "image_records": len(by_record_id),
        "images_written": images_written,
        "missing_images": missing_images,
        "image_dir": image_dir_name,
    }


def load_acl_markdown_papers(
    split: str = "train",
    streaming: bool = False,
    limit: int | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    prefer_local: bool = True,
) -> Iterable[dict[str, Any]]:
    local_path = (
        dataset_cache_path(data_dir, ACL_ANTHOLOGY_MD_DATASET, split, ACL_ANTHOLOGY_MD_CONFIG)
        if data_dir is not None
        else None
    )
    if prefer_local and local_path is not None and local_path.exists():
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset(
            ACL_ANTHOLOGY_MD_DATASET,
            ACL_ANTHOLOGY_MD_CONFIG,
            split=split,
            streaming=streaming,
        )
    if limit is None:
        return dataset
    return islice(dataset, limit)


def iter_with_limit(items: Iterable[dict[str, Any]], limit: int | None = None) -> Iterator[dict[str, Any]]:
    yield from islice(items, limit) if limit is not None else items
