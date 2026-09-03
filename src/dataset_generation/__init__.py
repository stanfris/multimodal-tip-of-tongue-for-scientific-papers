"""Dataset generation package for ACL-fig plus ACL Anthology markdown."""

from dataset_generation.build import BuildConfig, build_combined_dataset
from dataset_generation.matching import (
    extract_paper_id_from_filename,
    match_records_to_papers,
    normalize_id,
)

__all__ = [
    "BuildConfig",
    "build_combined_dataset",
    "extract_paper_id_from_filename",
    "match_records_to_papers",
    "normalize_id",
]
