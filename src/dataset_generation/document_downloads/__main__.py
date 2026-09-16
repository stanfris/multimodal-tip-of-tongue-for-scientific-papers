"""Run the centralized document downloader as a module."""

from __future__ import annotations

import sys

from dataset_generation.document_downloads.cli import main


if __name__ == "__main__":
    sys.exit(main())
