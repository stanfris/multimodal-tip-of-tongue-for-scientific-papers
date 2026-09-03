#!/usr/bin/env python
"""CLI wrapper for dataset_generation.acl_subset."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_generation.acl_subset import main  # noqa: E402


if __name__ == "__main__":
    main()
