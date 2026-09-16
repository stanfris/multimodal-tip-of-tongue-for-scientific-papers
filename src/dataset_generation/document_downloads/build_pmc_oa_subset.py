#!/usr/bin/env python
"""CLI wrapper for the strict PMC OA Biology/Medical dataset builder."""

from __future__ import annotations

import json
import logging

from dataset_generation.document_downloads.pmc_oa_subset import build_parser, run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
