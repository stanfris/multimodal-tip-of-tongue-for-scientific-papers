#!/usr/bin/env python3
"""Prepare, validate, and upload the Hugging Face PDF corpus dataset."""

from __future__ import annotations

from dataset_generation.hf_dataset_packaging import main


if __name__ == "__main__":
    raise SystemExit(main())
