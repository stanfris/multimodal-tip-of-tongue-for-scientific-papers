"""Run MinerU extraction without importing the main project's full CLI graph."""

from __future__ import annotations

from extraction.mineru_extraction import build_extract_parser, run_extract


def main() -> int:
    args = build_extract_parser().parse_args()
    run_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
