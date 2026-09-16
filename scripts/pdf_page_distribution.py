#!/usr/bin/env python3
"""Fast page-count statistics for the local PMC, arXiv, and ACL PDF sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BINS = (1, 4, 8, 12, 16, 20, 30, 40, 60, 100)
DEFAULT_MAX_DOCS_PER_SUBSET = 22_000
PAGE_RE = re.compile(rb"/Type\s*/Page\b")
NOT_PAGE_RE = re.compile(rb"/Type\s*/Pages\b")


@dataclass(frozen=True)
class PdfRecord:
    set_name: str
    path: Path
    pages: int | None
    error: str | None = None


@dataclass(frozen=True)
class PruneDecision:
    record: PdfRecord
    subset: str
    action: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count pages for PDFs in data/pmc*/pdfs, data/arxiv_open_reuse/pdfs, "
            "and data/acl_subset/pdfs, then write per-file and distribution stats."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pdf_page_stats"))
    parser.add_argument("--workers", type=int, default=min(64, (os.cpu_count() or 1) * 4))
    parser.add_argument(
        "--bins",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BINS),
        help="Comma-separated inclusive upper bounds for histogram bins.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail PDFs that pdfinfo cannot read instead of using the quick byte-scan fallback.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help=(
            "Delete PDFs with 1-4 pages or 30+ pages, then cap each subset by deleting "
            "the longest remaining PDFs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --prune, write deletion reports without deleting files.",
    )
    parser.add_argument(
        "--max-docs-per-subset",
        type=int,
        default=DEFAULT_MAX_DOCS_PER_SUBSET,
        help="Maximum retained PDFs for each of: PMC biology, PMC medical, arXiv physics, arXiv engineering, ACL.",
    )
    return parser.parse_args()


def discover_sets(data_root: Path) -> dict[str, list[Path]]:
    sets: dict[str, list[Path]] = {
        "pmc": sorted(data_root.glob("pmc*/pdfs/**/*.pdf")),
        "arxiv": sorted((data_root / "arxiv_open_reuse" / "pdfs").glob("**/*.pdf")),
        "acl": sorted((data_root / "acl_subset" / "pdfs").glob("**/*.pdf")),
    }
    return {name: paths for name, paths in sets.items() if paths}


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_arxiv_domain_map(data_root: Path) -> dict[str, str]:
    arxiv_root = data_root / "arxiv_open_reuse"
    domain_by_filename: dict[str, str] = {}
    for manifest_name in ("eligible_records.jsonl", "domain_records.jsonl", "metadata_records.jsonl"):
        for row in iter_jsonl(arxiv_root / manifest_name):
            filename = row.get("pdf_filename")
            if not isinstance(filename, str):
                arxiv_id = row.get("arxiv_id")
                if isinstance(arxiv_id, str):
                    filename = f"{arxiv_id.replace('/', '_')}.pdf"
            if not filename:
                continue

            domain_by_filename[filename] = arxiv_subset_from_manifest_row(row)
    return domain_by_filename


def arxiv_subset_from_manifest_row(row: dict[str, object]) -> str:
    selection_domain = str(row.get("selection_domain") or "").lower()
    if selection_domain.startswith("eess."):
        return "arxiv_engineering"
    if selection_domain.startswith("physics."):
        return "arxiv_physics"

    broad_domains = row.get("broad_domains")
    domains = [str(value).lower() for value in broad_domains] if isinstance(broad_domains, list) else []
    source_query = str(row.get("source_api_query") or "").lower()
    primary_category = str(row.get("primary_category") or "").lower()
    categories = row.get("categories") or row.get("subject_categories") or row.get("category_matches")
    category_values = [str(value).lower() for value in categories] if isinstance(categories, list) else []
    has_eess = any(category.startswith("eess.") for category in category_values)
    has_physics = any(category.startswith("physics.") for category in category_values)

    if "engineering" in domains and "physics" not in domains:
        return "arxiv_engineering"
    if "physics" in domains and "engineering" not in domains:
        return "arxiv_physics"
    if source_query.startswith("eess"):
        return "arxiv_engineering"
    if source_query.startswith("physics"):
        return "arxiv_physics"
    if has_eess and not has_physics:
        return "arxiv_engineering"
    if has_physics and not has_eess:
        return "arxiv_physics"
    if primary_category.startswith("eess"):
        return "arxiv_engineering"
    return "arxiv_physics"


def subset_for_path(path: Path, data_root: Path, arxiv_domain_map: dict[str, str]) -> str:
    try:
        relative = path.relative_to(data_root)
    except ValueError:
        relative = path
    parts = relative.parts
    if len(parts) >= 4 and parts[0].startswith("pmc") and parts[1] == "pdfs":
        category = parts[2].lower()
        if category == "biology":
            return "pmc_biology"
        if category in {"medical_clinical_research", "medical/clinical research"}:
            return "pmc_medical"
    if len(parts) >= 3 and parts[0] == "arxiv_open_reuse" and parts[1] == "pdfs":
        return arxiv_domain_map.get(path.name, "arxiv_unknown")
    if len(parts) >= 3 and parts[0] == "acl_subset" and parts[1] == "pdfs":
        return "acl"
    return "unknown"


def count_with_pdfinfo(path: Path, pdfinfo_bin: str) -> int:
    proc = subprocess.run(
        [pdfinfo_bin, str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"pdfinfo exited with {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("pdfinfo output did not include a Pages field")


def count_with_byte_scan(path: Path) -> int:
    data = path.read_bytes()
    page_like = len(PAGE_RE.findall(data))
    page_trees = len(NOT_PAGE_RE.findall(data))
    pages = page_like - page_trees
    if pages < 1:
        raise RuntimeError("fallback byte scan found no pages")
    return pages


def count_one(set_name: str, path: Path, pdfinfo_bin: str | None, use_fallback: bool) -> PdfRecord:
    try:
        if pdfinfo_bin:
            return PdfRecord(set_name=set_name, path=path, pages=count_with_pdfinfo(path, pdfinfo_bin))
        if use_fallback:
            return PdfRecord(set_name=set_name, path=path, pages=count_with_byte_scan(path))
        return PdfRecord(set_name=set_name, path=path, pages=None, error="pdfinfo not found")
    except Exception as exc:
        if use_fallback:
            try:
                return PdfRecord(set_name=set_name, path=path, pages=count_with_byte_scan(path))
            except Exception as fallback_exc:
                return PdfRecord(set_name=set_name, path=path, pages=None, error=f"{exc}; fallback: {fallback_exc}")
        return PdfRecord(set_name=set_name, path=path, pages=None, error=str(exc))


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def histogram(values: Iterable[int], bins: list[int]) -> list[dict[str, int | str]]:
    counts = [0] * (len(bins) + 1)
    for value in values:
        for idx, upper in enumerate(bins):
            if value <= upper:
                counts[idx] += 1
                break
        else:
            counts[-1] += 1

    labels: list[str] = []
    lower = 1
    for upper in bins:
        labels.append(str(upper) if lower == upper else f"{lower}-{upper}")
        lower = upper + 1
    labels.append(f"{lower}+")
    return [{"range": label, "count": count} for label, count in zip(labels, counts)]


def summarize(records: list[PdfRecord], bins: list[int]) -> dict[str, object]:
    values = sorted(record.pages for record in records if record.pages is not None)
    failures = [record for record in records if record.pages is None]
    if not values:
        return {
            "pdfs": len(records),
            "counted": 0,
            "failed": len(failures),
            "histogram": histogram([], bins),
        }

    return {
        "pdfs": len(records),
        "counted": len(values),
        "failed": len(failures),
        "min": values[0],
        "p25": round(percentile(values, 0.25), 2),
        "median": round(percentile(values, 0.50), 2),
        "mean": round(statistics.fmean(values), 2),
        "p75": round(percentile(values, 0.75), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": values[-1],
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "histogram": histogram(values, bins),
    }


def plan_pruning(
    records: list[PdfRecord],
    data_root: Path,
    max_docs_per_subset: int,
) -> list[PruneDecision]:
    arxiv_domain_map = load_arxiv_domain_map(data_root)
    decisions: list[PruneDecision] = []
    retained_by_subset: dict[str, list[PdfRecord]] = {}

    for record in records:
        subset = subset_for_path(record.path, data_root, arxiv_domain_map)
        if record.pages is None:
            decisions.append(PruneDecision(record, subset, "keep", "page-count-failed"))
        elif record.pages <= 4:
            decisions.append(PruneDecision(record, subset, "delete", "too-short-1-4-pages"))
        elif record.pages >= 30:
            decisions.append(PruneDecision(record, subset, "delete", "too-long-30-plus-pages"))
        else:
            retained_by_subset.setdefault(subset, []).append(record)

    capped_subsets = {"pmc_biology", "pmc_medical", "arxiv_physics", "arxiv_engineering", "acl"}
    delete_for_cap: set[Path] = set()
    for subset, retained in retained_by_subset.items():
        if subset not in capped_subsets or len(retained) <= max_docs_per_subset:
            continue
        longest_first = sorted(
            retained,
            key=lambda record: (record.pages or -1, str(record.path)),
            reverse=True,
        )
        delete_for_cap.update(record.path for record in longest_first[: len(retained) - max_docs_per_subset])

    already_decided = {decision.record.path for decision in decisions}
    for subset, retained in retained_by_subset.items():
        for record in retained:
            if record.path in already_decided:
                continue
            if record.path in delete_for_cap:
                decisions.append(PruneDecision(record, subset, "delete", f"cap-{max_docs_per_subset}-longest-prune"))
            else:
                decisions.append(PruneDecision(record, subset, "keep", "within-page-and-subset-limits"))

    return sorted(decisions, key=lambda decision: (decision.subset, decision.action, str(decision.record.path)))


def summarize_pruning(decisions: list[PruneDecision]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for decision in decisions:
        subset_stats = summary.setdefault(
            decision.subset,
            {
                "before": 0,
                "deleted": 0,
                "kept": 0,
                "deleted_by_reason": {},
            },
        )
        subset_stats["before"] += 1
        if decision.action == "delete":
            subset_stats["deleted"] += 1
            reason_counts = subset_stats["deleted_by_reason"]
            reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        else:
            subset_stats["kept"] += 1

    total = {
        "before": sum(int(stats["before"]) for stats in summary.values()),
        "deleted": sum(int(stats["deleted"]) for stats in summary.values()),
        "kept": sum(int(stats["kept"]) for stats in summary.values()),
        "deleted_by_reason": {},
    }
    for stats in summary.values():
        for reason, count in stats["deleted_by_reason"].items():
            total["deleted_by_reason"][reason] = total["deleted_by_reason"].get(reason, 0) + count
    summary["all"] = total
    return summary


def apply_pruning(decisions: list[PruneDecision], dry_run: bool) -> list[dict[str, str | int | None]]:
    deleted_rows: list[dict[str, str | int | None]] = []
    for decision in decisions:
        if decision.action != "delete":
            continue
        error = None
        if not dry_run:
            try:
                decision.record.path.unlink()
            except FileNotFoundError:
                error = "already missing"
            except OSError as exc:
                error = str(exc)
        deleted_rows.append(
            {
                "subset": decision.subset,
                "pages": decision.record.pages,
                "path": str(decision.record.path),
                "reason": decision.reason,
                "error": error,
            }
        )
    return deleted_rows


def print_pruning_summary(prune_summary: dict[str, object], dry_run: bool) -> None:
    verb = "Would delete" if dry_run else "Deleted"
    print(f"\n{verb} PDFs")
    print("\nsubset                before  deleted     kept")
    print("-" * 46)
    for subset in ("pmc_biology", "pmc_medical", "arxiv_physics", "arxiv_engineering", "acl", "all"):
        stats = prune_summary.get(subset)
        if not stats:
            continue
        print(f"{subset:<22}{stats['before']:>6}{stats['deleted']:>9}{stats['kept']:>9}")

    print("\nDeletion reasons")
    total = prune_summary.get("all", {})
    for reason, count in sorted(total.get("deleted_by_reason", {}).items()):
        print(f"  {reason}: {count}")


def bar(count: int, max_count: int, width: int = 32) -> str:
    if max_count <= 0 or count <= 0:
        return ""
    return "#" * max(1, round((count / max_count) * width))


def print_summary(summary: dict[str, dict[str, object]], elapsed: float) -> None:
    print(f"\nCounted PDF pages in {elapsed:.2f}s")
    print("\nset       pdfs  failed    min    p25    med   mean    p75    p90    p95    p99    max")
    print("-" * 86)
    for set_name in ("pmc", "arxiv", "acl", "all"):
        stats = summary.get(set_name)
        if not stats:
            continue
        print(
            f"{set_name:<7}"
            f"{stats['pdfs']:>6}"
            f"{stats['failed']:>8}"
            f"{stats.get('min', '-'):>7}"
            f"{stats.get('p25', '-'):>7}"
            f"{stats.get('median', '-'):>7}"
            f"{stats.get('mean', '-'):>7}"
            f"{stats.get('p75', '-'):>7}"
            f"{stats.get('p90', '-'):>7}"
            f"{stats.get('p95', '-'):>7}"
            f"{stats.get('p99', '-'):>7}"
            f"{stats.get('max', '-'):>7}"
        )

    print("\nDistribution")
    for set_name in ("pmc", "arxiv", "acl", "all"):
        stats = summary.get(set_name)
        if not stats:
            continue
        hist = stats["histogram"]
        max_count = max((bucket["count"] for bucket in hist), default=0)
        print(f"\n{set_name}")
        for bucket in hist:
            count = int(bucket["count"])
            print(f"  {bucket['range']:>7} | {count:>5} {bar(count, max_count)}")


def markdown_report(summary: dict[str, dict[str, object]]) -> str:
    lines = [
        "# PDF Page Count Summary",
        "",
        "| Set | PDFs | Failed | Min | P25 | Median | Mean | P75 | P90 | P95 | P99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_name in ("pmc", "arxiv", "acl", "all"):
        stats = summary.get(set_name)
        if not stats:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    set_name,
                    str(stats["pdfs"]),
                    str(stats["failed"]),
                    str(stats.get("min", "-")),
                    str(stats.get("p25", "-")),
                    str(stats.get("median", "-")),
                    str(stats.get("mean", "-")),
                    str(stats.get("p75", "-")),
                    str(stats.get("p90", "-")),
                    str(stats.get("p95", "-")),
                    str(stats.get("p99", "-")),
                    str(stats.get("max", "-")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Distribution")
    for set_name in ("pmc", "arxiv", "acl", "all"):
        stats = summary.get(set_name)
        if not stats:
            continue
        lines.extend(["", f"### {set_name}", "", "| Pages | PDFs |", "| ---: | ---: |"])
        for bucket in stats["histogram"]:
            lines.append(f"| {bucket['range']} | {bucket['count']} |")
    lines.append("")
    return "\n".join(lines)


def pruning_markdown_report(prune_summary: dict[str, object], dry_run: bool) -> str:
    title = "PDF Pruning Dry Run" if dry_run else "PDF Pruning Report"
    lines = [
        f"# {title}",
        "",
        "| Subset | Before | Deleted | Kept |",
        "| --- | ---: | ---: | ---: |",
    ]
    for subset in ("pmc_biology", "pmc_medical", "arxiv_physics", "arxiv_engineering", "acl", "all"):
        stats = prune_summary.get(subset)
        if not stats:
            continue
        lines.append(f"| {subset} | {stats['before']} | {stats['deleted']} | {stats['kept']} |")

    lines.extend(["", "## Deletion Reasons", "", "| Reason | PDFs |", "| --- | ---: |"])
    total = prune_summary.get("all", {})
    for reason, count in sorted(total.get("deleted_by_reason", {}).items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(records: list[PdfRecord], summary: dict[str, dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pdf_page_counts.csv"
    json_path = output_dir / "pdf_page_summary.json"
    md_path = output_dir / "pdf_page_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("set", "pages", "path", "error"))
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row.set_name, str(row.path))):
            writer.writerow(
                {
                    "set": record.set_name,
                    "pages": "" if record.pages is None else record.pages,
                    "path": str(record.path),
                    "error": record.error or "",
                }
            )

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def write_pruning_outputs(
    decisions: list[PruneDecision],
    deleted_rows: list[dict[str, str | int | None]],
    prune_summary: dict[str, object],
    output_dir: Path,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".dry_run" if dry_run else ""
    plan_path = output_dir / f"pdf_prune_plan{suffix}.csv"
    deleted_path = output_dir / f"pdf_pruned_files{suffix}.csv"
    summary_path = output_dir / f"pdf_prune_summary{suffix}.json"
    md_path = output_dir / f"pdf_prune_summary{suffix}.md"

    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subset", "action", "reason", "pages", "path", "error"))
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "subset": decision.subset,
                    "action": decision.action,
                    "reason": decision.reason,
                    "pages": decision.record.pages if decision.record.pages is not None else "",
                    "path": str(decision.record.path),
                    "error": decision.record.error or "",
                }
            )

    with deleted_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subset", "pages", "path", "reason", "error"))
        writer.writeheader()
        writer.writerows(deleted_rows)

    payload = {
        "dry_run": dry_run,
        "summary": prune_summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(pruning_markdown_report(prune_summary, dry_run), encoding="utf-8")

    print(f"\nWrote {plan_path}")
    print(f"Wrote {deleted_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    bins = sorted({int(value) for value in args.bins.split(",") if value.strip()})
    pdfinfo_bin = shutil.which("pdfinfo")
    use_fallback = not args.no_fallback

    sets = discover_sets(data_root)
    jobs = [(set_name, path) for set_name, paths in sets.items() for path in paths]
    if not jobs:
        print(f"No PDFs found under {data_root}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    records: list[PdfRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(count_one, set_name, path, pdfinfo_bin, use_fallback)
            for set_name, path in jobs
        ]
        for future in as_completed(futures):
            records.append(future.result())

    summary = {
        set_name: summarize([record for record in records if record.set_name == set_name], bins)
        for set_name in sorted(sets)
    }
    summary["all"] = summarize(records, bins)
    elapsed = time.perf_counter() - started
    summary["_meta"] = {
        "elapsed_seconds": round(elapsed, 3),
        "workers": args.workers,
        "pdfinfo": pdfinfo_bin,
        "fallback_enabled": use_fallback,
    }

    print_summary(summary, elapsed)
    write_outputs(records, summary, args.output_dir)
    if args.prune:
        decisions = plan_pruning(records, data_root, args.max_docs_per_subset)
        deleted_rows = apply_pruning(decisions, args.dry_run)
        prune_summary = summarize_pruning(decisions)
        print_pruning_summary(prune_summary, args.dry_run)
        write_pruning_outputs(decisions, deleted_rows, prune_summary, args.output_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
