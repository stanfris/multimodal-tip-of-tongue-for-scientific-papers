"""Explain why selected papers do not have generated queries, without loading a model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common.jsonl import read_jsonl_objects
from document_splits.document_splits import read_split_paper_ids
from preprocessing.preprocessed import clue_domain_dir, read_preprocessed_papers
from queries.query_generation import (
    _components_by_paper,
    _eligible_papers,
    _has_required_modalities,
    _limit_paper_images,
    _query_paper_id,
    load_query_generation_config,
    select_components,
)


def audit(settings: Path, query_set: str) -> dict:
    config = load_query_generation_config(argparse.Namespace(settings=settings, set=query_set))
    papers = [_limit_paper_images(paper, config.max_images) for paper in read_preprocessed_papers(config.dataset)]
    selected = _eligible_papers(papers, config)
    components = _components_by_paper(
        config.dataset,
        clues_dir=config.clues_dir,
        visual_clues_path=config.visual_interpretations,
        textual_clues_path=config.textual_interpretations,
        max_images=config.max_images,
    )
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for _, paper in selected:
        domain = clue_domain_dir(config.clues_dir, str(paper["paper_id"]), paper.get("source_paper_dataset")).name
        by_domain[domain].append(paper)

    report = {
        "split": query_set,
        "split_entries": len(read_split_paper_ids(config.split_index, config.split_name)) if config.split_index else None,
        "preprocessed_papers_loaded": len(papers),
        "papers_selected_for_queries": len(selected),
        "domains": {},
    }
    for domain, domain_papers in sorted(by_domain.items()):
        domain_report = {}
        root_path = config.clues_dir / domain / "queries.jsonl"
        root_rows = read_jsonl_objects(root_path, missing_ok=True)
        for mode in config.modes:
            collection_path = (
                config.output_dir / domain / config.collection_id / mode.replace("-", "_") / "queries.jsonl"
            )
            collection_rows = read_jsonl_objects(collection_path, missing_ok=True)
            collection_ids = {_query_paper_id(row) for row in collection_rows}
            root_ids = {
                _query_paper_id(row) for row in root_rows
                if isinstance(row.get("metadata"), dict) and row["metadata"].get("mode") == mode
            }
            reasons: dict[str, list[str]] = defaultdict(list)
            for paper in domain_papers:
                paper_id = str(paper["paper_id"])
                if paper_id in collection_ids:
                    reasons["query_in_collection"].append(paper_id)
                    continue
                available = components.get(paper_id, [])
                expected_figures = {str(figure["figure_id"]) for figure in paper.get("figures", [])}
                described_figures = {component.record_id for component in available if component.kind == "visual"}
                if not expected_figures:
                    reason = "no_figures"
                elif not any(component.kind == "textual" for component in available):
                    reason = "no_usable_textual_clue"
                elif expected_figures - described_figures:
                    reason = "missing_usable_figure_clue"
                else:
                    chosen = select_components(available, mode=mode)
                    if not chosen:
                        reason = "no_components_for_mode"
                    elif not config.allow_partial_components and not _has_required_modalities(chosen, mode):
                        reason = "missing_required_modality"
                    else:
                        reason = "eligible_but_no_query"
                reasons[reason].append(paper_id)
            domain_report[mode] = {
                "collection_path": str(collection_path),
                "root_path": str(root_path),
                "selected_papers": len(domain_papers),
                "collection_rows": len(collection_rows),
                "root_rows_for_mode": sum(
                    isinstance(row.get("metadata"), dict) and row["metadata"].get("mode") == mode
                    for row in root_rows
                ),
                "collection_missing_from_root": sorted(collection_ids - root_ids),
                "reasons": {reason: sorted(ids) for reason, ids in sorted(reasons.items())},
            }
        report["domains"][domain] = domain_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True, help="Resolved settings.yaml from a query generation run")
    parser.add_argument("--set", choices=("train", "test"), default="train")
    parser.add_argument("--output", type=Path, help="Save the full report, including all paper IDs, as JSON")
    args = parser.parse_args()
    report = audit(args.settings, args.set)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Split entries: {report['split_entries']}; preprocessed loaded: {report['preprocessed_papers_loaded']}; "
          f"selected for queries: {report['papers_selected_for_queries']}")
    for domain, modes in report["domains"].items():
        for mode, details in modes.items():
            counts = {reason: len(ids) for reason, ids in details["reasons"].items()}
            print(f"{domain}/{mode}: {counts}; collection rows={details['collection_rows']}; "
                  f"collection rows absent from root={len(details['collection_missing_from_root'])}")
            for reason, ids in details["reasons"].items():
                if reason != "query_in_collection":
                    print(f"  {reason}: {', '.join(ids[:5])}")
    if args.output:
        print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
