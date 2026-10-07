"""Explain why selected papers do not have generated queries, without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from clues.component_parsing import split_component_text
from common.jsonl import read_jsonl_objects
from document_splits.document_splits import read_split_paper_ids
from preprocessing.preprocessed import (
    clue_domain_dir, read_clue_rows, read_preprocessed_papers, safe_path_name,
    textual_clue_path, visual_clue_path,
)
from queries.query_generation import (
    MemoryComponent,
    _components_by_paper,
    _eligible_papers,
    _has_required_modalities,
    _limit_paper_images,
    _query_paper_id,
    load_query_generation_config,
    select_components,
)


def _split_paper_dir(root: Path, entry: str) -> tuple[Path | None, set[str]]:
    pdf_path = Path(entry)
    subset = safe_path_name(pdf_path.parts[0] if len(pdf_path.parts) > 1 else "Unknown")
    paper_id = safe_path_name(pdf_path.with_suffix("").as_posix())
    collision_id = f"{paper_id}.{hashlib.sha256(pdf_path.as_posix().encode()).hexdigest()[:12]}"
    for candidate_id in (paper_id, collision_id):
        candidates = (
            root / "papers" / subset / candidate_id,
            root / subset / "papers" / candidate_id,
            root / subset / "papers" / subset / candidate_id,
            root / subset / candidate_id,
            root / candidate_id,
        )
        for candidate in candidates:
            if (candidate / "markdown.md").is_file():
                return candidate, {paper_id, collision_id}
    return None, {paper_id, collision_id}


def _paper_reason(paper: dict, config, mode: str) -> tuple[str, list[str]]:
    paper_id = str(paper["paper_id"])
    source = paper.get("source_paper_dataset")
    figures = paper.get("figures", [])[:config.max_images]
    if not figures:
        return "no_figures", []
    components: list[MemoryComponent] = []
    if config.textual_interpretations is not None and config.textual_interpretations.is_file():
        textual_rows = read_clue_rows(config.textual_interpretations)
    else:
        textual_rows = read_clue_rows(textual_clue_path(config.clues_dir, paper_id, source))
    for row in textual_rows:
        if row.get("kind") == "textual" and str(row.get("paper_id", "")) == paper_id:
            components.extend(
                MemoryComponent(paper_id, "textual", value)
                for value in split_component_text(str(row.get("output", "")).strip(), "textual")
            )
    if not any(component.kind == "textual" for component in components):
        return "no_usable_textual_clue", []
    missing_figures = []
    for figure in figures:
        figure_id = str(figure["figure_id"])
        if config.visual_interpretations is not None and config.visual_interpretations.is_file():
            visual_rows = read_clue_rows(config.visual_interpretations)
        else:
            visual_rows = read_clue_rows(visual_clue_path(config.clues_dir, paper_id, figure_id, source))
        values = [
            value
            for row in visual_rows
            if row.get("kind") == "visual"
            and str(row.get("paper_id", "")) == paper_id
            and str(row.get("figure_id", "")) == figure_id
            for value in split_component_text(str(row.get("output", "")).strip(), "visual")
        ]
        if not values:
            missing_figures.append(figure_id)
        components.extend(MemoryComponent(figure_id, "visual", value) for value in values)
    if missing_figures and not (config.allow_missing_figure_clues and any(component.kind == "visual" for component in components)):
        return "missing_usable_figure_clue", missing_figures
    chosen = select_components(components, mode=mode)
    if not chosen:
        return "no_components_for_mode", []
    if not config.allow_partial_components and not _has_required_modalities(chosen, mode):
        return "missing_required_modality", []
    return "eligible_but_no_query", []


def stream_audit(settings: Path, query_set: str, output: Path | None) -> None:
    """Write each missing split entry as soon as it has been checked."""
    config = load_query_generation_config(argparse.Namespace(settings=settings, set=query_set))
    if config.split_index is None:
        raise ValueError("Streaming audit requires a split index in the selected settings file")
    entries = read_split_paper_ids(config.split_index, config.split_name)
    print(f"Checking {len(entries)} {query_set} split entries; loading existing query IDs...", flush=True)
    existing: dict[str, set[str]] = {}
    for mode in config.modes:
        existing[mode] = {
            paper_id
            for path in config.output_dir.glob(
                f"*/{config.collection_id}/{mode.replace('-', '_')}/queries.jsonl"
            )
            for row in read_jsonl_objects(path)
            if (paper_id := _query_paper_id(row)) is not None
        }
        print(f"{mode}: {len(existing[mode])} papers already have queries", flush=True)
    handle = None
    try:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = output.open("w", encoding="utf-8")
            print(f"Writing each missing paper immediately to {output}", flush=True)
        counts: dict[str, int] = defaultdict(int)
        for index, entry in enumerate(entries, 1):
            if not entry.lower().endswith(".pdf"):
                raise ValueError(f"Streaming audit expects PDF paths in the split index; found {entry!r}")
            paper_dir, candidate_ids = _split_paper_dir(config.dataset, entry)
            missing_modes = [mode for mode in config.modes if not candidate_ids & existing[mode]]
            if not missing_modes:
                continue
            paper = None
            if paper_dir is not None:
                loaded = read_preprocessed_papers(paper_dir)
                paper = _limit_paper_images(loaded[0], config.max_images) if loaded else None
            for mode in missing_modes:
                if paper is not None and str(paper["paper_id"]) in existing[mode]:
                    continue
                reason, missing_figures = (
                    _paper_reason(paper, config, mode) if paper is not None
                    else ("missing_preprocessed_markdown", [])
                )
                counts[reason] += 1
                result = {
                    "split_position": index, "split_entry": entry, "mode": mode,
                    "paper_id": str(paper["paper_id"]) if paper else sorted(candidate_ids)[0],
                    "reason": reason, "missing_figure_ids": missing_figures,
                }
                line = json.dumps(result, ensure_ascii=False)
                print(line, flush=True)
                if handle is not None:
                    handle.write(line + "\n")
                    handle.flush()
        print(f"Finished: {dict(sorted(counts.items()))}", flush=True)
    finally:
        if handle is not None:
            handle.close()


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
                elif expected_figures - described_figures and not (
                    config.allow_missing_figure_clues and expected_figures & described_figures
                ):
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
    parser.add_argument("--stream", action="store_true", help="Print and save each missing split entry immediately as JSONL")
    args = parser.parse_args()
    if args.stream:
        stream_audit(args.settings, args.set, args.output)
        return
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
