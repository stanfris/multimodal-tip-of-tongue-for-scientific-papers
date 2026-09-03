#!/usr/bin/env python3
"""Check canonical clue coverage and optionally remove stale generated rows."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VISUAL_COMPONENT_BUDGET = 3
DEFAULT_TEXTUAL_COMPONENT_BUDGET = 3
VISUAL_SCHEMA_FIELDS = {
    "figure_type",
    "layout",
    "visual_elements",
    "colors_and_styles",
    "relationships",
    "distinctive_visual_cues",
}

try:
    import yaml
except ImportError:  # pragma: no cover - depends on shell environment
    yaml = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check which canonical papers do not have enough clues for query generation. "
            "Optionally remove visual clue rows for papers that cannot satisfy the visual budget."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs/runs/query_generation_default.yaml",
        help="Query generation config to validate against.",
    )
    parser.add_argument(
        "--visual-clues",
        type=Path,
        default=None,
        help="Canonical visual clues JSONL file. Defaults to <config dataset>/visual_clues.jsonl.",
    )
    parser.add_argument(
        "--delete-wrong-visual-lines",
        action="store_true",
        help="Rewrite visual_clues.jsonl without rows for papers below the visual budget.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak copy before rewriting visual_clues.jsonl.",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Generated queries.jsonl file to inspect for selected visual components based on empty JSON sets. "
            "May be passed more than once. The canonical queries.jsonl is always included when present."
        ),
    )
    parser.add_argument(
        "--delete-empty-visual-query-lines",
        action="store_true",
        help="Rewrite checked queries.jsonl files without rows that used empty visual JSON inputs.",
    )
    parser.add_argument(
        "--delete-underfilled-query-lines",
        action="store_true",
        help="Rewrite checked queries.jsonl files without rows whose paper no longer satisfies the mode budgets.",
    )
    return parser


def visual_component_counts(paper_ids, components_by_paper):
    return {
        pid: sum(1 for component in components_by_paper.get(pid, []) if component.kind == "visual")
        for pid in paper_ids
    }


def textual_component_counts(paper_ids, components_by_paper):
    return {
        pid: sum(1 for component in components_by_paper.get(pid, []) if component.kind == "textual")
        for pid in paper_ids
    }


def canonical_dataset_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name == "papers.jsonl":
        return candidate
    return candidate / "papers.jsonl"


def canonical_textual_clues_path(path: str | Path) -> Path:
    return canonical_dataset_path(path).parent / "textual_clues.jsonl"


def canonical_visual_clues_path(path: str | Path) -> Path:
    return canonical_dataset_path(path).parent / "visual_clues.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_canonical_papers(path: str | Path) -> list[dict[str, Any]]:
    data_path = canonical_dataset_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"No papers.jsonl found at {data_path}")
    return read_jsonl(data_path)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _resolve_optional_path(path: str | Path | None, base_dir: Path) -> Path | None:
    return None if path is None else _resolve_path(path, base_dir)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    pending_list_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            section = {}
            parsed[line[:-1]] = section
            pending_list_key = None
            continue
        if section is None:
            continue
        if line.startswith("- ") and pending_list_key is not None:
            section[pending_list_key].append(_parse_scalar(line[2:]))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            section[key] = _parse_scalar(value)
            pending_list_key = None
        else:
            section[key] = []
            pending_list_key = key
    return parsed


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        raw = yaml.safe_load(text)
    else:
        raw = _parse_simple_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def config_from_yaml(path: Path) -> SimpleNamespace:
    config_path = path.expanduser().resolve()
    raw = load_yaml_mapping(config_path)
    config_dir = config_path.parent
    run = _section(raw, "run")
    inputs = _section(raw, "inputs")
    selection = _section(raw, "selection")
    output = _section(raw, "output")
    return SimpleNamespace(
        dataset=_resolve_path(inputs["dataset"], config_dir),
        visual_interpretations=_resolve_optional_path(inputs.get("visual_interpretations"), config_dir),
        textual_interpretations=_resolve_optional_path(inputs.get("textual_interpretations"), config_dir),
        output_dir=_resolve_path(output["dir"], config_dir),
        visual_component_budget=int(
            selection.get(
                "visual_component_budget",
                selection.get("component_budget", DEFAULT_VISUAL_COMPONENT_BUDGET),
            )
        ),
        textual_component_budget=int(
            selection.get(
                "textual_component_budget",
                selection.get("max_text_components", selection.get("component_budget", DEFAULT_TEXTUAL_COMPONENT_BUDGET)),
            )
        ),
        allow_partial_components=bool(selection.get("allow_partial_components", False)),
        collection_id=run.get("name", "query_generation_default"),
    )


def _repair_json_dict(text: str) -> dict[str, Any] | None:
    closings = ["", '"', '"]}', ']}', '}', '"}']
    stripped = re.sub(r",\s*$", "", text.strip())
    for closing in closings:
        try:
            value = json.loads(stripped + closing)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    while "," in text:
        text = text[: text.rfind(",")]
        stripped = text.strip()
        for closing in closings:
            try:
                value = json.loads(stripped + closing)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    return None


def _trim_to_json_start(text: str) -> str:
    text = text.strip()
    first_json = min((index for index in (text.find("["), text.find("{")) if index != -1), default=-1)
    return text[first_json:] if first_json > 0 else text


def _decode_jsonish_value(text: str) -> Any | None:
    text = _trim_to_json_start(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("{"):
        return _repair_json_dict(text)
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
        return value
    except json.JSONDecodeError:
        return None


def _decode_jsonish_components(text: str) -> list[Any] | None:
    text = _trim_to_json_start(text)
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]

    decoder = json.JSONDecoder()
    index = 0
    values: list[Any] = []
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] in ","):
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            remaining = text[index:].strip()
            dict_start = remaining.find("{")
            if dict_start != -1:
                repaired = _repair_json_dict(remaining[dict_start:])
                if repaired:
                    values.append(repaired)
            break
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
        index = end
    return values or None


def _textual_component(value: Any) -> str | None:
    if isinstance(value, dict):
        cue = value.get("cue")
        return str(cue) if cue is not None else None
    return str(value)


def _visual_components(value: Any) -> list[str]:
    if isinstance(value, dict):
        components: list[str] = []
        for field_value in value.values():
            if isinstance(field_value, list):
                components.extend(str(item) for item in field_value)
            elif field_value is not None:
                components.append(str(field_value))
        return components
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def split_component_text(text: str, kind: str) -> list[str]:
    decoded = _decode_jsonish_components(text)
    if decoded is None:
        return [text]
    if kind == "textual":
        components = [_textual_component(value) for value in decoded]
    else:
        components = []
        for value in decoded:
            components.extend(_visual_components(value))
    cleaned = [component.strip() for component in components if component and component.strip()]
    return cleaned


def is_empty_visual_json_value(value: Any) -> bool:
    if isinstance(value, list):
        return not value or all(is_empty_visual_json_value(item) for item in value)
    if isinstance(value, dict):
        relevant_values = [value[field] for field in VISUAL_SCHEMA_FIELDS if field in value]
        return bool(relevant_values) and all(
            field_value in (None, "", []) or is_empty_visual_json_value(field_value)
            for field_value in relevant_values
        )
    return False


def is_empty_visual_json_text(text: str) -> bool:
    value = _decode_jsonish_value(text)
    return value is not None and is_empty_visual_json_value(value)


def canonical_query_path(config: SimpleNamespace) -> Path:
    return canonical_dataset_path(config.dataset).parent / "queries.jsonl"


def configured_collection_query_paths(config: SimpleNamespace) -> list[Path]:
    paths: list[Path] = []
    collection_root = config.output_dir / str(config.collection_id)
    for mode in ("visual_only", "visual_and_text"):
        query_path = collection_root / mode / "queries.jsonl"
        if query_path.exists():
            paths.append(query_path)
    return paths


def checked_query_paths(config: SimpleNamespace, query_paths: list[Path] | None) -> list[Path]:
    paths: list[Path] = []
    canonical_queries = canonical_query_path(config)
    if canonical_queries.exists():
        paths.append(canonical_queries)
    paths.extend(query_paths if query_paths else configured_collection_query_paths(config))
    return _unique_paths(paths)


def _bad_empty_visual_components(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    selected_components = metadata.get("selected_components")
    if not isinstance(selected_components, list):
        return []
    bad_components: list[dict[str, Any]] = []
    for component_index, component in enumerate(selected_components):
        if not isinstance(component, dict) or component.get("kind") != "visual":
            continue
        component_text = str(component.get("text", ""))
        if is_empty_visual_json_text(component_text):
            bad_components.append(
                {
                    "component_index": component_index,
                    "record_id": str(component.get("record_id", "")),
                    "text_first_line": component_text.splitlines()[0] if component_text else "",
                }
            )
    return bad_components


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def find_queries_with_empty_visual_inputs(query_paths: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for query_path in _unique_paths(query_paths):
        if not query_path.exists():
            continue
        for line_number, line in enumerate(query_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            bad_components = _bad_empty_visual_components(row)
            if bad_components:
                findings.append(
                    {
                        "path": query_path,
                        "line_number": line_number,
                        "query_id": row.get("query_id"),
                        "paper_id": metadata.get("paper_id"),
                        "mode": metadata.get("mode"),
                        "bad_components": bad_components,
                    }
                )
    return findings


def delete_empty_visual_query_lines(query_paths: list[Path], *, backup: bool) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for query_path in _unique_paths(query_paths):
        if not query_path.exists():
            reports.append(
                {
                    "path": query_path,
                    "total_lines": 0,
                    "kept_lines": 0,
                    "deleted_lines": 0,
                    "invalid_json_lines_kept": 0,
                    "missing": True,
                    "deleted": [],
                }
            )
            continue
        lines = query_path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept: list[str] = []
        deleted: list[tuple[int, str, str]] = []
        invalid_json = 0
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                kept.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                kept.append(line)
                continue
            if not isinstance(row, dict) or not _bad_empty_visual_components(row):
                kept.append(line)
                continue
            metadata = row.get("metadata", {})
            paper_id = metadata.get("paper_id", "") if isinstance(metadata, dict) else ""
            deleted.append((line_number, str(row.get("query_id", "")), str(paper_id)))

        if deleted:
            if backup:
                backup_path = query_path.with_suffix(query_path.suffix + ".bak")
                shutil.copy2(query_path, backup_path)
            tmp_path = query_path.with_suffix(query_path.suffix + ".tmp")
            tmp_path.write_text("".join(kept), encoding="utf-8")
            tmp_path.replace(query_path)

        reports.append(
            {
                "path": query_path,
                "total_lines": len(lines),
                "kept_lines": len(kept),
                "deleted_lines": len(deleted),
                "invalid_json_lines_kept": invalid_json,
                "missing": False,
                "deleted": deleted,
            }
        )
    return reports


def query_row_paper_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("paper_id"):
        return str(metadata["paper_id"])
    relevant_ids = row.get("relevant_ids")
    if isinstance(relevant_ids, list) and relevant_ids:
        return str(relevant_ids[0])
    return ""


def query_row_mode(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("mode"):
        return str(metadata["mode"])
    query_id = str(row.get("query_id") or "")
    if query_id.startswith("visual_only_"):
        return "visual-only"
    if query_id.startswith("visual_and_text_"):
        return "visual-and-text"
    return ""


def query_row_underfilled_reason(
    row: dict[str, Any],
    *,
    visual_missing_paper_ids: set[str],
    both_missing_paper_ids: set[str],
) -> str | None:
    paper_id = query_row_paper_id(row)
    mode = query_row_mode(row)
    if mode == "visual-only" and paper_id in visual_missing_paper_ids:
        return "paper has fewer visual components than required for visual-only"
    if mode == "visual-and-text" and paper_id in both_missing_paper_ids:
        return "paper has fewer components than required for visual-and-text"
    return None


def find_underfilled_query_rows(
    query_paths: list[Path],
    *,
    visual_missing_paper_ids: set[str],
    both_missing_paper_ids: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for query_path in _unique_paths(query_paths):
        if not query_path.exists():
            continue
        for line_number, line in enumerate(query_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            reason = query_row_underfilled_reason(
                row,
                visual_missing_paper_ids=visual_missing_paper_ids,
                both_missing_paper_ids=both_missing_paper_ids,
            )
            if reason is None:
                continue
            findings.append(
                {
                    "path": query_path,
                    "line_number": line_number,
                    "query_id": row.get("query_id"),
                    "paper_id": query_row_paper_id(row),
                    "mode": query_row_mode(row),
                    "reason": reason,
                }
            )
    return findings


def delete_underfilled_query_lines(
    query_paths: list[Path],
    *,
    visual_missing_paper_ids: set[str],
    both_missing_paper_ids: set[str],
    backup: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for query_path in _unique_paths(query_paths):
        if not query_path.exists():
            reports.append(
                {
                    "path": query_path,
                    "total_lines": 0,
                    "kept_lines": 0,
                    "deleted_lines": 0,
                    "invalid_json_lines_kept": 0,
                    "missing": True,
                    "deleted": [],
                }
            )
            continue
        lines = query_path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept: list[str] = []
        deleted: list[tuple[int, str, str, str, str]] = []
        invalid_json = 0
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                kept.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                kept.append(line)
                continue
            if not isinstance(row, dict):
                kept.append(line)
                continue
            reason = query_row_underfilled_reason(
                row,
                visual_missing_paper_ids=visual_missing_paper_ids,
                both_missing_paper_ids=both_missing_paper_ids,
            )
            if reason is None:
                kept.append(line)
                continue
            paper_id = query_row_paper_id(row)
            mode = query_row_mode(row)
            deleted.append((line_number, str(row.get("query_id", "")), str(mode), str(paper_id), reason))

        if deleted:
            if backup:
                backup_path = query_path.with_suffix(query_path.suffix + ".bak")
                shutil.copy2(query_path, backup_path)
            tmp_path = query_path.with_suffix(query_path.suffix + ".tmp")
            tmp_path.write_text("".join(kept), encoding="utf-8")
            tmp_path.replace(query_path)

        reports.append(
            {
                "path": query_path,
                "total_lines": len(lines),
                "kept_lines": len(kept),
                "deleted_lines": len(deleted),
                "invalid_json_lines_kept": invalid_json,
                "missing": False,
                "deleted": deleted,
            }
        )
    return reports


def canonical_components_by_paper(
    dataset: Path,
    *,
    visual_clues_path: Path | None,
    textual_clues_path: Path | None,
) -> dict[str, list[SimpleNamespace]]:
    papers = read_canonical_papers(dataset)
    paper_ids = {str(paper["paper_id"]) for paper in papers}
    figure_to_paper = {
        str(figure["figure_id"]): str(paper["paper_id"])
        for paper in papers
        for figure in paper.get("figures", [])
    }
    visual_path = (
        visual_clues_path
        if visual_clues_path is not None and visual_clues_path.is_file()
        else canonical_visual_clues_path(dataset)
    )
    text_path = (
        textual_clues_path
        if textual_clues_path is not None and textual_clues_path.is_file()
        else canonical_textual_clues_path(dataset)
    )

    components: dict[str, list[SimpleNamespace]] = {}
    for clue in read_jsonl(text_path):
        if clue.get("kind") != "textual":
            continue
        paper_id = str(clue.get("paper_id", ""))
        if paper_id not in paper_ids:
            continue
        for component_text in split_component_text(str(clue.get("output", "")).strip(), "textual"):
            components.setdefault(paper_id, []).append(SimpleNamespace(kind="textual", text=component_text))

    for clue in read_jsonl(visual_path):
        if clue.get("kind") != "visual":
            continue
        paper_id = str(clue.get("paper_id", ""))
        figure_id = str(clue.get("figure_id", ""))
        if paper_id not in paper_ids or figure_id not in figure_to_paper:
            continue
        for component_text in split_component_text(str(clue.get("output", "")).strip(), "visual"):
            components.setdefault(paper_id, []).append(SimpleNamespace(kind="visual", text=component_text))
    return components


def invalid_visual_clue_reasons(
    clue: dict,
    *,
    insufficient_paper_ids: set[str],
) -> list[str]:
    if clue.get("kind") != "visual":
        return []
    reasons: list[str] = []
    if is_empty_visual_json_text(str(clue.get("output", "")).strip()):
        reasons.append("visual clue output is empty JSON")
    paper_id = str(clue.get("paper_id", ""))
    if paper_id in insufficient_paper_ids:
        reasons.append("paper has fewer visual components than required")
    return reasons


def delete_wrong_visual_lines(
    visual_clues_path: Path,
    *,
    insufficient_paper_ids: set[str],
    backup: bool,
) -> dict:
    if not visual_clues_path.exists():
        raise FileNotFoundError(f"Visual clues file does not exist: {visual_clues_path}")

    lines = visual_clues_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    deleted: list[tuple[int, str, list[str]]] = []
    invalid_json = 0

    for line_number, line in enumerate(lines, start=1):
        try:
            clue = json.loads(line)
        except json.JSONDecodeError:
            invalid_json += 1
            kept.append(line)
            continue
        reasons = invalid_visual_clue_reasons(
            clue,
            insufficient_paper_ids=insufficient_paper_ids,
        )
        if reasons:
            identifier = str(clue.get("figure_id") or clue.get("paper_id") or "")
            deleted.append((line_number, identifier, reasons))
            continue
        kept.append(line)

    if deleted:
        if backup:
            backup_path = visual_clues_path.with_suffix(visual_clues_path.suffix + ".bak")
            shutil.copy2(visual_clues_path, backup_path)
        tmp_path = visual_clues_path.with_suffix(visual_clues_path.suffix + ".tmp")
        tmp_path.write_text("".join(kept), encoding="utf-8")
        tmp_path.replace(visual_clues_path)

    return {
        "total_lines": len(lines),
        "kept_lines": len(kept),
        "deleted_lines": len(deleted),
        "invalid_json_lines_kept": invalid_json,
        "deleted": deleted,
    }


def find_invalid_visual_clue_lines(
    visual_clues_path: Path,
    *,
    insufficient_paper_ids: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not visual_clues_path.exists():
        return findings
    for line_number, line in enumerate(visual_clues_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            clue = json.loads(line)
        except json.JSONDecodeError:
            continue
        reasons = invalid_visual_clue_reasons(
            clue,
            insufficient_paper_ids=insufficient_paper_ids,
        )
        if not reasons:
            continue
        findings.append(
            {
                "path": visual_clues_path,
                "line_number": line_number,
                "paper_id": clue.get("paper_id"),
                "figure_id": clue.get("figure_id"),
                "reasons": reasons,
            }
        )
    return findings


def main():
    args = build_parser().parse_args()
    config_path = args.config
    
    print(f"Loading config from {config_path.name}...")
    config = config_from_yaml(config_path)
    
    print(f"Reading canonical dataset from {config.dataset}...")
    papers = read_canonical_papers(config.dataset)
    paper_ids = [str(p["paper_id"]) for p in papers]
    
    components_by_paper = canonical_components_by_paper(
        dataset=config.dataset,
        visual_clues_path=config.visual_interpretations,
        textual_clues_path=config.textual_interpretations
    )
    
    print("\nConfiguration Requirements:")
    print(f" - Visual Component Budget: {config.visual_component_budget}")
    print(f" - Textual Component Budget: {config.textual_component_budget}")
    print(f" - Allow Partial Components: {config.allow_partial_components}")

    visual_counts = visual_component_counts(paper_ids, components_by_paper)
    textual_counts = textual_component_counts(paper_ids, components_by_paper)
    
    print("\n--- Papers missing sufficient clues for 'visual-only' ---")
    visual_missing_paper_ids: set[str] = set()
    for pid in paper_ids:
        visual_count = visual_counts[pid]
        if not config.allow_partial_components and visual_count < config.visual_component_budget:
            print(f" [!] {pid}: only has {visual_count} visual components")
            visual_missing_paper_ids.add(pid)
            
    print(f"\n--- Papers missing sufficient clues for 'visual-and-text' ---")
    both_missing = 0
    both_missing_paper_ids: set[str] = set()
    for pid in paper_ids:
        visual_count = visual_counts[pid]
        textual_count = textual_counts[pid]
        
        missing_v = visual_count < config.visual_component_budget
        missing_t = textual_count < config.textual_component_budget
        
        if not config.allow_partial_components and (missing_v or missing_t):
            print(f" [!] {pid}: visual ({visual_count}/{config.visual_component_budget}), textual ({textual_count}/{config.textual_component_budget})")
            both_missing += 1
            both_missing_paper_ids.add(pid)
            
    print("\nSummary:")
    print(f"Total Papers: {len(paper_ids)}")
    print(f"Cannot generate 'visual-only': {len(visual_missing_paper_ids)} papers")
    print(f"Cannot generate 'visual-and-text': {both_missing} papers")

    visual_path = args.visual_clues or canonical_visual_clues_path(config.dataset)
    invalid_visual_clue_findings = find_invalid_visual_clue_lines(
        visual_path,
        insufficient_paper_ids=visual_missing_paper_ids,
    )
    print("\n--- Invalid visual clue rows ---")
    print(f"Visual clues file: {visual_path}")
    print(f"Affected visual clue rows: {len(invalid_visual_clue_findings)}")
    for finding in invalid_visual_clue_findings[:50]:
        print(
            f" [!] {finding['path']}:{finding['line_number']} "
            f"{finding['figure_id']} (paper {finding['paper_id']}): {'; '.join(finding['reasons'])}"
        )
    if len(invalid_visual_clue_findings) > 50:
        print(f" ... {len(invalid_visual_clue_findings) - 50} more affected visual clue rows")

    query_paths = checked_query_paths(config, args.query_path)
    empty_visual_query_findings = find_queries_with_empty_visual_inputs(query_paths)
    print("\n--- Generated queries using empty visual JSON inputs ---")
    print(f"Query files checked: {len(query_paths)}")
    print(f"Affected query rows: {len(empty_visual_query_findings)}")
    for finding in empty_visual_query_findings[:50]:
        bad = "; ".join(
            f"component #{component['component_index']} {component['record_id']} {component['text_first_line']}"
            for component in finding["bad_components"]
        )
        print(
            f" [!] {finding['path']}:{finding['line_number']} "
            f"{finding['query_id']} ({finding['mode']}, paper {finding['paper_id']}): {bad}"
        )
    if len(empty_visual_query_findings) > 50:
        print(f" ... {len(empty_visual_query_findings) - 50} more affected query rows")

    underfilled_query_findings = find_underfilled_query_rows(
        query_paths,
        visual_missing_paper_ids=visual_missing_paper_ids,
        both_missing_paper_ids=both_missing_paper_ids,
    )
    print("\n--- Generated queries for underfilled papers ---")
    print(f"Affected query rows: {len(underfilled_query_findings)}")
    for finding in underfilled_query_findings[:50]:
        print(
            f" [!] {finding['path']}:{finding['line_number']} "
            f"{finding['query_id']} ({finding['mode']}, paper {finding['paper_id']}): {finding['reason']}"
        )
    if len(underfilled_query_findings) > 50:
        print(f" ... {len(underfilled_query_findings) - 50} more affected query rows")

    if args.delete_empty_visual_query_lines:
        reports = delete_empty_visual_query_lines(query_paths, backup=not args.no_backup)
        print("\n--- Deleted generated query lines using empty visual JSON inputs ---")
        for report in reports:
            if report["missing"]:
                print(f" [!] Missing query file: {report['path']}")
                continue
            print(
                f"Query file: {report['path']} "
                f"(deleted {report['deleted_lines']}, kept {report['kept_lines']}, total {report['total_lines']})"
            )
            if report["invalid_json_lines_kept"]:
                print(f"Invalid JSON lines kept: {report['invalid_json_lines_kept']}")
            for line_number, query_id, paper_id in report["deleted"][:20]:
                print(f" [x] line {line_number}: {query_id} (paper {paper_id})")
            if report["deleted_lines"] > 20:
                print(f" ... {report['deleted_lines'] - 20} more deleted lines")

    if args.delete_underfilled_query_lines:
        reports = delete_underfilled_query_lines(
            query_paths,
            visual_missing_paper_ids=visual_missing_paper_ids,
            both_missing_paper_ids=both_missing_paper_ids,
            backup=not args.no_backup,
        )
        print("\n--- Deleted generated query lines for underfilled papers ---")
        for report in reports:
            if report["missing"]:
                print(f" [!] Missing query file: {report['path']}")
                continue
            print(
                f"Query file: {report['path']} "
                f"(deleted {report['deleted_lines']}, kept {report['kept_lines']}, total {report['total_lines']})"
            )
            if report["invalid_json_lines_kept"]:
                print(f"Invalid JSON lines kept: {report['invalid_json_lines_kept']}")
            for line_number, query_id, mode, paper_id, reason in report["deleted"][:20]:
                print(f" [x] line {line_number}: {query_id} ({mode}, paper {paper_id}): {reason}")
            if report["deleted_lines"] > 20:
                print(f" ... {report['deleted_lines'] - 20} more deleted lines")

    if args.delete_wrong_visual_lines:
        report = delete_wrong_visual_lines(
            visual_path,
            insufficient_paper_ids=visual_missing_paper_ids,
            backup=not args.no_backup,
        )
        print("\n--- Deleted underfilled visual clue lines ---")
        print(f"Visual clues file: {visual_path}")
        print(f"Total lines: {report['total_lines']}")
        print(f"Deleted lines: {report['deleted_lines']}")
        print(f"Kept lines: {report['kept_lines']}")
        if report["invalid_json_lines_kept"]:
            print(f"Invalid JSON lines kept: {report['invalid_json_lines_kept']}")
        for line_number, identifier, reasons in report["deleted"][:20]:
            print(f" [x] line {line_number}: {identifier} ({'; '.join(reasons)})")
        if report["deleted_lines"] > 20:
            print(f" ... {report['deleted_lines'] - 20} more deleted lines")

if __name__ == "__main__":
    main()
