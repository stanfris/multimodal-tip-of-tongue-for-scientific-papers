"""Local Streamlit review UI for generated tip-of-the-tongue queries."""

from __future__ import annotations

import html
import json
import random
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dataset_generation.query_generation import _split_component_text


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COLLECTION = PROJECT_ROOT / "data/canonical"
DEFAULT_DATASET = PROJECT_ROOT / "data/canonical"
ANNOTATIONS_PATH = PROJECT_ROOT / "data/reviews/query_review_annotations.jsonl"
QUERY_COLLECTION_ROOT = PROJECT_ROOT / "data/query_collections"
PRIORITY_COLLECTION_IDS = ("query_generation_train", "query_generation_test")
QUERY_MODES = ("visual_only", "visual_and_text")
GENERATE_QUERY_COMMANDS = {
    "query_generation_train": "SPLIT=train COLLECTION_ID=query_generation_train scripts/08_generate_queries.sh",
    "query_generation_test": "SPLIT=test COLLECTION_ID=query_generation_test scripts/08_generate_queries.sh",
}

RATINGS = ["Unreviewed", "Good", "Questionable", "Bad"]
ISSUES = [
    "leakage",
    "unnatural query",
    "incorrect visual clue",
    "incorrect textual clue",
    "too easy",
    "too vague",
    "other",
]


def main() -> None:
    st.set_page_config(page_title="TOT Query Review", layout="wide")
    st.markdown(
        """
        <style>
        .review-header {border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.8rem 1rem; background: #f8fafc; color: #000; margin-bottom: 1rem;}
        .review-title-row {display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;}
        .review-header .paper-id {font-size: 1.35rem; font-weight: 700; color: #000; line-height: 1.2;}
        .review-header .query-id {font-size: 1.15rem; font-weight: 700; color: #334155; line-height: 1.2;}
        .query-box {border: 1px solid #cbd5e1; border-radius: 6px; padding: 1rem; background: #f8fafc; color: #000; font-size: 1.1rem;}
        .scroll-box {border: 1px solid #cbd5e1; border-radius: 6px; padding: 0.75rem; background: #fff; color: #000; max-height: 430px; overflow-y: auto; margin-bottom: 0.75rem;}
        .paper-scroll {height: 300px; max-height: 300px;}
        .clues-scroll {height: 315px; max-height: 315px;}
        .figure-title {font-weight: 600; margin: 0.5rem 0 0.15rem 0; color: #000;}
        .pdf-page-scroll {height: 420px; overflow-y: auto; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0.75rem; background: #f8fafc;}
        .paper-text {white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.85rem; line-height: 1.35;}
        .clue {border-left: 4px solid #cbd5e1; padding: 0.45rem 0.65rem; margin: 0.35rem 0; background: #f8fafc; color: #000;}
        .clue-selected {border-left-color: #2563eb; background: #eff6ff; color: #000;}
        .clue-missing {border-left-color: #dc2626; background: #fef2f2; color: #000;}
        .small-muted {color: #334155; font-size: 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Tip-of-the-Tongue Query Review")

    collection = collection_sidebar()
    examples = load_examples(collection)
    annotations = load_annotations(ANNOTATIONS_PATH)
    if not examples:
        render_no_queries_message(collection)
        return

    dataset_dir = dataset_dir_for_collection(collection)
    records = load_dataset_records(dataset_dir)
    visual_rows = load_interpretations(dataset_dir / "visual_clues.jsonl")
    textual_rows = load_interpretations(dataset_dir / "textual_clues.jsonl")

    filtered = filter_examples(examples, annotations)
    if not filtered:
        st.warning("No examples match the current filters.")
        return

    sync_index(filtered)
    controls(filtered)
    example = filtered[st.session_state.example_index]
    key = annotation_key(collection, example)
    annotation = annotations.get(key, {})

    record = records.get(source_record_id(example), {})
    same_paper_records = records_by_paper(records).get(resolved_paper_id(example, record), [])
    visual_clues = clues_for_paper(visual_rows, same_paper_records, "visual")
    textual_clues = clues_for_paper(textual_rows, same_paper_records, "textual")

    st.caption(f"{st.session_state.example_index + 1} of {len(filtered)} matching examples")
    render_example(
        collection,
        example,
        record,
        same_paper_records,
        visual_rows,
        visual_clues,
        textual_clues,
        dataset_dir,
    )
    render_annotation(collection, example, annotation, annotations)


def collection_sidebar() -> Path:
    collections = discover_query_collections()
    default_collection = QUERY_COLLECTION_ROOT / "query_generation_train"
    default_index = collections.index(default_collection) if default_collection in collections else 0
    with st.sidebar:
        st.header("Data")
        selected_collection = st.selectbox(
            "Split / collection",
            collections,
            index=default_index,
            format_func=collection_label,
        )
        modes = mode_options_for_collection(selected_collection)
        mode = st.selectbox("Mode", modes, format_func=mode_label)
        selected = query_path_for_mode(selected_collection, mode)
        st.caption(f"Reading `{relative_path(selected)}`")
        st.caption("Annotations are saved separately in data/reviews.")
    return selected


def discover_query_collections() -> list[Path]:
    roots = [QUERY_COLLECTION_ROOT, PROJECT_ROOT / "data/processed/test_collections", DEFAULT_COLLECTION]
    priority = [QUERY_COLLECTION_ROOT / collection_id for collection_id in PRIORITY_COLLECTION_IDS]
    collections = [*priority]
    for root in roots:
        if not root.exists():
            continue
        if is_query_collection_root(root):
            collections.append(root)
        for path in sorted(root.iterdir()):
            if path.is_dir() and is_query_collection_root(path):
                collections.append(path)
    return dedupe_paths(collections)


def is_query_collection_root(path: Path) -> bool:
    return (path / "queries.jsonl").exists() or any((path / mode / "queries.jsonl").exists() for mode in QUERY_MODES) or (
        path / "metadata.json"
    ).exists()


def mode_options_for_collection(collection: Path) -> list[str]:
    modes = []
    if collection.name in PRIORITY_COLLECTION_IDS:
        modes.extend(QUERY_MODES)
    if collection.exists():
        modes.extend(
            path.name
            for path in sorted(collection.iterdir())
            if path.is_dir() and (path / "queries.jsonl").exists()
        )
    if (collection / "queries.jsonl").exists():
        modes.append("default")
    return dedupe_strings(modes) or list(QUERY_MODES)


def query_path_for_mode(collection: Path, mode: str) -> Path:
    return collection if mode == "default" else collection / mode


def collection_label(path: Path) -> str:
    prefix = ""
    if path.name == "query_generation_train":
        prefix = "train - "
    elif path.name == "query_generation_test":
        prefix = "test - "
    missing = "" if path.exists() else " (missing)"
    return f"{prefix}{relative_path(path)}{missing}"


def mode_label(mode: str) -> str:
    labels = {
        "visual_only": "visual_only",
        "visual_and_text": "visual_and_text",
        "default": "default / legacy",
    }
    return labels.get(mode, mode)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def dedupe_strings(values: list[str]) -> list[str]:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def render_no_queries_message(collection: Path) -> None:
    st.warning(f"No queries found under `{relative_path(collection)}`.")
    root = collection.parent if collection.parent.name in PRIORITY_COLLECTION_IDS else collection
    command = GENERATE_QUERY_COMMANDS.get(root.name)
    if command:
        st.info(
            "Generate the selected split collection with:\n\n"
            f"```bash\n{command}\n```"
        )
        return
    expected = [QUERY_COLLECTION_ROOT / collection_id for collection_id in PRIORITY_COLLECTION_IDS]
    if not any(path.exists() for path in expected):
        st.info(
            "Expected train/test query collections were not found. Generate them with:\n\n"
            "```bash\n"
            "SPLIT=train COLLECTION_ID=query_generation_train scripts/08_generate_queries.sh\n"
            "SPLIT=test COLLECTION_ID=query_generation_test scripts/08_generate_queries.sh\n"
            "```"
        )


def filter_examples(examples: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    conditions = sorted({condition_name(example) for example in examples})
    with st.sidebar:
        st.header("Filters")
        selected_conditions = st.multiselect("Generation condition", conditions, default=conditions)
        selected_statuses = st.multiselect("Review status", RATINGS, default=RATINGS)
        search = st.text_input("Paper or example ID").strip().lower()
        st.toggle("Show figure", value=True, key="show_figure")
        st.toggle("Show clues", value=True, key="show_clues")

    filtered = []
    for example in examples:
        if condition_name(example) not in selected_conditions:
            continue
        rating = annotations.get(annotation_key_for_example(example), {}).get("rating", "Unreviewed")
        if rating not in selected_statuses:
            continue
        if search:
            haystack = " ".join(
                [
                    str(example.get("query_id", "")),
                    source_record_id(example),
                    paper_id_for_example(example, {}),
                    " ".join(str(value) for value in example.get("relevant_ids", [])),
                ]
            ).lower()
            if search not in haystack:
                continue
        filtered.append(example)
    return filtered


def controls(examples: list[dict[str, Any]]) -> None:
    cols = st.columns([1, 1, 1, 5])
    if cols[0].button("Previous", use_container_width=True):
        st.session_state.example_index = max(0, st.session_state.example_index - 1)
        st.rerun()
    if cols[1].button("Next", use_container_width=True):
        st.session_state.example_index = min(len(examples) - 1, st.session_state.example_index + 1)
        st.rerun()
    if cols[2].button("Random", use_container_width=True):
        st.session_state.example_index = random.randrange(len(examples))
        st.rerun()


def render_example(
    collection: Path,
    example: dict[str, Any],
    record: dict[str, Any],
    same_paper_records: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    visual_clues: list[dict[str, str]],
    textual_clues: list[dict[str, str]],
    dataset_dir: Path,
) -> None:
    selected = selected_components(example)
    render_example_header(example, record)
    render_pdf_panel(example, record, dataset_dir)
    left, right = st.columns([1, 1])
    with left:
        if st.session_state.get("show_figure", True):
            render_paper_figures(same_paper_records, visual_rows, selected, dataset_dir)
    with right:
        render_paper_text(record, dataset_dir)
        if st.session_state.get("show_clues", True):
            st.subheader("Textual Clues")
            render_clues(textual_clues, selected, "textual", box_class="clues-scroll")

    st.subheader("Generated Query")
    st.markdown(f"<div class='query-box'>{html.escape(str(example.get('query', '')))}</div>", unsafe_allow_html=True)
    render_query_metadata(example)
    render_selected_components(selected)
    render_missing_selected(visual_clues + textual_clues, selected)
    with st.expander("Paper JSON Description"):
        st.json(metadata_context(collection, example, record), expanded=False)


def render_example_header(example: dict[str, Any], record: dict[str, Any]) -> None:
    paper_id = html.escape(resolved_paper_id(example, record) or "unknown paper")
    query_id = html.escape(str(example.get("query_id", "unknown query")))
    st.markdown(
        f"""
        <div class='review-header'>
          <div class='review-title-row'>
            <div class='paper-id'>{paper_id}</div>
            <div class='query-id'>{query_id}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pdf_panel(example: dict[str, Any], record: dict[str, Any], dataset_dir: Path) -> None:
    paper_id = resolved_paper_id(example, record)
    pdf_url = pdf_url_for_paper(paper_id)
    local_pdf = local_pdf_for_record(record, dataset_dir, paper_id)
    expected_pdf = canonical_pdf_path(dataset_dir, paper_id)
    with st.expander("PDF", expanded=False):
        info_col, pdf_col = st.columns([1, 3])
        with info_col:
            st.markdown(f"**Paper ID**  \n`{paper_id or 'unknown'}`")
            if local_pdf:
                st.markdown(f"**Local PDF**  \n`{local_pdf.relative_to(PROJECT_ROOT)}`")
                if st.checkbox("Prepare download", key=f"prepare_pdf_download_{paper_id}"):
                    st.download_button(
                        "Download PDF",
                        data=local_pdf.read_bytes(),
                        file_name=local_pdf.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            elif expected_pdf:
                st.markdown(f"**Expected PDF**  \n`{expected_pdf.relative_to(PROJECT_ROOT)}`")
            if pdf_url:
                st.link_button("Open ACL PDF", pdf_url, use_container_width=True)
            else:
                st.caption("No paper ID was available to construct a PDF URL.")
        with pdf_col:
            if local_pdf:
                if st.checkbox("Show small PDF preview", key=f"show_pdf_preview_{paper_id}"):
                    page_count = st.slider("Preview pages", min_value=1, max_value=3, value=1)
                    with st.container(height=420, border=True):
                        render_local_pdf_pages(local_pdf, page_count)
                else:
                    st.caption("PDF preview is not loaded until requested, which keeps example navigation fast.")
            else:
                st.info("Local PDF not found at the expected canonical path.")


def render_clues(
    clues: list[dict[str, str]],
    selected: list[dict[str, str]],
    kind: str,
    *,
    box_class: str = "",
) -> None:
    classes = f"scroll-box {box_class}".strip()
    if not clues:
        st.markdown(
            f"<div class='{classes}'>No {html.escape(kind)} clues found for the same paper in the current sidecar.</div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(f"<div class='{classes}'>{''.join(render_clue_items(clues, selected, kind))}</div>", unsafe_allow_html=True)


def render_clue_items(clues: list[dict[str, str]], selected: list[dict[str, str]], kind: str) -> list[str]:
    selected_keys = {clue_key(item) for item in selected if item.get("kind") == kind}
    selected_text_keys = {text_key(item.get("text", "")) for item in selected if item.get("kind") == kind}
    rendered = []
    for clue in clues:
        is_selected = clue_key(clue) in selected_keys or text_key(clue.get("text", "")) in selected_text_keys
        css = "clue clue-selected" if is_selected else "clue"
        badge = "selected" if is_selected else "unused"
        category = html.escape(str(clue.get("category", kind)))
        text = html.escape(str(clue.get("text", "")))
        record_id = html.escape(str(clue.get("record_id", "")))
        rendered.append(
            f"<div class='{css}'><strong>{badge}</strong> <span class='small-muted'>{category} | {record_id}</span><br>{text}</div>",
        )
    return rendered


def render_paper_figures(
    same_paper_records: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    selected: list[dict[str, str]],
    dataset_dir: Path,
) -> None:
    st.subheader("Paper Figures")
    if not same_paper_records:
        st.warning("No same-paper figure records found.")
        return
    with st.container(height=680, border=True):
        index = 1
        for record in same_paper_records:
            figures = record.get("figures", [record]) if isinstance(record.get("figures"), list) else [record]
            for figure_record in figures:
                title = figure_record.get("filename") or figure_record.get("figure_id") or figure_record.get("record_id") or f"Figure {index}"
                st.markdown(f"<div class='figure-title'>{html.escape(str(title))}</div>", unsafe_allow_html=True)
                image_path = image_for_record(figure_record, dataset_dir)
                if image_path and image_path.exists():
                    st.image(str(image_path), caption=str(figure_record.get("label") or ""), use_container_width=True)
                else:
                    st.caption("Figure image not found.")
                if st.session_state.get("show_clues", True):
                    clues = visual_clues_for_record(visual_rows, figure_record)
                    if clues:
                        st.markdown(
                            "".join(render_clue_items(clues, selected, "visual")),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No visual clues found for this figure.")
                st.divider()
                index += 1


def render_paper_text(record: dict[str, Any], dataset_dir: Path) -> None:
    markdown = str(record.get("markdown", "") or "")
    if not markdown and record.get("markdown_relpath"):
        markdown_path = dataset_dir / str(record.get("markdown_relpath"))
        if markdown_path.exists():
            markdown = markdown_path.read_text(encoding="utf-8")
    
    if not markdown:
        return
    st.subheader("Paper Text")
    st.markdown(
        f"<div class='scroll-box paper-scroll paper-text'>{html.escape(markdown[:20000])}</div>",
        unsafe_allow_html=True,
    )


def render_missing_selected(all_clues: list[dict[str, str]], selected: list[dict[str, str]]) -> None:
    clue_keys = {clue_key(clue) for clue in all_clues}
    text_keys = {(clue.get("kind", ""), text_key(clue.get("text", ""))) for clue in all_clues}
    missing = [
        clue
        for clue in selected
        if clue_key(clue) not in clue_keys and (clue.get("kind", ""), text_key(clue.get("text", ""))) not in text_keys
    ]
    if not missing:
        return
    with st.expander("Selected Clues Not Found In Current Sidecars"):
        for clue in missing:
            st.markdown(
                "<div class='clue clue-missing'><strong>selected</strong> "
                f"<span class='small-muted'>{html.escape(str(clue.get('kind', '')))} | {html.escape(str(clue.get('record_id', '')))}</span><br>"
                f"{html.escape(str(clue.get('text', '')))}</div>",
                unsafe_allow_html=True,
            )


def render_query_metadata(example: dict[str, Any]) -> None:
    metadata = query_metadata(example)
    st.subheader("Query Metadata")
    current_fields = [
        ("split", "Split"),
        ("split_index", "Split index"),
        ("split_paper_index", "Split paper index"),
        ("component_selection", "Component selection"),
        ("selected_component_count", "Selected components"),
        ("selected_visual_count", "Selected visual"),
        ("selected_text_count", "Selected textual"),
    ]
    rows = [{"Field": label, "Value": format_metadata_value(metadata.get(key))} for key, label in current_fields if key in metadata]
    if rows:
        st.table(rows)
    else:
        st.caption("This query does not include the newer split/component metadata fields.")

    legacy_fields = [
        ("component_budget", "Component budget"),
        ("visual_component_budget", "Visual component budget"),
        ("textual_component_budget", "Textual component budget"),
        ("max_text_components", "Max text components"),
    ]
    legacy_rows = [
        {"Legacy field": label, "Value": format_metadata_value(metadata.get(key))}
        for key, label in legacy_fields
        if key in metadata and metadata.get(key) is not None
    ]
    if legacy_rows:
        with st.expander("Legacy Selection Budget Fields", expanded=False):
            st.caption("These fields may appear in older metadata but are ignored by all-available component selection.")
            st.table(legacy_rows)


def render_selected_components(selected: list[dict[str, str]]) -> None:
    st.subheader("Selected Components")
    if not selected:
        st.caption("No selected components were recorded in this query metadata.")
        return
    grouped: dict[str, list[dict[str, str]]] = {"visual": [], "textual": [], "other": []}
    for component in selected:
        kind = normalized_kind(component.get("kind"))
        grouped.setdefault(kind, []).append(component)
    for kind in ("visual", "textual", "other"):
        components = grouped.get(kind, [])
        if not components:
            continue
        expanded = len(selected) <= 8
        with st.expander(f"{kind.title()} Components ({len(components)})", expanded=expanded):
            for index, component in enumerate(components, start=1):
                record_id = html.escape(str(component.get("record_id", "")))
                text = html.escape(str(component.get("text", "")))
                category = html.escape(str(component.get("category", kind)))
                st.markdown(
                    f"<div class='clue clue-selected'><strong>{index}</strong> "
                    f"<span class='small-muted'>{category} | {record_id}</span><br>{text}</div>",
                    unsafe_allow_html=True,
                )


def format_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render_annotation(
    collection: Path,
    example: dict[str, Any],
    annotation: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
) -> None:
    st.subheader("Manual Annotation")
    key_base = annotation_key(collection, example)
    cols = st.columns([1, 2])
    rating = cols[0].radio(
        "Rating",
        ["Good", "Questionable", "Bad"],
        index=["Good", "Questionable", "Bad"].index(annotation["rating"]) if annotation.get("rating") in ["Good", "Questionable", "Bad"] else None,
        horizontal=True,
        key=f"rating_{key_base}",
    )
    chosen = []
    for issue in ISSUES:
        checked = cols[1].checkbox(issue, value=issue in annotation.get("issues", []), key=f"issue_{key_base}_{issue}")
        if checked:
            chosen.append(issue)
    note = st.text_area("Note", value=annotation.get("note", ""), key=f"note_{key_base}", height=100)
    current = {"rating": rating or "Unreviewed", "issues": chosen, "note": note}
    if annotation_changed(annotation, current):
        save_annotation(collection, example, current)
        annotations[annotation_key_for_example(example)] = current


@st.cache_data(show_spinner=False)
def load_examples(collection: Path) -> list[dict[str, Any]]:
    examples = []
    
    # Support both direct queries.jsonl and subfolders
    paths = []
    if (collection / "queries.jsonl").exists():
        paths.append(collection / "queries.jsonl")
    paths.extend(collection.glob("*/queries.jsonl"))
    
    for query_path in sorted(set(paths)):
        condition = query_path.parent.name if query_path.parent != collection else "default"
        for line in query_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("query"):
                continue
            row["_condition"] = query_metadata(row).get("mode", condition)
            row["_collection"] = str(collection.relative_to(PROJECT_ROOT))
            examples.append(row)
    return examples


@st.cache_data(show_spinner=False)
def load_dataset_records(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    data_path = dataset_dir / "data.parquet"
    if data_path.exists():
        frame = pd.read_parquet(data_path)
    elif (dataset_dir / "data.jsonl").exists():
        frame = pd.read_json(dataset_dir / "data.jsonl", orient="records", lines=True)
    elif (dataset_dir / "data.csv").exists():
        frame = pd.read_csv(dataset_dir / "data.csv")
    elif (dataset_dir / "papers.jsonl").exists():
        frame = pd.read_json(dataset_dir / "papers.jsonl", orient="records", lines=True)
    else:
        return {}
    records = {}
    for row in frame.to_dict(orient="records"):
        cleaned = clean_values(row)
        paper_id = str(row.get("record_id") or row.get("paper_id") or "")
        if paper_id:
            records[paper_id] = cleaned
        for fig in row.get("figures", []):
            fig_id = str(fig.get("figure_id") or "")
            if fig_id:
                records[fig_id] = cleaned
            for legacy_id in fig.get("legacy_record_ids", []):
                records[str(legacy_id)] = cleaned
    return records


@st.cache_data(show_spinner=False)
def load_interpretations(path: Path) -> list[dict[str, Any]]:
    data_path = path / "interpretations.jsonl" if path.is_dir() else path
    if not data_path.exists() or not data_path.is_file():
        return []
    return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return annotations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if key:
            annotations[key] = row
    return annotations


def save_annotation(collection: Path, example: dict[str, Any], values: dict[str, Any]) -> None:
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "key": annotation_key_for_example(example),
        "collection": str(collection.relative_to(PROJECT_ROOT)),
        "condition": condition_name(example),
        "query_id": example.get("query_id"),
        "source_record_id": source_record_id(example),
        "resolved_paper_id": paper_id_for_example(example, {}),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    with ANNOTATIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
        handle.write("\n")


def dataset_dir_for_collection(collection: Path) -> Path:
    if collection == DEFAULT_COLLECTION:
        return DEFAULT_DATASET
    for candidate in (collection, collection.parent):
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset = metadata.get("config", {}).get("dataset")
        if dataset:
            path = Path(dataset)
            return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_DATASET


def clues_for_paper(rows: list[dict[str, Any]], same_paper_records: list[dict[str, Any]], kind: str) -> list[dict[str, str]]:
    record_ids = set()
    filenames = set()
    for row in same_paper_records:
        add_if_present(record_ids, row.get("record_id") or row.get("paper_id"))
        add_if_present(filenames, row.get("filename"))
        for fig in row.get("figures", []):
            add_if_present(record_ids, fig.get("figure_id"))
            add_if_present(record_ids, fig.get("record_id"))
            add_if_present(filenames, fig.get("filename"))
    clues = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if row.get("kind") and row.get("kind") != kind:
            continue
        row_id = str(row.get("record_id", "") or row.get("figure_id", "") or row.get("paper_id", ""))
        filename = str(metadata.get("filename", "") or "")
        id_match = bool(row_id) and row_id in record_ids
        filename_match = bool(filename) and filename in filenames
        if not id_match and not filename_match:
            continue
        clues.extend(parse_clues(str(row.get("output") or row.get("text", "")), kind, row_id))
    return clues


def add_if_present(values: set[str], value: Any) -> None:
    text = str(value or "").strip()
    if text:
        values.add(text)


def visual_clues_for_record(rows: list[dict[str, Any]], figure_record: dict[str, Any]) -> list[dict[str, str]]:
    record_id = str(figure_record.get("record_id", "") or figure_record.get("figure_id", ""))
    filename = str(figure_record.get("filename", ""))
    clues = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if row.get("kind") and row.get("kind") != "visual":
            continue
        row_id = str(row.get("record_id", "") or row.get("figure_id", ""))
        if row_id != record_id and str(metadata.get("filename", "")) != filename:
            continue
        clues.extend(parse_clues(str(row.get("output") or row.get("text", "")), "visual", row_id))
    return clues


def parse_clues(text: str, kind: str, record_id: str) -> list[dict[str, str]]:
    components = _split_component_text(text, kind)
    clues = []
    for comp in components:
        clues.append({"kind": kind, "record_id": record_id, "category": kind, "text": comp})
    return clues


def decode_jsonish(text: str) -> list[Any] | None:
    decoder = json.JSONDecoder()
    index = 0
    values = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return None
        values.append(value)
        index = end
    return values or None


def records_by_paper(records: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for record in records.values():
        paper_id = str(record.get("paper_id", "") or record.get("resolved_paper_id", "") or "")
        if paper_id:
            key = (paper_id, id(record))
            if key in seen:
                continue
            seen.add(key)
            grouped.setdefault(paper_id, []).append(record)
    return grouped


def metadata_context(collection: Path, example: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection": str(collection.relative_to(PROJECT_ROOT)),
        "query_example": {key: value for key, value in example.items() if not key.startswith("_")},
        "paper_record": {key: value for key, value in record.items() if key != "markdown"},
    }


def selected_components(example: dict[str, Any]) -> list[dict[str, str]]:
    components = query_metadata(example).get("selected_components", [])
    return [normalize_selected_component(component) for component in components if isinstance(component, dict)]


def normalize_selected_component(component: dict[str, Any]) -> dict[str, str]:
    kind = normalized_kind(component.get("kind"))
    return {
        **component,
        "kind": kind,
        "record_id": str(component.get("record_id") or component.get("figure_id") or component.get("paper_id") or ""),
        "category": str(component.get("category") or kind),
        "text": str(component.get("text") or component.get("description") or component.get("output") or ""),
    }


def normalized_kind(value: Any) -> str:
    cleaned = str(value or "").strip().replace("-", "_").casefold()
    if cleaned in {"visual", "image", "figure", "visual_clue"}:
        return "visual"
    if cleaned in {"textual", "text", "paper_text", "textual_clue"}:
        return "textual"
    return cleaned or "other"


def sync_index(examples: list[dict[str, Any]]) -> None:
    if "example_index" not in st.session_state:
        st.session_state.example_index = 0
    st.session_state.example_index = min(st.session_state.example_index, len(examples) - 1)


def annotation_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    old = {
        "rating": previous.get("rating", "Unreviewed"),
        "issues": sorted(previous.get("issues", [])),
        "note": previous.get("note", ""),
    }
    new = {"rating": current["rating"], "issues": sorted(current["issues"]), "note": current["note"]}
    return old != new and (new["rating"] != "Unreviewed" or new["issues"] or new["note"])


def annotation_key(collection: Path, example: dict[str, Any]) -> str:
    return f"{collection.relative_to(PROJECT_ROOT)}::{condition_name(example)}::{example.get('query_id')}"


def annotation_key_for_example(example: dict[str, Any]) -> str:
    return f"{example.get('_collection')}::{condition_name(example)}::{example.get('query_id')}"


def source_record_id(example: dict[str, Any]) -> str:
    metadata = query_metadata(example)
    if metadata.get("source_record_id"):
        return str(metadata["source_record_id"])
    if metadata.get("paper_id"):
        return str(metadata["paper_id"])
    relevant_ids = example.get("relevant_ids", [])
    return str(relevant_ids[0]) if relevant_ids else ""


def resolved_paper_id(example: dict[str, Any], record: dict[str, Any]) -> str:
    return paper_id_for_example(example, record)


def paper_id_for_example(example: dict[str, Any], record: dict[str, Any]) -> str:
    metadata = query_metadata(example)
    relevant_ids = example.get("relevant_ids", [])
    return str(
        metadata.get("resolved_paper_id")
        or metadata.get("paper_id")
        or record.get("paper_id")
        or record.get("resolved_paper_id")
        or (relevant_ids[0] if relevant_ids else "")
        or ""
    )


def condition_name(example: dict[str, Any]) -> str:
    return str(query_metadata(example).get("mode") or example.get("_condition", "")).replace("-", "_")


def query_metadata(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def image_for_record(record: dict[str, Any], dataset_dir: Path) -> Path | None:
    image_path = record.get("image_path")
    if image_path:
        path = Path(str(image_path))
        if path.exists():
            return path
    image_relpath = record.get("image_relpath")
    if image_relpath:
        return dataset_dir / str(image_relpath)
    return None


def local_pdf_for_record(record: dict[str, Any], dataset_dir: Path, paper_id: str) -> Path | None:
    canonical_pdf = canonical_pdf_path(dataset_dir, paper_id)
    if canonical_pdf and canonical_pdf.exists():
        return canonical_pdf
    for field in ("pdf_path", "paper_pdf_path"):
        value = record.get(field)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = dataset_dir / path
            if path.exists():
                return path
    for field in ("pdf_relpath", "paper_pdf_relpath"):
        value = record.get(field)
        if value:
            path = dataset_dir / str(value)
            if path.exists():
                return path
    return None


def canonical_pdf_path(dataset_dir: Path, paper_id: str) -> Path | None:
    cleaned = str(paper_id or "").strip()
    if not cleaned:
        return None
    return dataset_dir / "pdfs" / f"{cleaned}.pdf"


def render_local_pdf_pages(path: Path, page_count: int) -> None:
    try:
        pages = render_pdf_pages(path, page_count)
    except RuntimeError as error:
        st.warning(str(error))
        return
    if not pages:
        st.info("No preview pages were rendered for this PDF.")
        return
    for index, page in enumerate(pages, start=1):
        st.image(page, caption=f"Page {index}", use_container_width=True)


@st.cache_data(show_spinner=False)
def render_pdf_pages(path: Path, page_count: int) -> list[bytes]:
    pdftoppm = find_pdftoppm()
    if not pdftoppm:
        raise RuntimeError("PDF page preview requires pdftoppm, but it was not found on PATH.")
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        result = subprocess.run(
            [
                pdftoppm,
                "-png",
                "-r",
                "130",
                "-f",
                "1",
                "-l",
                str(page_count),
                str(path),
                str(prefix),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "pdftoppm failed to render the PDF."
            raise RuntimeError(message)
        return [page.read_bytes() for page in sorted(Path(tmpdir).glob("page-*.png"))]


def find_pdftoppm() -> str | None:
    found = shutil.which("pdftoppm")
    if found:
        return found
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm"
    if bundled.exists():
        return str(bundled)
    return None


def pdf_url_for_paper(paper_id: str) -> str:
    cleaned = str(paper_id or "").strip()
    if not cleaned:
        return ""
    return f"https://aclanthology.org/{cleaned}.pdf"


def clean_values(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in row.items():
        missing = False
        if not isinstance(value, (list, dict)):
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
        if missing:
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def clue_key(clue: dict[str, Any]) -> tuple[str, str, str]:
    return (str(clue.get("kind", "")), str(clue.get("record_id", "")), text_key(clue.get("text", "")))


def text_key(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip()).casefold()
    cleaned = re.sub(r"^figure\s*\d+[a-z]?:\s*", "", cleaned)
    return cleaned


if __name__ == "__main__":
    main()
