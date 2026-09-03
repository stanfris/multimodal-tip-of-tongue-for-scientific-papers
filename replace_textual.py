import re

with open("src/dataset_generation/textual_clue_descriptions.py", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "from dataset_generation.sources import DEFAULT_DATA_DIR, default_processed_dataset_dir",
    "from dataset_generation.sources import DEFAULT_DATA_DIR"
)
content = content.replace(
    "from dataset_generation.storage import read_dataset_artifact\n",
    ""
)

# 2. default_dataset_dir
content = re.sub(
    r"def default_dataset_dir\(data_dir: Path, split: str\) -> Path:\n    canonical_dir = Path\(data_dir\) / \"canonical\"\n    return canonical_dir if \(canonical_dir / \"papers.jsonl\"\).exists\(\) else default_processed_dataset_dir\(data_dir, split\)",
    "def default_dataset_dir(data_dir: Path, split: str) -> Path:\n    return Path(data_dir) / \"canonical\"",
    content
)

# 3. samples_from_dataset and iter_samples_from_dataset
import ast
# To simplify, we will just regex out samples_from_dataset to iter_samples_from_canonical_dataset
old_funcs = re.search(r"(def samples_from_dataset.*?)(?=def iter_samples_from_canonical_dataset)", content, re.DOTALL)
if old_funcs:
    new_funcs = """def iter_samples_from_dataset(
    dataset_dir: Path,
    *,
    limit: int | None,
    max_markdown_chars: int,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[TextSample]:
    if start_index < 0:
        raise ValueError("start-index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end-index must be greater than or equal to start-index")
    return iter_samples_from_canonical_dataset(
        dataset_dir,
        limit=limit,
        max_markdown_chars=max_markdown_chars,
        start_index=start_index,
        end_index=end_index,
    )\n\n\n"""
    content = content[:old_funcs.start()] + new_funcs + content[old_funcs.end():]

# 4. run function changes
# We need to simplify run()
content = re.sub(r"    canonical_mode = is_canonical_dataset\(dataset_dir\)\n", "", content)
content = content.replace(
"""    output_dir = args.output_dir or (
        canonical_dataset_path(dataset_dir).parent if canonical_mode else default_interpretation_dir(args.data_dir, args.run_id)
    )""",
"""    output_dir = args.output_dir or canonical_dataset_path(dataset_dir).parent"""
)
content = content.replace("    legacy_rows: list[dict[str, Any]] = []\n", "")

content = content.replace(
"""        if canonical_mode:
            completed.update(
                _completed_textual_keys_from_canonical(
                    dataset_dir,
                    model=model_name,
                    prompt_id=args.prompt_id,
                    prompt_version=args.prompt_version,
                )
            )""",
"""        completed.update(
            _completed_textual_keys_from_canonical(
                dataset_dir,
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
        )"""
)

content = content.replace(
"""    output_file = canonical_textual_clues_path(dataset_dir) if canonical_mode and args.output_dir is None else output_path / "interpretations.jsonl"
    failure_file = output_path / ("textual_failures.jsonl" if canonical_mode and args.output_dir is None else "failures.jsonl")""",
"""    output_file = canonical_textual_clues_path(dataset_dir) if args.output_dir is None else output_path / "interpretations.jsonl"
    failure_file = output_path / "textual_failures.jsonl" if args.output_dir is None else output_path / "failures.jsonl\""""
)

legacy_row_append = """            legacy_rows.append(
                {
                    "record_id": sample.record_id,
                    "metadata": sample.metadata,
                    "backend": args.backend,
                    "model_name": model_name,
                    "prompt": prompt,
                    "generated_description": description,
                }
            )
"""
content = content.replace(legacy_row_append, "")

output_file_logic = """            if canonical_mode and args.output_dir is None:
                _append_canonical_textual_clue(record, output_file)
            else:
                append_interpretation_record(record, output_file)"""
content = content.replace(output_file_logic, """            _append_canonical_textual_clue(record, output_file)""")


metadata_logic = """    if args.all or (canonical_mode and args.output_dir is None):
        metadata_name = "textual_clues.metadata.json" if canonical_mode and args.output_dir is None else "metadata.json"
        (output_path / metadata_name).write_text(
            json.dumps({"record_count": processed, **metadata}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        write_interpretations(records, output_dir, metadata)
        (output_path / "results.jsonl").write_text(
            "\\n".join(json.dumps(row, ensure_ascii=False) for row in legacy_rows)
            + ("\\n" if legacy_rows else ""),
            encoding="utf-8",
        )"""
content = content.replace(metadata_logic, """    metadata_name = "textual_clues.metadata.json" if args.output_dir is None else "metadata.json"
    (output_path / metadata_name).write_text(
        json.dumps({"record_count": processed, **metadata}, indent=2, sort_keys=True),
        encoding="utf-8",
    )""")

with open("src/dataset_generation/textual_clue_descriptions.py", "w") as f:
    f.write(content)

print("done textual")
