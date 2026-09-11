"""Generate textual memory cues from matched paper markdown."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dataset_generation.document_splits import filter_papers_by_split
from dataset_generation.generation_utils import batched, progress, record_generation_failure
from dataset_generation.interpretations import (
    InterpretationRecord,
    interpretation_key,
    read_completed_interpretation_keys,
)
from dataset_generation.managed_settings import (
    DEFAULT_SETTINGS_PATH,
    load_managed_settings,
    resolve_dataset_path,
    resolve_path,
    section,
)
from dataset_generation.preprocessed import (
    DEFAULT_DATA_DIR,
    append_clue_row,
    read_clue_rows,
    read_preprocessed_markdown,
    read_preprocessed_papers,
    textual_clue_path,
)
from dataset_generation.validation import validate_index_window


DEFAULT_MODELS: dict[str, str | None] = {
    "mlx": "Qwen/Qwen3-1.7B-MLX-8bit",
    "transformers": "Qwen/Qwen3-4B",
}
DEFAULT_PROMPT = Path("prompts/textual_interpretation.v1.txt")
DEFAULT_RUN_ID = "qwen3_textual_clue_description"
DEFAULT_MAX_MARKDOWN_CHARS = 30_000


@dataclass(frozen=True)
class TextSample:
    record_id: str
    markdown: str
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate textual memory cues from paper markdown.")
    parser.add_argument("--settings", type=Path, default=None, help="Managed settings YAML for standard full runs.")
    parser.add_argument("--set", choices=["train", "test"], default="train", help="Managed dataset split to process.")
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default="mlx")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to Qwen/Qwen3-1.7B-MLX-8bit for MLX.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local data root.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Interpretation artifact run ID.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Interpretation artifact directory.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Preprocessed papers directory. Defaults to data-dir/preprocessed.",
    )
    parser.add_argument("--clues-dir", type=Path, default=None, help="Clue output directory. Defaults to data-dir/clues.")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="Prompt template path.")
    parser.add_argument("--prompt-id", default="textual_interpretation", help="Prompt identifier for metadata.")
    parser.add_argument("--prompt-version", default="v1", help="Prompt version for metadata.")
    parser.add_argument("--split", default="train", help="Combined dataset split to read when --dataset is omitted.")
    parser.add_argument(
        "--split-index",
        type=Path,
        default=None,
        help="JSON train/test paper-id index. When set, --split selects the paper-id list to process.",
    )
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to process, clamped to 1-5.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every eligible dataset row. When unset, only a small smoke sample is generated.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for --all runs.")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row index to consider.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive dataset row index to consider.")
    parser.add_argument("--resume", action="store_true", help="Skip records already present in interpretation JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate records even when --resume is set.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for compatible backends.")
    parser.add_argument(
        "--max-markdown-chars",
        type=int,
        default=DEFAULT_MAX_MARKDOWN_CHARS,
        help="Maximum markdown characters passed to the model per sample.",
    )
    parser.add_argument("--max-tokens", type=int, default=700, help="Concise generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0.0 for deterministic decoding.")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking/reasoning mode when the tokenizer supports it. Disabled by default.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers backend device_map.")
    parser.add_argument("--dtype", default="auto", help="Transformers backend dtype.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, e.g. flash_attention_2 on DGX.",
    )
    return parser


def default_dataset_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "preprocessed"


def iter_samples_from_dataset(
    dataset_dir: Path,
    *,
    limit: int | None,
    max_markdown_chars: int,
    start_index: int = 0,
    end_index: int | None = None,
    split_index: Path | None = None,
    split_name: str | None = None,
) -> list[TextSample]:
    validate_index_window(start_index, end_index, start_name="start-index", end_name="end-index")
    return iter_samples_from_preprocessed_dataset(
        dataset_dir,
        limit=limit,
        max_markdown_chars=max_markdown_chars,
        start_index=start_index,
        end_index=end_index,
        split_index=split_index,
        split_name=split_name,
    )


def iter_samples_from_preprocessed_dataset(
    dataset_dir: Path,
    *,
    limit: int | None,
    max_markdown_chars: int,
    start_index: int = 0,
    end_index: int | None = None,
    split_index: Path | None = None,
    split_name: str | None = None,
) -> list[TextSample]:
    papers = filter_papers_by_split(
        read_preprocessed_papers(dataset_dir),
        split_index_path=split_index,
        split_name=split_name,
    )
    selected: list[TextSample] = []
    for paper_index, paper in enumerate(papers):
        if paper_index < start_index:
            continue
        if end_index is not None and paper_index >= end_index:
            break
        markdown = read_preprocessed_markdown(paper)
        if not markdown.strip():
            continue
        truncated = markdown[:max_markdown_chars]
        selected.append(
            TextSample(
                record_id=str(paper["paper_id"]),
                markdown=truncated,
                metadata={
                    "source_entity": "paper",
                    "paper_index": paper_index,
                    "paper_id": paper["paper_id"],
                    "resolved_paper_id": paper["paper_id"],
                    "paper_dir": paper["paper_dir"],
                    "markdown_path": paper["markdown_path"],
                    "markdown_chars": len(markdown),
                    "markdown_chars_used": len(truncated),
                    "markdown_truncated": len(truncated) < len(markdown),
                },
            )
        )
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise RuntimeError(f"No preprocessed papers with non-empty markdown found in {dataset_dir}.")
    return selected


def format_prompt(prompt_template: str, markdown: str) -> str:
    return prompt_template.replace("{paper_text}", markdown)


def apply_text_chat_template(tokenizer: Any, prompt: str, thinking: bool = False) -> str:
    messages = [{"role": "user", "content": prompt}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_mlx_model(model_name: str) -> dict[str, Any]:
    from mlx_lm import load

    model, tokenizer = load(model_name)
    return {"model": model, "tokenizer": tokenizer}


def generate_with_mlx(
    loaded: dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float,
    thinking: bool = False,
) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer = loaded["tokenizer"]
    prompt = apply_text_chat_template(tokenizer, prompt, thinking=thinking)
    output = generate(
        loaded["model"],
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
    )
    text = str(output).strip()
    return text if thinking else strip_thinking(text)


def load_transformers_model(
    model_name: str,
    device_map: str,
    dtype: str,
    attn_implementation: str | None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoProcessor.from_pretrained(model_name)
    except (OSError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return {"model": model, "tokenizer": tokenizer}


def generate_with_transformers(
    loaded: dict[str, Any],
    prompt: str,
    max_tokens: int,
    temperature: float,
    thinking: bool = False,
) -> str:
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    prompt = apply_text_chat_template(tokenizer, prompt, thinking=thinking)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False
    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    text = output_text[0].strip()
    return text if thinking else strip_thinking(text)


def generate_batch_with_transformers(
    loaded: dict[str, Any],
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    thinking: bool = False,
) -> list[str]:
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    formatted = [apply_text_chat_template(tokenizer, prompt, thinking=thinking) for prompt in prompts]

    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False
    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return [text.strip() if thinking else strip_thinking(text) for text in output_text]


def load_generator(args: argparse.Namespace, model_name: str) -> tuple[dict[str, Any], Callable[..., str]]:
    print(f"Loading backend: {args.backend}")
    print(f"Loading model: {model_name}")
    if args.backend == "mlx":
        return load_mlx_model(model_name), generate_with_mlx
    return (
        load_transformers_model(
            model_name=model_name,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        ),
        generate_with_transformers,
    )


def print_sample(index: int, sample: TextSample, backend: str, model_name: str, description: str) -> None:
    paper_id = sample.metadata.get("resolved_paper_id") or sample.record_id
    print("=" * 60)
    print(f"SAMPLE {index}")
    print(f"Record ID: {sample.record_id}")
    print(f"Paper ID: {paper_id}")
    print(f"Backend: {backend}")
    print(f"Model: {model_name}")
    print()
    print("TEXTUAL CUES:")
    print(description)
    print("=" * 60)


def generate_text_batch(
    args: argparse.Namespace,
    loaded: dict[str, Any],
    generate_description: Callable[..., str],
    samples: list[TextSample],
    prompt_template: str,
) -> list[str]:
    prompts = [format_prompt(prompt_template, sample.markdown) for sample in samples]
    if args.backend == "transformers" and len(prompts) > 1:
        return generate_batch_with_transformers(
            loaded=loaded,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
        )
    return [
        generate_description(
            loaded=loaded,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
        )
        for prompt in prompts
    ]


def run(args: argparse.Namespace) -> Path:
    if args.settings is not None:
        args = load_managed_textual_description_args(args)
    model_name = args.model or DEFAULT_MODELS[args.backend]
    if model_name is None:
        raise ValueError(f"No default model is configured for backend {args.backend!r}; pass --model explicitly.")
    dataset_dir = args.dataset or default_dataset_dir(args.data_dir)
    output_dir = args.output_dir or args.clues_dir or (Path(args.data_dir) / "clues")
    prompt_path = args.prompt.expanduser().resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    limit = args.limit if args.all else max(1, min(args.num_samples, 5))
    samples = iter_samples_from_dataset(
        dataset_dir,
        limit=limit,
        max_markdown_chars=args.max_markdown_chars,
        start_index=args.start_index,
        end_index=args.end_index,
        split_index=args.split_index,
        split_name=args.split,
    )

    loaded, generate_description = load_generator(args, model_name)
    completed = set()
    if args.resume and not args.overwrite:
        completed = read_completed_interpretation_keys(
            output_dir,
            kind="textual",
            model=model_name,
            prompt_id=args.prompt_id,
            prompt_version=args.prompt_version,
        )
        completed.update(
            _completed_textual_keys_from_clues(
                output_dir,
                read_preprocessed_papers(dataset_dir),
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    failure_file = output_path / "textual_failures.jsonl"
    if args.all and args.overwrite:
        for sample in samples:
            textual_clue_path(output_path, str(sample.metadata["paper_id"])).unlink(missing_ok=True)
        failure_file.unlink(missing_ok=True)
    started = time.time()
    processed = 0
    skipped = 0
    failed = 0

    batches = batched(samples, args.batch_size)
    for batch in progress(batches, total=len(batches), enabled=args.all, description="Textual clue batches"):
        pending: list[TextSample] = []
        for sample in batch:
            candidate = InterpretationRecord(
                record_id=sample.record_id,
                kind="textual",
                text="",
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
            if interpretation_key(candidate) in completed:
                skipped += 1
            else:
                pending.append(sample)
        if not pending:
            continue
        try:
            descriptions = generate_text_batch(args, loaded, generate_description, pending, prompt_template)
        except Exception as exc:
            failed += len(pending)
            for sample in pending:
                record_generation_failure(
                    failure_file,
                    record_id=sample.record_id,
                    kind="textual",
                    error=exc,
                    metadata=sample.metadata,
                )
            continue
        for sample, description in zip(pending, descriptions, strict=True):
            record = InterpretationRecord(
                record_id=sample.record_id,
                kind="textual",
                text=description,
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
                metadata={
                    **sample.metadata,
                    "backend": args.backend,
                    "thinking": args.thinking,
                },
            )
            _append_textual_clue(record, output_path)
            completed.add(interpretation_key(record))
            processed += 1
            if not args.all:
                print_sample(processed, sample, args.backend, model_name, description)

    metadata = {
            "backend": args.backend,
            "model": model_name,
            "prompt_path": str(prompt_path),
            "prompt_id": args.prompt_id,
            "prompt_version": args.prompt_version,
            "base_dataset": str(dataset_dir),
            "source_split": args.split,
            "split_index": str(args.split_index) if args.split_index is not None else None,
            "max_markdown_chars": args.max_markdown_chars,
            "thinking": args.thinking,
            "all": args.all,
            "requested_limit": args.limit,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "batch_size": args.batch_size,
            "resume": args.resume,
            "processed_records": processed,
            "skipped_existing_records": skipped,
            "failed_records": failed,
            "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_path / "textual_clues.metadata.json").write_text(
        json.dumps({"record_count": processed, **metadata}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {processed} interpretation records to {output_path.resolve()} ({skipped} skipped, {failed} failed)")
    return output_path


def load_managed_textual_description_args(args: argparse.Namespace) -> argparse.Namespace:
    raw, settings_path = load_managed_settings(args.settings or DEFAULT_SETTINGS_PATH)
    config_dir = settings_path.parent
    dataset = section(raw, "dataset")
    textual = section(raw, "textual_descriptions")
    prompt = section(textual, "prompt")
    model = section(textual, "model")
    selection = section(textual, "selection")
    generation = section(textual, "generation")

    managed = argparse.Namespace(**vars(args))
    managed.backend = model.get("provider", "transformers")
    managed.model = model.get("name")
    managed.data_dir = resolve_path(dataset.get("root", "../data"), config_dir)
    managed.dataset = resolve_dataset_path(dataset, "preprocessed", config_dir)
    managed.clues_dir = resolve_dataset_path(dataset, "clues_dir", config_dir)
    managed.output_dir = None
    managed.prompt = resolve_path(prompt["template"], config_dir)
    managed.prompt_id = prompt.get("name", "textual_interpretation")
    managed.prompt_version = prompt.get("version", "v1")
    managed.split = args.set
    managed.split_index = resolve_dataset_path(dataset, "split_index", config_dir)
    managed.all = bool(selection.get("all", True))
    managed.limit = selection.get("limit")
    managed.start_index = int(selection.get("start_index", 0))
    managed.end_index = selection.get("end_index")
    managed.resume = bool(selection.get("resume", True))
    managed.overwrite = bool(selection.get("overwrite", False))
    managed.batch_size = int(generation.get("batch_size", 1))
    managed.max_markdown_chars = int(generation.get("max_markdown_chars", DEFAULT_MAX_MARKDOWN_CHARS))
    managed.max_tokens = int(generation.get("max_tokens", 700))
    managed.temperature = float(generation.get("temperature", 0.0))
    managed.thinking = bool(generation.get("thinking", False))
    managed.device_map = generation.get("device_map", "auto")
    managed.dtype = generation.get("dtype", "auto")
    managed.attn_implementation = generation.get("attn_implementation")
    return managed


def _append_textual_clue(record: InterpretationRecord, clues_dir: str | Path) -> None:
    row = {
        "paper_id": record.record_id,
        "kind": record.kind,
        "model": record.model,
        "prompt_id": record.prompt_id,
        "prompt_version": record.prompt_version,
        "output": record.text,
    }
    append_clue_row(textual_clue_path(clues_dir, record.record_id), row)


def _completed_textual_keys_from_clues(
    clues_dir: Path,
    papers: list[dict[str, Any]],
    *,
    model: str,
    prompt_id: str,
    prompt_version: str,
) -> set[tuple[str, str, str, str, str]]:
    completed: set[tuple[str, str, str, str, str]] = set()
    for paper in papers:
        paper_id = str(paper["paper_id"])
        for clue in read_clue_rows(textual_clue_path(clues_dir, paper_id)):
            if (
                clue.get("kind") == "textual"
                and clue.get("model") == model
                and clue.get("prompt_id") == prompt_id
                and clue.get("prompt_version") == prompt_version
            ):
                completed.add((paper_id, "textual", model, prompt_id, prompt_version))
                break
    return completed


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
