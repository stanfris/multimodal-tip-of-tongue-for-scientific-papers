"""Generate visual figure descriptions with Qwen-VL backends."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from dataset_generation.interpretations import (
    InterpretationRecord,
    append_failure_record,
    interpretation_key,
    read_completed_interpretation_keys,
)
from dataset_generation.preprocessed import (
    DEFAULT_DATA_DIR,
    append_clue_row,
    read_clue_rows,
    read_preprocessed_papers,
    visual_clue_path,
)


DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "transformers": "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_PROMPT = Path("prompts/visual_interpretation.v1.txt")
DEFAULT_RUN_ID = "qwen3_vl_figure_description"
T = TypeVar("T")


@dataclass(frozen=True)
class FigureSample:
    record_id: str
    image_path: Path
    metadata: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Qwen-VL visual descriptions for scientific figures.")
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODELS), default="mlx")
    parser.add_argument("--model", default=None, help="Model name. Defaults depend on --backend.")
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
    parser.add_argument("--prompt-id", default="visual_interpretation", help="Prompt identifier for metadata.")
    parser.add_argument("--prompt-version", default="v1", help="Prompt version for metadata.")
    parser.add_argument("--split", default="train", help="Combined dataset split to read when --dataset is omitted.")
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
    parser.add_argument("--batch-size", type=int, default=1, help="Scheduling batch size; visual generations run serially.")
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Optional local image path. Repeat to pass known figure images.",
    )
    parser.add_argument("--max-tokens", type=int, default=180, help="Concise generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0.0 for deterministic decoding.")
    parser.add_argument("--debug", action="store_true", help="Print image path, prompt, and raw model output.")
    parser.add_argument("--device-map", default="auto", help="Transformers backend device_map.")
    parser.add_argument("--dtype", default="auto", help="Transformers backend dtype.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, e.g. flash_attention_2 on DGX.",
    )
    return parser


def safe_stem(value: str, fallback: str) -> str:
    stem = Path(value).stem or fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or fallback


def samples_from_paths(paths: list[Path], limit: int) -> list[FigureSample]:
    samples: list[FigureSample] = []
    for index, image_path in enumerate(paths[:limit], start=1):
        resolved = image_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Image does not exist: {resolved}")
        samples.append(
            FigureSample(
                record_id=f"local-{index}",
                image_path=resolved,
                metadata={"source": "local_path", "filename": resolved.name},
            )
        )
    return samples


def iter_samples_from_dataset(
    dataset_dir: Path,
    *,
    limit: int | None,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[FigureSample]:
    if start_index < 0:
        raise ValueError("start-index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end-index must be greater than or equal to start-index")
    return iter_samples_from_preprocessed_dataset(
        dataset_dir,
        limit=limit,
        start_index=start_index,
        end_index=end_index,
    )


def iter_samples_from_preprocessed_dataset(
    dataset_dir: Path,
    *,
    limit: int | None,
    start_index: int = 0,
    end_index: int | None = None,
) -> list[FigureSample]:
    samples: list[FigureSample] = []
    figure_index = 0
    for paper in read_preprocessed_papers(dataset_dir):
        for figure in paper.get("figures", []):
            if figure_index < start_index:
                figure_index += 1
                continue
            if end_index is not None and figure_index >= end_index:
                break
            image_path = Path(str(figure["image_path"])).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Canonical figure image does not exist: {image_path}")
            samples.append(
                FigureSample(
                    record_id=str(figure["figure_id"]),
                    image_path=image_path,
                    metadata={
                        "source_entity": "figure",
                        "figure_index": figure_index,
                        "paper_id": paper["paper_id"],
                        "resolved_paper_id": paper["paper_id"],
                        "figure_id": figure["figure_id"],
                        "filename": figure.get("filename"),
                        "paper_dir": paper.get("paper_dir"),
                        "image_relpath": figure.get("image_relpath"),
                        "image_path": str(image_path),
                    },
                )
            )
            figure_index += 1
            if limit is not None and len(samples) >= limit:
                break
        if end_index is not None and figure_index >= end_index:
            break
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError(f"No preprocessed figures with images found in {dataset_dir}.")
    return samples


def load_mlx_model(model_name: str) -> dict[str, Any]:
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model, processor = load(model_name)
    config = load_config(model_name)
    return {"model": model, "processor": processor, "config": config}


def generate_with_mlx(loaded: dict[str, Any], image_path: Path, prompt: str, max_tokens: int, temperature: float) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted_prompt = apply_chat_template(loaded["processor"], loaded["config"], prompt, num_images=1)
    output = generate(
        loaded["model"],
        loaded["processor"],
        formatted_prompt,
        [str(image_path)],
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    text = getattr(output, "text", output)
    return str(text).strip()


def load_transformers_model(
    model_name: str,
    device_map: str,
    dtype: str,
    attn_implementation: str | None,
) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return {"model": model, "processor": processor}


def generate_with_transformers(
    loaded: dict[str, Any],
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    model = loaded["model"]
    processor = loaded["processor"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False
    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


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


def print_sample(index: int, sample: FigureSample, backend: str, model_name: str, description: str) -> None:
    figure_id = sample.metadata.get("filename") or sample.record_id
    print("=" * 60)
    print(f"SAMPLE {index}")
    print(f"Record ID: {sample.record_id}")
    print(f"Figure ID: {figure_id}")
    if sample.metadata.get("label") is not None:
        print(f"Label: {sample.metadata['label']}")
    print(f"Image: {sample.image_path}")
    print(f"Backend: {backend}")
    print(f"Model: {model_name}")
    print()
    print("DESCRIPTION:")
    print(description)
    print("=" * 60)


def batched(items: list[FigureSample], batch_size: int) -> list[list[FigureSample]]:
    if batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def progress(items: Iterable[T], *, total: int, enabled: bool, description: str) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        processed = 0
        next_report = 10
        for item in items:
            yield item
            processed += 1
            percent = int((processed / total) * 100) if total else 100
            if percent >= next_report or processed == total:
                print(f"{description}: {processed}/{total} rows ({percent}%)", flush=True)
                next_report += 10
        return
    yield from tqdm(items, total=total, desc=description, unit="batch")


def print_debug_trace(sample: FigureSample, prompt: str, description: str) -> None:
    print("=" * 60, flush=True)
    print("DEBUG VISUAL GENERATION", flush=True)
    print(f"Record ID: {sample.record_id}", flush=True)
    print(f"Image: {sample.image_path}", flush=True)
    print(f"Metadata: {json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True)}", flush=True)
    print("PROMPT:", flush=True)
    print(prompt, flush=True)
    print("MODEL OUTPUT:", flush=True)
    print(description, flush=True)
    print("=" * 60, flush=True)


def run(args: argparse.Namespace) -> Path:
    model_name = args.model or DEFAULT_MODELS[args.backend]
    dataset_dir = args.dataset or (Path(args.data_dir) / "preprocessed")
    output_dir = args.output_dir or args.clues_dir or (Path(args.data_dir) / "clues")
    prompt_path = args.prompt.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    limit = args.limit if args.all else max(1, min(args.num_samples, 5))

    if args.image:
        samples = samples_from_paths(args.image, limit if limit is not None else len(args.image))
    else:
        samples = iter_samples_from_dataset(
            dataset_dir,
            limit=limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )

    loaded, generate_description = load_generator(args, model_name)
    records: list[InterpretationRecord] = []
    completed = set()
    if args.resume and not args.overwrite:
        completed = read_completed_interpretation_keys(
            output_dir,
            kind="visual",
            model=model_name,
            prompt_id=args.prompt_id,
            prompt_version=args.prompt_version,
        )
        completed.update(
            _completed_visual_keys_from_clues(
                output_dir,
                read_preprocessed_papers(dataset_dir),
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    failure_file = output_path / "visual_failures.jsonl"
    if args.all and args.overwrite:
        for sample in samples:
            visual_clue_path(
                output_path,
                str(sample.metadata["paper_id"]),
                str(sample.metadata["figure_id"]),
            ).unlink(missing_ok=True)
        failure_file.unlink(missing_ok=True)
    started = time.time()
    processed = 0
    skipped = 0
    failed = 0

    batches = batched(samples, args.batch_size)
    for batch in progress(batches, total=len(batches), enabled=args.all, description="Figure description batches"):
        for sample in batch:
            candidate = InterpretationRecord(
                record_id=sample.record_id,
                kind="visual",
                text="",
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
            )
            if interpretation_key(candidate) in completed:
                skipped += 1
                continue
            try:
                description = generate_description(
                    loaded=loaded,
                    image_path=sample.image_path,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                if args.debug:
                    print_debug_trace(sample, prompt, description)
            except Exception as exc:
                failed += 1
                append_failure_record(
                    {
                        "record_id": sample.record_id,
                        "kind": "visual",
                        "image_path": str(sample.image_path),
                        "metadata": sample.metadata,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    failure_file,
                )
                continue
            record = InterpretationRecord(
                record_id=sample.record_id,
                kind="visual",
                text=description,
                model=model_name,
                prompt_id=args.prompt_id,
                prompt_version=args.prompt_version,
                metadata={
                    **sample.metadata,
                    "backend": args.backend,
                    "image_path": str(sample.image_path),
                },
            )
            records.append(record)
            _append_visual_clue(record, output_path)
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
            "all": args.all,
            "requested_limit": args.limit,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "batch_size": args.batch_size,
            "resume": args.resume,
            "debug": args.debug,
            "processed_records": processed,
            "skipped_existing_records": skipped,
            "failed_records": failed,
            "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_path / "visual_clues.metadata.json").write_text(
        json.dumps({"record_count": processed, **metadata}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {processed} interpretation records to {output_path.resolve()} ({skipped} skipped, {failed} failed)")
    return output_path


def _append_visual_clue(record: InterpretationRecord, clues_dir: str | Path) -> None:
    paper_id = str(record.metadata.get("paper_id") or record.metadata.get("resolved_paper_id"))
    figure_id = str(record.metadata.get("figure_id") or record.record_id)
    row = {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "kind": record.kind,
        "model": record.model,
        "prompt_id": record.prompt_id,
        "prompt_version": record.prompt_version,
        "output": record.text,
    }
    append_clue_row(visual_clue_path(clues_dir, paper_id, figure_id), row)


def _completed_visual_keys_from_clues(
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
        for figure in paper.get("figures", []):
            figure_id = str(figure["figure_id"])
            for clue in read_clue_rows(visual_clue_path(clues_dir, paper_id, figure_id)):
                if (
                    clue.get("kind") == "visual"
                    and clue.get("model") == model
                    and clue.get("prompt_id") == prompt_id
                    and clue.get("prompt_version") == prompt_version
                ):
                    completed.add((figure_id, "visual", model, prompt_id, prompt_version))
                    break
    return completed


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
