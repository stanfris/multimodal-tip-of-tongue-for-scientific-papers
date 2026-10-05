"""Benchmark representative saved workloads without writing pipeline artifacts.

Example: python -m inference.benchmark visual --provider vllm --dataset data/preprocessed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common.generation_utils import batched
from inference import GenerationRequest, load_backend
from inference.base import ordered_results


MODELS = {"visual": "Qwen/Qwen3-VL-4B-Instruct", "text": "Qwen/Qwen3-4B",
          "query": "microsoft/phi-4", "judge": "google/gemma-3-27b-it"}
SEQS = {"visual": [32, 64, 128], "text": [128, 256, 512],
        "query": [64, 128, 256], "judge": [32, 64, 128]}


def representative_requests(stage: str, dataset: Path, clues_dir: Path, count: int) -> list[GenerationRequest]:
    if stage == "visual":
        from clues.vl_figure_descriptions import iter_samples_from_dataset
        prompt = Path("prompts/visual_interpretation.v2.txt").read_text(encoding="utf-8")
        samples = iter_samples_from_dataset(dataset, limit=count)
        return [GenerationRequest(str(index), prompt, images=(sample.image_path,), max_tokens=600)
                for index, sample in enumerate(samples)]
    if stage == "text":
        from clues.textual_clue_descriptions import format_prompt, iter_samples_from_dataset
        prompt = Path("prompts/textual_interpretation.v1.txt").read_text(encoding="utf-8")
        samples = iter_samples_from_dataset(dataset, limit=count, max_markdown_chars=12000)
        return [GenerationRequest(str(index), format_prompt(prompt, sample.markdown), max_tokens=600)
                for index, sample in enumerate(samples)]
    from preprocessing.preprocessed import read_preprocessed_papers
    from queries.query_generation import (_components_by_paper, select_components,
                                          format_query_prompt, format_judgement_prompt, paper_image_paths)
    papers = read_preprocessed_papers(dataset)
    components = _components_by_paper(dataset, clues_dir=clues_dir,
                                      visual_clues_path=None, textual_clues_path=None)
    template = Path("prompts/query_generation.v1.txt" if stage == "query"
                    else "prompts/query_judgement.v1.txt").read_text(encoding="utf-8")
    requests = []
    for paper in papers:
        selected = select_components(components.get(str(paper["paper_id"]), []), mode="visual-and-text")
        if not selected:
            continue
        if stage == "query":
            prompt = format_query_prompt(template, selected)
            images = ()
            max_tokens = 300
        else:
            prompt = format_judgement_prompt(template, "visual-and-text", paper, selected,
                                             "Which scientific paper matches these clues?")
            images = tuple(paper_image_paths(paper))
            max_tokens = 900
        requests.append(GenerationRequest(str(paper["paper_id"]), prompt, images=images,
                                          max_tokens=max_tokens))
        if len(requests) >= count:
            break
    if not requests:
        raise RuntimeError(f"No usable {stage} examples found in {dataset}")
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=MODELS)
    parser.add_argument("--provider", choices=["vllm", "transformers"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--clues-dir", type=Path, default=Path("data/clues"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, action="append")
    parser.add_argument("--batch-size", type=int, action="append")
    parser.add_argument("--visual-tokens", type=int, action="append")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()
    requests = representative_requests(args.stage, args.dataset, args.clues_dir, args.count)
    seqs = args.max_num_seqs or SEQS[args.stage]
    sizes = args.batch_size or ([1, 4, 8] if args.provider == "transformers" else [max(seqs)])
    budgets = args.visual_tokens or ([512, 768, 1024] if args.stage == "visual" else [None])
    for budget in budgets:
        for seq in (seqs if args.provider == "vllm" else [None]):
            runtime = ({"tensor_parallel_size": 1, "gpu_memory_utilization": args.gpu_memory_utilization,
                        "max_num_seqs": seq, "max_num_batched_tokens": args.max_num_batched_tokens,
                        "dtype": "bfloat16", "kv_cache_dtype": "auto", "enable_prefix_caching": True,
                        "mm_processor_cache_gb": 1} if args.provider == "vllm" else
                       {"device_map": "auto", "dtype": "bfloat16"})
            if args.provider == "vllm" and args.stage in {"visual", "judge"}:
                runtime["limit_mm_per_prompt"] = {"image": 16}
            backend = load_backend(args.provider, args.model or MODELS[args.stage], runtime=runtime,
                                   image={"max_visual_tokens": budget} if budget else {})
            for size in sizes:
                start = time.monotonic()
                failures = tokens = 0
                for batch in batched(requests, size):
                    results = ordered_results(batch, backend.generate(batch))
                    failures += sum(result.error is not None for result in results)
                    tokens += sum(result.generated_tokens or 0 for result in results)
                elapsed = time.monotonic() - start
                print(json.dumps({"stage": args.stage, "model": args.model or MODELS[args.stage],
                                  "provider": args.provider, "max_num_seqs": seq, "batch_size": size,
                                  "max_num_batched_tokens": args.max_num_batched_tokens if seq else None,
                                  "max_visual_tokens": budget, "requests": len(requests),
                                  "failures": failures, "elapsed_seconds": round(elapsed, 3),
                                  "requests_per_second": round((len(requests) - failures) / max(elapsed, 0.001), 3),
                                  "generated_tokens_per_second": round(tokens / max(elapsed, 0.001), 3)}), flush=True)


if __name__ == "__main__":
    main()
