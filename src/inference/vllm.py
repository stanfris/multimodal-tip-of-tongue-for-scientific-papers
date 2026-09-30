"""Offline vLLM backend. A request window enters one engine call."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.base import GenerationRequest, GenerationResult, ordered_results
from inference.formatting import chat_prompt, processor_image_kwargs


class VLLMBackend:
    def __init__(self, model: str, runtime: dict[str, Any], image: dict[str, Any]) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM backend requested but vllm is not installed. Install it in a supported GPU environment.") from exc
        from transformers import AutoProcessor, AutoTokenizer

        self.sampling_params = SamplingParams
        kwargs = {key: value for key, value in runtime.items() if value is not None}
        if ("vl" in model.lower() or "gemma-3" in model.lower()) and "limit_mm_per_prompt" not in kwargs:
            kwargs["limit_mm_per_prompt"] = {"image": 16}
        mm_kwargs = processor_image_kwargs(model, image)
        if mm_kwargs:
            kwargs["mm_processor_kwargs"] = {**kwargs.get("mm_processor_kwargs", {}), **mm_kwargs}
        self.llm = LLM(model=model, **kwargs)
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            if not hasattr(self.processor, "apply_chat_template"):
                self.processor = AutoTokenizer.from_pretrained(model)
        except (OSError, ValueError):
            self.processor = AutoTokenizer.from_pretrained(model)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        try:
            return self._generate_batch(requests)
        except Exception as exc:
            if len(requests) == 1:
                return [GenerationResult(requests[0].request_id, error=exc)]
            middle = len(requests) // 2
            return self.generate(requests[:middle]) + self.generate(requests[middle:])

    def _generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        from PIL import Image

        prompts: list[dict[str, Any]] = []
        params = []
        opened = []
        try:
            for request in requests:
                prompt = chat_prompt(self.processor, request.prompt, len(request.images), thinking=request.thinking)
                item: dict[str, Any] = {"prompt": prompt}
                if request.images:
                    images = []
                    for path in request.images:
                        with Image.open(Path(path)) as source:
                            images.append(source.convert("RGB"))
                    opened.extend(images)
                    item["multi_modal_data"] = {"image": images[0] if len(images) == 1 else images}
                prompts.append(item)
                params.append(self.sampling_params(max_tokens=request.max_tokens, temperature=request.temperature))
            outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
            results = [GenerationResult(request_id=request.request_id,
                                        text=output.outputs[0].text.strip(),
                                        generated_tokens=len(output.outputs[0].token_ids))
                       for request, output in zip(requests, outputs, strict=True)]
            return ordered_results(requests, results)
        finally:
            for image in opened:
                image.close()
