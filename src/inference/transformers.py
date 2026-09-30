"""Batched Transformers reference backend."""
from __future__ import annotations

from typing import Any

from inference.base import GenerationRequest, GenerationResult
from inference.formatting import chat_prompt, processor_image_kwargs


class TransformersBackend:
    def __init__(self, model: str, runtime: dict[str, Any], image: dict[str, Any]) -> None:
        from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                                  AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer)

        self.model_id = model
        self.image = image
        kwargs = {key: runtime[key] for key in ("device_map", "dtype", "attn_implementation")
                  if runtime.get(key) is not None}
        self.visual = "vl" in model.lower() or "gemma-3" in model.lower()
        if self.visual:
            if "gemma-3" in model.lower():
                try:
                    self.model = AutoModelForMultimodalLM.from_pretrained(model, **kwargs)
                except (OSError, ValueError):
                    self.model = AutoModelForImageTextToText.from_pretrained(model, **kwargs)
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(model, **kwargs)
            self.processor = AutoProcessor.from_pretrained(model)
            pixel_limits = processor_image_kwargs(model, image)
            if pixel_limits:
                size = dict(self.processor.image_processor.size)
                if "min_pixels" in pixel_limits:
                    size["shortest_edge"] = pixel_limits["min_pixels"]
                if "max_pixels" in pixel_limits:
                    size["longest_edge"] = pixel_limits["max_pixels"]
                self.processor.image_processor.size = size
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            self.processor = AutoTokenizer.from_pretrained(model)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        # A bad image should not discard unrelated requests in the same batch.
        if any(bool(request.images) for request in requests) != all(bool(request.images) for request in requests):
            raise ValueError("A Transformers batch must be entirely text or entirely multimodal")
        try:
            return self._generate_batch(requests)
        except Exception as exc:
            if len(requests) == 1:
                return [GenerationResult(requests[0].request_id, error=exc)]
            middle = len(requests) // 2
            return self.generate(requests[:middle]) + self.generate(requests[middle:])

    def _generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        from PIL import Image

        # Sampling parameters are shared by a model.generate call. Stages normally
        # use one setting per window; heterogeneous requests fall back to subbatches.
        settings = {(request.max_tokens, request.temperature) for request in requests}
        if len(settings) != 1:
            raise ValueError("A Transformers batch requires uniform sampling settings")
        max_tokens, temperature = settings.pop()
        opened = []
        try:
            if requests[0].images:
                conversations = []
                for request in requests:
                    images = []
                    for path in request.images:
                        with Image.open(path) as source:
                            images.append(source.convert("RGB"))
                    opened.extend(images)
                    content = [{"type": "image", "image": image} for image in images]
                    content.append({"type": "text", "text": request.prompt})
                    conversations.append([{"role": "user", "content": content}])
                inputs = self.processor.apply_chat_template(
                    conversations, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True)
            else:
                prompts = [chat_prompt(self.processor, request.prompt, thinking=request.thinking)
                           for request in requests]
                inputs = self.processor(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(self.model.device)
            generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
            generated = self.model.generate(**inputs, **generate_kwargs)
            prompt_width = inputs.input_ids.shape[1]
            decoded = self.processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            return [GenerationResult(request.request_id, text.strip(),
                                     generated_tokens=len(generated[index]) - prompt_width)
                    for index, (request, text) in enumerate(zip(requests, decoded, strict=True))]
        finally:
            for image in opened:
                image.close()
