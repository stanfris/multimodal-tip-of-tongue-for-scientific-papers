"""Chat formatting and generic visual budget conversion."""
from __future__ import annotations

from typing import Any


def chat_prompt(processor: Any, prompt: str, image_count: int = 0, *, thinking: bool = False) -> str:
    if not hasattr(processor, "apply_chat_template"):
        if image_count:
            raise ValueError("The selected multimodal processor has no chat template")
        return prompt
    content: Any = prompt
    if image_count:
        content = [{"type": "image"} for _ in range(image_count)] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def processor_image_kwargs(model: str, image: dict[str, Any]) -> dict[str, int]:
    """Convert generic token budgets only at the model adapter boundary."""
    if not image or not model.lower().startswith("qwen/qwen3-vl"):
        return {}
    # Qwen3-VL uses 16-pixel patches and a 2x2 spatial merge.
    patch_area = (16 * 2) ** 2
    return {pixel_key: int(image[token_key]) * patch_area
            for token_key, pixel_key in (("min_visual_tokens", "min_pixels"),
                                         ("max_visual_tokens", "max_pixels"))
            if image.get(token_key) is not None}
