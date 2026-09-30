from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference import GenerationRequest, GenerationResult, load_backend
from inference.base import ordered_results
from inference.formatting import processor_image_kwargs
from inference.vllm import VLLMBackend


def test_request_order_and_ids() -> None:
    requests = [GenerationRequest("a", "first"), GenerationRequest("b", "second")]
    assert [result.text for result in ordered_results(requests, [GenerationResult("b", "B"),
                                                       GenerationResult("a", "A")])] == ["A", "B"]
    with pytest.raises(ValueError, match="missing"):
        ordered_results(requests, [GenerationResult("a", "A")])


def test_vllm_text_and_multimodal_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            calls.append((prompts, sampling_params, use_tqdm))
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"reply-{index}", token_ids=[1, 2])])
                    for index in range(len(prompts))]

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return str(messages)

    class FakeAuto:
        @staticmethod
        def from_pretrained(model):
            return FakeProcessor()

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM,
                       SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=FakeAuto,
                       AutoTokenizer=FakeAuto))
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (8, 8)).save(image_path)
    backend = VLLMBackend("Qwen/Qwen3-VL-4B-Instruct", {"max_num_seqs": 64},
                          {"max_visual_tokens": 768})
    requests = [GenerationRequest("text", "explain"),
                GenerationRequest("image", "describe", images=(image_path,))]
    results = backend.generate(requests)
    assert [result.request_id for result in results] == ["text", "image"]
    assert len(calls) == 1 and len(calls[0][0]) == 2
    assert "multi_modal_data" not in calls[0][0][0]
    assert calls[0][0][1]["multi_modal_data"]["image"].size == (8, 8)
    assert backend.llm.kwargs["mm_processor_kwargs"]["max_pixels"] == 768 * 32 * 32
    assert processor_image_kwargs("google/gemma-3-27b-it", {"max_visual_tokens": 768}) == {}


def test_missing_vllm_is_lazy_and_helpful(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(RuntimeError, match="vllm is not installed"):
        load_backend("vllm", "Qwen/Qwen3-4B")


def test_vllm_isolates_one_failed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackend(VLLMBackend):
        def _generate_batch(self, requests):
            if any(request.request_id == "bad" for request in requests):
                raise ValueError("bad prompt")
            return [GenerationResult(request.request_id, "ok") for request in requests]

    backend = FakeBackend.__new__(FakeBackend)
    results = backend.generate([GenerationRequest("one", "a"), GenerationRequest("bad", "b"),
                                GenerationRequest("three", "c")])
    assert [result.error is None for result in results] == [True, False, True]


@pytest.mark.parametrize("with_images", [False, True])
def test_transformers_submits_one_padded_batch(tmp_path: Path, with_images: bool) -> None:
    import torch
    from PIL import Image
    from inference.transformers import TransformersBackend

    calls = []

    class Batch(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("tokenize") is True:
                calls.append((messages, kwargs))
                return Batch(input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]))
            return "chat prompt"

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Batch(input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]))

        def batch_decode(self, ids, **kwargs):
            return [f"decoded-{int(row[0])}" for row in ids]

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return torch.cat((kwargs["input_ids"], torch.tensor([[7], [8]])), dim=1)

    backend = TransformersBackend.__new__(TransformersBackend)
    backend.model = Model()
    backend.processor = Processor()
    backend.model_id = "Qwen/Qwen3-VL-4B-Instruct" if with_images else "Qwen/Qwen3-4B"
    backend.image = {}
    if with_images:
        image_path = tmp_path / "figure.png"
        Image.new("RGB", (8, 8)).save(image_path)
        images = (image_path,)
    else:
        images = ()
    results = backend.generate([GenerationRequest("a", "first", images=images),
                                GenerationRequest("b", "second", images=images)])
    assert [result.text for result in results] == ["decoded-7", "decoded-8"]
    assert sum(call[0] == "generate" for call in calls if isinstance(call[0], str)) == 1
    assert calls[0][1]["padding"] is True
