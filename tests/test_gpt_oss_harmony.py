from __future__ import annotations

from typing import Any

from dataset_generation.textual_clue_descriptions import extract_gpt_oss_final_text, generate_with_transformers


class FakeInputIds:
    shape = (1, 3)


class FakeInputs(dict[str, Any]):
    input_ids = FakeInputIds()

    def __init__(self) -> None:
        super().__init__({"input_ids": self.input_ids})

    def to(self, device: str) -> "FakeInputs":
        return self


class FakeGeneratedIds:
    def __getitem__(self, key: object) -> "FakeGeneratedIds":
        return self


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.max_new_tokens: list[int] = []

    def generate(self, **kwargs: Any) -> FakeGeneratedIds:
        self.max_new_tokens.append(kwargs["max_new_tokens"])
        return FakeGeneratedIds()


class FakeTokenizer:
    def __init__(self, decoded: list[str]) -> None:
        self.decoded = decoded
        self.skip_special_tokens: list[bool] = []
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.messages = messages
        return "formatted prompt"

    def __call__(self, prompt: str, *, return_tensors: str) -> FakeInputs:
        return FakeInputs()

    def batch_decode(self, generated_ids: FakeGeneratedIds, *, skip_special_tokens: bool) -> list[str]:
        self.skip_special_tokens.append(skip_special_tokens)
        return [self.decoded.pop(0)]


def test_extract_gpt_oss_final_text_discards_analysis_and_trailing_token() -> None:
    output = (
        "<|channel|>analysis<|message|>hidden reasoning"
        "<|channel|>final<|message|>Find the paper with a blue pipeline diagram.<|return|>"
    )

    assert extract_gpt_oss_final_text(output) == "Find the paper with a blue pipeline diagram."


def test_gpt_oss_generation_retries_when_final_channel_is_missing() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer(
        [
            "<|channel|>analysis<|message|>still thinking",
            "<|channel|>final<|message|>Final query text.<|end|>",
        ]
    )

    text = generate_with_transformers(
        loaded={"model": model, "tokenizer": tokenizer, "model_id": "openai/gpt-oss-20b"},
        prompt="Write a query.",
        max_tokens=220,
        temperature=0.0,
    )

    assert text == "Final query text."
    assert model.max_new_tokens == [220, 512]
    assert tokenizer.skip_special_tokens == [False, False]
    assert tokenizer.messages is not None
    assert tokenizer.messages[0] == {"role": "system", "content": "Reasoning: low"}
