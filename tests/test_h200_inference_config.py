from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hydra_run import managed_settings


ROOT = Path(__file__).resolve().parents[1]


def settings_for(stage: str, *overrides: str) -> dict:
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="config", overrides=[f"stage={stage}", "launcher=pbs_h200",
                                                           "model.provider=vllm", *overrides])
    resolved = OmegaConf.to_container(config, resolve=True)
    return managed_settings(resolved, ROOT / "data", ROOT)


def test_h200_model_priors_and_visual_limit() -> None:
    stages = (("describe_figures", "visual_descriptions", 64),
              ("describe_textual_clues", "textual_descriptions", 256),
              ("generate_queries", "visual_query", 128))
    for stage, section, expected in stages:
        settings = settings_for(stage)
        assert settings[section]["runtime"]["vllm"]["max_num_seqs"] == expected
        assert settings[section]["runtime"]["vllm"]["tensor_parallel_size"] == 1
    judge = settings_for("judge_queries")["visual_query"]["judgement"]
    assert judge["runtime"]["vllm"]["max_num_seqs"] == 64
    assert judge["runtime"]["vllm"]["dtype"] == "bfloat16"
    assert settings_for("describe_figures")["visual_descriptions"]["image"]["max_visual_tokens"] == 768


def test_h200_overrides() -> None:
    settings = settings_for("describe_figures", "launcher.vllm.max_num_seqs=32",
                            "launcher.vllm.max_num_batched_tokens=8192",
                            "visual_descriptions.image.max_visual_tokens=1024")
    visual = settings["visual_descriptions"]
    assert visual["runtime"]["vllm"]["max_num_seqs"] == 32
    assert visual["runtime"]["vllm"]["max_num_batched_tokens"] == 8192
    assert visual["image"]["max_visual_tokens"] == 1024
