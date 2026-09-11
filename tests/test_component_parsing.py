from __future__ import annotations

from dataset_generation.component_parsing import split_component_text


def test_split_textual_components_from_jsonish_output() -> None:
    text = '[{"category": "method", "cue": "uses contrastive retrieval"}, {"cue": "mentions ACL figures"}]'

    assert split_component_text(text, "textual") == [
        "textual method: uses contrastive retrieval",
        "mentions ACL figures",
    ]


def test_split_visual_components_preserves_preferred_field_order() -> None:
    text = '{"relationships": ["arrows connect modules"], "figure_type": "pipeline"}'

    assert split_component_text(text, "visual") == [
        "figure_type: pipeline",
        "relationships: arrows connect modules",
    ]
