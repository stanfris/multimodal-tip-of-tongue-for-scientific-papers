from __future__ import annotations

from dataset_generation.query_generation import MemoryComponent, format_query_prompt


def test_format_query_prompt_groups_visual_clues_by_figure_once() -> None:
    selected = [
        MemoryComponent(record_id="fig1", kind="visual", text="Figure 1: figure_type: pipeline diagram"),
        MemoryComponent(record_id="fig1", kind="visual", text="Figure 1: relationships: arrows connect modules"),
        MemoryComponent(record_id="fig2", kind="visual", text="Figure 2: layout: two-panel plot"),
    ]

    assert format_query_prompt("{selected_cues}", selected) == "\n".join(
        [
            "Figure 1",
            "- pipeline diagram",
            "- arrows connect modules",
            "",
            "Figure 2",
            "- two-panel plot",
        ]
    )


def test_format_query_prompt_strips_textual_categories() -> None:
    selected = [
        MemoryComponent(record_id="paper1", kind="textual", text="textual method: uses contrastive retrieval"),
        MemoryComponent(record_id="paper1", kind="textual", text="mentions ACL figures"),
        MemoryComponent(record_id="fig1", kind="visual", text="Figure 1: colors_and_styles: blue and gray blocks"),
    ]

    assert format_query_prompt("MEMORY CUES:\n{selected_cues}", selected) == "\n".join(
        [
            "MEMORY CUES:",
            "- uses contrastive retrieval",
            "- mentions ACL figures",
            "",
            "Figure 1",
            "- blue and gray blocks",
        ]
    )
