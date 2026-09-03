# Prompt Registry

Prompt files are versioned inputs to generation runs. Store prompts as plain text and reference them from sidecar artifacts by `prompt_id` and `prompt_version`.

Recommended prompt IDs:

- `visual_interpretation`: describe visible figure content, layout, marks, axes, labels, and visual relationships.
- `textual_interpretation`: summarize the document context around the figure using matched markdown.
- `query_generation`: generate later tip-of-the-tongue queries from the base record plus interpretation sidecars.

Generated outputs should go under `data/interim/interpretations/<run_id>/interpretations.jsonl`, keyed by `record_id`.
