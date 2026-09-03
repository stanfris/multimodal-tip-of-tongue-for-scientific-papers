"""Parser adapter for the ACL-fig plus ACL Anthology markdown dataset."""

from __future__ import annotations

from dataset_generation.build import BuildConfig, build_combined_dataset
from dataset_generation.config import ManagedRunConfig
from dataset_generation.parsers.base import ParserResult


class AclFigMarkdownParser:
    name = "acl_fig_markdown"

    def parse(self, config: ManagedRunConfig) -> ParserResult:
        dataset = config.dataset
        build_config = BuildConfig(
            fig_dataset=dataset.figure_dataset,
            fig_split=dataset.figure_split,
            paper_dataset=dataset.paper_dataset,
            paper_config=dataset.paper_config,
            paper_split=dataset.paper_split,
            data_dir=dataset.data_dir,
            prefer_local_sources=dataset.prefer_local_sources,
            streaming=dataset.streaming,
            limit=dataset.limit,
            paper_limit=dataset.paper_limit,
            seed=config.run.seed,
        )
        records, stats, metadata = build_combined_dataset(build_config)
        metadata["parser"] = {
            "name": config.parser.name,
            "options": config.parser.options,
        }
        return ParserResult(records=records, stats=stats, metadata=metadata)
