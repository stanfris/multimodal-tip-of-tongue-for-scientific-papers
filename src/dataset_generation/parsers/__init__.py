"""Parser registry for managed dataset generation runs."""

from dataset_generation.parsers.acl_fig_markdown import AclFigMarkdownParser
from dataset_generation.parsers.base import DatasetParser, ParserResult


PARSERS: dict[str, type[DatasetParser]] = {
    AclFigMarkdownParser.name: AclFigMarkdownParser,
}


def get_parser(name: str) -> type[DatasetParser]:
    try:
        return PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown parser {name!r}; expected one of {sorted(PARSERS)}") from exc


__all__ = ["DatasetParser", "ParserResult", "get_parser"]
