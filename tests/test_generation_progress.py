"""Progress output should be usable as a live stage log."""

from common.generation_utils import progress


def test_progress_prints_each_completed_item_before_iteration_ends(capsys) -> None:
    items = progress(["first", "second"], total=2, enabled=True, description="Judging", unit="query")
    assert next(items) == "first"
    assert next(items) == "second"
    assert "Judging: 1/2 queries (50%)" in capsys.readouterr().out
    list(items)
    assert "Judging: 2/2 queries (100%)" in capsys.readouterr().out
