"""Unit tests for the notebook-state diff helper.

We don't need a kernel here — ``diff_notebooks`` is a pure function over
nbformat dicts. The combined ``collect_user_activity`` is exercised at
the tool layer in integration tests.
"""

from __future__ import annotations

from mcp_jupyter_driver.awareness import diff_notebooks


def _cell(cid: str, source: str = "", cell_type: str = "code") -> dict:
    return {"id": cid, "cell_type": cell_type, "source": source, "outputs": [], "execution_count": None}


def _nb(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_first_sight_returns_empty() -> None:
    """Diffing against None (no prior snapshot) reports no changes — the
    first call to ``recent_user_activity`` shouldn't claim a brand-new
    notebook is one big add-storm."""
    assert diff_notebooks(None, _nb([_cell("a")])) == []


def test_no_change_returns_empty() -> None:
    nb = _nb([_cell("a", "x = 1")])
    assert diff_notebooks(nb, nb) == []


def test_added_cell_reported() -> None:
    old = _nb([_cell("a", "x = 1")])
    new = _nb([_cell("a", "x = 1"), _cell("b", "y = 2")])
    changes = diff_notebooks(old, new)
    assert len(changes) == 1
    c = changes[0]
    assert c.kind == "added"
    assert c.cell_id == "b"
    assert c.index == 1
    assert "y = 2" in c.source_preview


def test_removed_cell_reports_last_known_index() -> None:
    old = _nb([_cell("a", "x = 1"), _cell("b", "y = 2")])
    new = _nb([_cell("a", "x = 1")])
    changes = diff_notebooks(old, new)
    assert len(changes) == 1
    c = changes[0]
    assert c.kind == "removed"
    assert c.cell_id == "b"
    assert c.index == 1


def test_edited_cell_carries_before_after_previews() -> None:
    old = _nb([_cell("a", "x = 1")])
    new = _nb([_cell("a", "x = 99")])
    changes = diff_notebooks(old, new)
    assert len(changes) == 1
    c = changes[0]
    assert c.kind == "edited"
    assert c.cell_id == "a"
    assert "x = 1" in c.old_source_preview
    assert "x = 99" in c.new_source_preview


def test_pure_reorder_reports_move_not_edit() -> None:
    old = _nb([_cell("a", "x = 1"), _cell("b", "y = 2")])
    new = _nb([_cell("b", "y = 2"), _cell("a", "x = 1")])
    changes = diff_notebooks(old, new)
    assert sorted((c.kind, c.cell_id) for c in changes) == [
        ("moved", "a"),
        ("moved", "b"),
    ]


def test_edit_and_reorder_reports_edit_only() -> None:
    """If source changed, prefer the edit signal over a misleading move."""
    old = _nb([_cell("a", "x = 1"), _cell("b", "y = 2")])
    new = _nb([_cell("b", "y = 2"), _cell("a", "x = 99")])
    changes = diff_notebooks(old, new)
    kinds = {(c.kind, c.cell_id) for c in changes}
    assert ("edited", "a") in kinds
    # b only moved, so:
    assert ("moved", "b") in kinds


def test_output_change_alone_does_not_report_an_edit() -> None:
    """A cell that got run shouldn't show up as an edit — its source is
    unchanged; only outputs/execution_count moved."""
    old = _nb([_cell("a", "x = 1")])
    new_cells = [_cell("a", "x = 1")]
    new_cells[0]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "1\n"}]
    new_cells[0]["execution_count"] = 5
    new = _nb(new_cells)
    assert diff_notebooks(old, new) == []


def test_cells_without_ids_are_skipped() -> None:
    """Pre-4.5 notebooks may have id-less cells. We ignore them rather than
    raise — the worst case is reduced awareness, not a crash."""
    old = _nb([{"cell_type": "code", "source": "x = 1", "outputs": [], "execution_count": None}])
    new = _nb([{"cell_type": "code", "source": "x = 2", "outputs": [], "execution_count": None}])
    assert diff_notebooks(old, new) == []
