"""End-to-end tests that spin up a real Jupyter kernel.

These are slower (a kernel startup is ~1-2s) but exercise the full flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

from mcp_jupyter_driver import execution, inspection, registry
from mcp_jupyter_driver.notebook_io import _atomic_write, new_notebook


@pytest.fixture
async def session_factory(tmp_path: Path):
    """Yield a callable that opens a fresh notebook session; closes all on teardown."""
    opened: list[str] = []

    async def _open(seed_cells: list[str] | None = None):
        nb = new_notebook()
        for src in seed_cells or []:
            nb.cells.append(nbformat.v4.new_code_cell(source=src))
        path = tmp_path / f"nb_{len(opened)}.ipynb"
        _atomic_write(path, nb)
        s = await registry.open_session(str(path))
        opened.append(str(s.path))
        return s

    yield _open

    for p in opened:
        try:
            await registry.close_session(p)
        except Exception:
            pass


async def test_run_cell_persists_outputs(session_factory) -> None:
    session = await session_factory(["x = 1 + 2\nx"])
    result = await execution.run_cell(session, 0, timeout_s=15)
    assert result.status == "ok"
    assert result.execution_count == 1
    on_disk = json.loads(session.path.read_text())
    cell = on_disk["cells"][0]
    assert cell["execution_count"] == 1
    assert any(
        out.get("data", {}).get("text/plain") in (["3"], "3")
        for out in cell["outputs"]
    )


async def test_run_cell_error_path(session_factory) -> None:
    session = await session_factory(["1/0"])
    result = await execution.run_cell(session, 0, timeout_s=15)
    assert result.status == "error"
    assert result.error_name == "ZeroDivisionError"
    assert "division by zero" in (result.error_value or "")


async def test_list_variables_filters_mcp_and_dunder(session_factory) -> None:
    session = await session_factory(["import math\nval = 42\nxs = list(range(3))"])
    await execution.run_cell(session, 0, timeout_s=15)
    vars_ = await inspection.list_variables(session)
    names = {v["name"] for v in vars_}
    assert "val" in names and "xs" in names and "math" in names
    assert not any(n.startswith("_mcp_") for n in names)
    assert "get_ipython" not in names  # IPython stoplist
    assert "In" not in names and "Out" not in names


async def test_inspect_variable_pandas_like_shape(session_factory) -> None:
    session = await session_factory(["xs = list(range(10))"])
    await execution.run_cell(session, 0, timeout_s=15)
    detail = await inspection.inspect_variable(session, "xs")
    assert detail["found"] is True
    assert detail["type"] == "list"
    assert detail["length"] == 10


async def test_complete_finds_math_functions(session_factory) -> None:
    session = await session_factory(["import math"])
    await execution.run_cell(session, 0, timeout_s=15)
    result = await inspection.complete(session, "math.s", 6)
    assert any("sin" in m for m in result["matches"])


async def test_edit_cell_clears_outputs(session_factory) -> None:
    session = await session_factory(["1 + 1"])
    await execution.run_cell(session, 0, timeout_s=15)
    assert len(session.nb.cells[0]["outputs"]) > 0
    async with session.exec_lock:
        session.edit_cell(0, "2 + 2")
    assert session.nb.cells[0]["outputs"] == []
    assert session.nb.cells[0]["execution_count"] is None
