"""End-to-end tests via the Jupyter Server architecture.

Each test opens a fresh notebook session against a real `jupyter server`
subprocess. Slower than unit tests, but covers the full path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_jupyter_driver import execution, inspection, registry
from mcp_jupyter_driver.jserver import stop_server


@pytest.fixture(autouse=True)
async def _stop_server_between():
    """Reuse the server within a test, tear down between tests so state stays clean."""
    yield
    try:
        await registry.close_all()
    finally:
        await stop_server()


@pytest.fixture
async def open_nb(tmp_path: Path):
    """Factory that opens a fresh notebook session."""
    opened: list[str] = []

    async def _open(seed_sources: list[str] | None = None):
        path = tmp_path / f"nb_{len(opened)}.ipynb"
        session = await registry.open_session(str(path), create_if_missing=True)
        opened.append(str(session.canonical))
        if seed_sources:
            import nbformat

            nb = await session.read_notebook()
            for src in seed_sources:
                nb["cells"].append(nbformat.v4.new_code_cell(source=src))
            await session.write_notebook(nb)
        return session

    return _open


async def test_run_cell_outputs_persist_via_server(open_nb) -> None:
    session = await open_nb(["x = 1 + 2\nx"])
    result = await execution.run_cell(session, 0, timeout_s=20)
    assert result.status == "ok"
    assert result.execution_count == 1
    nb = await session.read_notebook()
    cell = nb["cells"][0]
    assert cell["execution_count"] == 1
    text_plain = None
    for out in cell["outputs"]:
        tp = (out.get("data") or {}).get("text/plain")
        if tp is not None:
            text_plain = "".join(tp) if isinstance(tp, list) else tp
    assert text_plain == "3"


async def test_error_cell(open_nb) -> None:
    session = await open_nb(["1/0"])
    result = await execution.run_cell(session, 0, timeout_s=20)
    assert result.status == "error"
    assert result.error_name == "ZeroDivisionError"


async def test_variables_persist_after_cell_delete(open_nb) -> None:
    """The headline reason for v3: kernel state survives notebook structural edits."""
    session = await open_nb(["x = 42"])
    await execution.run_cell(session, 0, timeout_s=20)

    # Confirm x is in the kernel
    vars_ = await inspection.list_variables(session)
    assert any(v["name"] == "x" for v in vars_)

    # Now delete the cell that defined x
    async with session.exec_lock:
        nb = await session.read_notebook()
        del nb["cells"][0]
        await session.write_notebook(nb)

    # x should STILL be alive — kernel is decoupled from the notebook
    vars2 = await inspection.list_variables(session)
    assert any(v["name"] == "x" for v in vars2), \
        "variables defined by a cell must survive that cell's deletion"


async def test_list_variables_filters(open_nb) -> None:
    session = await open_nb(["import math\nval = 42\nxs = list(range(3))"])
    await execution.run_cell(session, 0, timeout_s=20)
    vars_ = await inspection.list_variables(session)
    names = {v["name"] for v in vars_}
    assert "val" in names and "xs" in names and "math" in names
    assert not any(n.startswith("_mcp_") for n in names)
    assert "get_ipython" not in names


async def test_complete_via_kernel(open_nb) -> None:
    session = await open_nb(["import math"])
    await execution.run_cell(session, 0, timeout_s=20)
    res = await inspection.complete(session, "math.s", 6)
    assert any("sin" in m for m in res["matches"])


async def test_inspect_variable_pandas_like(open_nb) -> None:
    session = await open_nb(["xs = list(range(10))"])
    await execution.run_cell(session, 0, timeout_s=20)
    detail = await inspection.inspect_variable(session, "xs")
    assert detail["found"] is True
    assert detail["type"] == "list"
    assert detail["length"] == 10


async def test_auto_rejoin_to_user_kernel_via_basename(open_nb) -> None:
    """If a session for the same notebook exists under a different path
    encoding (basename match), Claude should auto-rejoin its kernel on the
    next op so variables are shared.
    """
    session = await open_nb(["x = 'claude_kernel'"])
    await execution.run_cell(session, 0, timeout_s=20)
    kA = session.kernel_id

    # Simulate VS Code creating a session at a different path encoding
    # (basename-only path, which is what Jupyter Server gets when VS Code
    # treats the notebook path as workspace-relative).
    body = {
        "kernel": {"name": "python3"},
        "name": "user.ipynb",
        "path": "user.ipynb",
        "type": "notebook",
    }
    r = await session.client._http.post("/api/sessions", json=body)
    r.raise_for_status()
    sessB = r.json()
    kB = sessB["kernel"]["id"]
    assert kA != kB

    # Define y on the simulated VS Code kernel
    async with session.client.kernel_channel(kB, sessB["id"]) as ch:
        mid = await ch.send(
            "execute_request",
            {
                "code": "y = 'user_kernel'",
                "silent": False, "store_history": True,
                "user_expressions": {}, "allow_stdin": False, "stop_on_error": True,
            },
        )
        while True:
            msg = await ch.recv(timeout=10.0)
            if (msg.get("parent_header") or {}).get("msg_id") != mid:
                continue
            if (msg.get("msg_type") == "status"
                and msg.get("content", {}).get("execution_state") == "idle"):
                break

    # Rejoin only fires if our path's basename matches the simulated session's.
    # Hack it for this test by renaming our session_relative on the fly.
    session.server_relative = "user.ipynb"

    vars_after = await inspection.list_variables(session)
    names = {v["name"] for v in vars_after}
    assert session.kernel_id == kB, "Claude should switch to user's kernel"
    assert "y" in names
    assert "x" not in names
