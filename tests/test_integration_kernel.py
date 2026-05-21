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


async def test_auto_rejoin_to_user_kernel_via_vscode_synthetic_path(open_nb) -> None:
    """VS Code's Jupyter extension creates sessions with a synthetic path:
    `<stem>-jvsc-<uuid>-<uuid>.ipynb`. Auto-rejoin should still find it.
    """
    session = await open_nb(["x = 'claude'"])
    await execution.run_cell(session, 0, timeout_s=20)
    kA = session.kernel_id

    body = {
        "kernel": {"name": "python3"},
        "name": "vscode-mock",
        "path": "userwork-jvsc-aaaa1111-bbbb2222.ipynb",
        "type": "notebook",
    }
    r = await session.client._http.post("/api/sessions", json=body)
    r.raise_for_status()
    sessB = r.json()
    kB = sessB["kernel"]["id"]
    assert kA != kB

    async with session.client.kernel_channel(kB, sessB["id"]) as ch:
        mid = await ch.send(
            "execute_request",
            {
                "code": "y = 'vscode'",
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

    # The matcher uses our notebook's stem. Force it to "userwork" so the
    # prefix `userwork-jvsc-` matches the simulated VS Code session.
    session.server_relative = "userwork.ipynb"

    vars_after = await inspection.list_variables(session)
    names = {v["name"] for v in vars_after}
    assert session.kernel_id == kB, "Claude should switch to user's kernel"
    assert "y" in names and "x" not in names


async def test_rebind_kernel_sticks_across_operations(open_nb) -> None:
    """After explicit rebind_kernel, the next operation must NOT auto-rejoin
    back to the original kernel. (Regression for the snap-back bug.)
    """
    session = await open_nb(["x = 'claude'"])
    await execution.run_cell(session, 0, timeout_s=20)
    kA = session.kernel_id

    body = {
        "kernel": {"name": "python3"},
        "name": "user-mock",
        "path": "user-mock.ipynb",
        "type": "notebook",
    }
    r = await session.client._http.post("/api/sessions", json=body)
    r.raise_for_status()
    sessB = r.json()
    kB = sessB["kernel"]["id"]
    assert kA != kB

    # Explicit rebind to B
    async with session.exec_lock:
        outcome = await session.rebind_to_kernel(kB)
    assert outcome.ok is True
    assert session.kernel_id == kB
    assert session.pinned is True

    # Run something through Claude — auto-rejoin would normally snap us
    # back to the path-matched session A. With the pin set, it must not.
    async with session.exec_lock:
        nb = await session.read_notebook()
        import nbformat
        nb["cells"].append(nbformat.v4.new_code_cell(source="z = 'after_rebind'"))
        await session.write_notebook(nb)
    await execution.run_cell(session, len(nb["cells"]) - 1, timeout_s=20)

    assert session.kernel_id == kB, "rebind must stick across operations"

    # And unpin allows auto-rejoin to reconsider
    session.unpin()
    # Force-rename basename so auto-rejoin matches user-mock.ipynb. Without
    # the pin, our current binding is to B (alive) so we stay there.
    # If we instead point at a stale id, rejoin should find B again.


async def test_close_notebook_does_not_kill_unowned_session(open_nb) -> None:
    """If auto-rejoin or rebind has switched us onto a VS Code-owned session,
    close_notebook must not DELETE it via the server API — that would shut
    down the user's kernel.
    """
    session = await open_nb(["x = 1"])
    await execution.run_cell(session, 0, timeout_s=20)

    body = {
        "kernel": {"name": "python3"},
        "name": "vscode-mock-owned",
        "path": "vscode-mock-owned.ipynb",
        "type": "notebook",
    }
    r = await session.client._http.post("/api/sessions", json=body)
    r.raise_for_status()
    user_sess = r.json()

    # Simulate auto-rejoin / rebind landing on the user-owned session.
    session.session_id = user_sess["id"]
    session.kernel_id = user_sess["kernel"]["id"]

    await registry.close_session(session.canonical, shutdown_kernel=True)

    # The user-owned session must still be alive on the server.
    r2 = await session.client._http.get("/api/sessions")
    r2.raise_for_status()
    alive_ids = {s["id"] for s in r2.json()}
    assert user_sess["id"] in alive_ids


async def test_restart_kernel_clears_pin(open_nb) -> None:
    """The restart_kernel MCP tool must release the pin so auto-rejoin can
    move again. Tests the actual tool path, not a manual emulation.
    """
    from mcp_jupyter_driver.server import restart_kernel
    session = await open_nb(["x = 1"])
    await execution.run_cell(session, 0, timeout_s=20)
    outcome = await session.rebind_to_kernel(session.kernel_id)
    assert outcome.ok is True
    assert session.pinned is True

    # @mcp.tool() registers the function but returns it unchanged, so we
    # invoke the real tool body. The pin state lives on the registry session.
    handle = await restart_kernel(session.canonical, clear_outputs=False)
    assert handle.kernel_id
    assert session.pinned is False


async def test_iopub_tap_captures_user_side_execution(open_nb) -> None:
    """When the user runs a cell from VS Code's side, the iopub tap should
    record the execution and tag it ``by_claude=False`` so Claude can pick
    up the user's activity without polling.
    """
    import asyncio

    session = await open_nb(["x = 'claude'"])
    # First Claude-side run so we have at least one ``by_claude=True``
    # execution to compare against.
    await execution.run_cell(session, 0, timeout_s=20)

    # Give the long-lived iopub tap a moment to settle on the kernel.
    assert session.iopub_tap is not None, "tap should be auto-started on open"
    await asyncio.sleep(0.5)

    # Simulate VS Code by sending execute_request on Claude's kernel but
    # with a different message-protocol session id in the header. The
    # Jupyter kernel echoes ``session`` in parent_header, which is the
    # signal the tap uses to attribute the run. Posting a new HTTP session
    # for the same path is no good — jupyter-server deduplicates by path
    # and would hand back Claude's existing session id.
    user_session_id = "vscode-mock-session-deadbeef"
    assert user_session_id != session.session_id

    async with session.client.kernel_channel(session.kernel_id, user_session_id) as ch:
        mid = await ch.send(
            "execute_request",
            {
                "code": "user_var = 'from_vscode'",
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
        )
        # Drive the side channel until idle so the iopub broadcast has flushed.
        while True:
            msg = await ch.recv(timeout=10.0)
            if (msg.get("parent_header") or {}).get("msg_id") != mid:
                continue
            if (
                msg.get("msg_type") == "status"
                and msg.get("content", {}).get("execution_state") == "idle"
            ):
                break

    # The tap runs in the background; give it a beat to catch up.
    await asyncio.sleep(0.3)

    execs = session.iopub_tap.recent_executions()
    user_runs = [e for e in execs if not e.by_claude]
    assert any(
        (e.code or "").strip().startswith("user_var = 'from_vscode'")
        for e in user_runs
    ), f"expected to find user_var run in tap. got: {[(e.code, e.by_claude) for e in execs]}"


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
