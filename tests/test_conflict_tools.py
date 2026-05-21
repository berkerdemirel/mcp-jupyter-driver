"""Tool-level conflict-protection tests.

These drive the @mcp.tool() functions in ``server.py`` directly against a
NotebookSession backed by an in-memory stub client, so we exercise the real
``mutate_notebook_fresh`` plumbing without spawning a Jupyter Server.

We also patch ``execution.run_cell`` for the run_code tests so we can
deterministically simulate normal completion, kernel-still-running timeout,
and execution-raised paths to check the finally-cleanup behavior.
"""

from __future__ import annotations

import copy

import pytest

from mcp_jupyter_driver import execution, registry, server
from mcp_jupyter_driver.errors import NotebookConflictError
from mcp_jupyter_driver.session import NotebookSession, canonical_key


class _InMemoryClient:
    """Minimal in-memory Contents+Sessions stub for the tool layer.

    Stores the live notebook on the instance; ``read_notebook`` returns a
    deep copy so tests can mutate intermediate state without leaking back.
    """

    def __init__(self, notebook: dict) -> None:
        self.notebook = notebook
        self.write_count = 0
        self.last_written: dict | None = None
        self.deleted_sessions: list[str] = []
        # Monotonically incremented on every write so tests can simulate
        # last_modified drift against the if_unmodified_since precondition.
        self._version = 0
        self.last_modified = f"v{self._version}"

    async def read_notebook(self, path: str) -> dict:
        return copy.deepcopy(self.notebook)

    async def read_notebook_with_meta(self, path: str) -> tuple[dict, str]:
        # Delegate through ``self.read_notebook`` so tests that patch
        # ``client.read_notebook`` to inject mid-flow mutations also affect
        # the fresh re-read inside ``mutate_notebook_fresh``.
        content = await self.read_notebook(path)
        return content, self.last_modified

    async def get_mtime(self, path: str) -> str:
        return self.last_modified

    async def write_notebook(
        self, path: str, nb: dict, *, if_unmodified_since: str | None = None
    ) -> dict:
        if if_unmodified_since is not None and if_unmodified_since != self.last_modified:
            from mcp_jupyter_driver.errors import ConcurrentWriteError

            raise ConcurrentWriteError(path, if_unmodified_since, self.last_modified)
        self.notebook = nb
        self.last_written = nb
        self.write_count += 1
        self._version += 1
        self.last_modified = f"v{self._version}"
        return {}

    async def list_sessions(self) -> list[dict]:
        return []

    async def get_kernel(self, kernel_id: str) -> dict:
        return {"id": kernel_id, "execution_state": "idle"}

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)


def _seed(nb_cells: list[dict]) -> tuple[NotebookSession, _InMemoryClient, str]:
    """Register a session in the global registry and return (session, client, path)."""
    path = "/test/work.ipynb"
    notebook = {"cells": nb_cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    client = _InMemoryClient(notebook)
    session = NotebookSession(
        canonical=canonical_key(path),
        server_relative="test/work.ipynb",
        session_id="claude-sess",
        kernel_id="claude-kernel",
        kernel_name="python3",
        client=client,  # type: ignore[arg-type]
    )
    registry._sessions[session.canonical] = session
    return session, client, path


@pytest.fixture(autouse=True)
def _clear_registry():
    """Each test starts with an empty registry so it can install its own session."""
    registry._sessions.clear()
    yield
    registry._sessions.clear()


def _code_cell(cid: str, source: str = "", outputs=None, exec_count=None) -> dict:
    return {
        "id": cid,
        "cell_type": "code",
        "source": source,
        "outputs": list(outputs or []),
        "execution_count": exec_count,
    }


# ---------------------------------------------------------------------------
# edit_cell: conflict detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_cell_preserves_concurrent_user_edit_to_different_cell() -> None:
    """Claude edits cell B; meanwhile the user edits cell A in VS Code. Both
    edits should persist — Claude's edit must not revert A.
    """
    session, client, path = _seed([
        _code_cell("a", "x = 1"),
        _code_cell("b", "y = 1"),
    ])
    # Simulate VS Code editing cell A right before our mutator runs.
    client.notebook["cells"][0]["source"] = "x = 99  # user"

    await server.edit_cell(path, ref="b", source="y = 42")

    sources = {c["id"]: c["source"] for c in client.notebook["cells"]}
    assert sources == {"a": "x = 99  # user", "b": "y = 42"}


@pytest.mark.asyncio
async def test_edit_cell_raises_when_target_deleted_concurrently() -> None:
    """Claude resolves edit_cell(b), but the user deletes cell b from VS Code
    before the mutator runs. Must raise NotebookConflictError, not silently
    edit cell at the same index.
    """
    session, client, path = _seed([
        _code_cell("a", "x = 1"),
        _code_cell("b", "y = 1"),
    ])

    # First read happens inside edit_cell to resolve ref → cell_id. We hook
    # into the client to mutate state between read and the helper's re-read.
    real_read = client.read_notebook
    call_count = {"n": 0}

    async def _intercept(p: str) -> dict:
        call_count["n"] += 1
        # After the *first* read (the one that resolves ref → cell_id),
        # the user deletes cell b. The helper's second read will not find it.
        if call_count["n"] == 1:
            result = await real_read(p)
            # Mutate underlying state to drop cell b.
            client.notebook["cells"] = [c for c in client.notebook["cells"] if c["id"] != "b"]
            return result
        return await real_read(p)

    client.read_notebook = _intercept  # type: ignore[assignment]

    with pytest.raises(NotebookConflictError):
        await server.edit_cell(path, ref="b", source="y = 42")

    # And the surviving cell A must be untouched.
    cells = client.notebook["cells"]
    assert [c["id"] for c in cells] == ["a"]
    assert cells[0]["source"] == "x = 1"


# ---------------------------------------------------------------------------
# add_cell: concurrent user insertion preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_cell_preserves_user_inserted_cell() -> None:
    """User added a cell from VS Code; Claude calls add_cell. The new cell
    should land at the end of the *fresh* state, with the user's cell intact.
    """
    session, client, path = _seed([_code_cell("a", "x = 1")])
    # User adds a cell from VS Code.
    client.notebook["cells"].append(_code_cell("user-new", "u = 99"))

    await server.add_cell(path, cell_type="code", source="claude_added = True")

    ids = [c["id"] for c in client.notebook["cells"]]
    assert ids[:2] == ["a", "user-new"]
    assert len(ids) == 3
    assert client.notebook["cells"][-1]["source"] == "claude_added = True"


# ---------------------------------------------------------------------------
# clear_cell_outputs (all): concurrent source edit preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_all_outputs_does_not_revert_concurrent_source_edit() -> None:
    session, client, path = _seed([
        _code_cell(
            "a", "x = 1",
            outputs=[{"output_type": "stream", "name": "stdout", "text": "1"}],
            exec_count=1,
        ),
    ])
    # User edits source from VS Code.
    client.notebook["cells"][0]["source"] = "x = 2  # user"

    await server.clear_cell_outputs(path, ref=None)

    cell = client.notebook["cells"][0]
    assert cell["source"] == "x = 2  # user"
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


# ---------------------------------------------------------------------------
# delete_cell + move_cell: conflict on missing target
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_cell_raises_when_target_already_gone() -> None:
    session, client, path = _seed([_code_cell("a", "x = 1"), _code_cell("b", "y = 1")])

    real_read = client.read_notebook
    call_count = {"n": 0}

    async def _intercept(p: str) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            result = await real_read(p)
            client.notebook["cells"] = [c for c in client.notebook["cells"] if c["id"] != "b"]
            return result
        return await real_read(p)

    client.read_notebook = _intercept  # type: ignore[assignment]

    with pytest.raises(NotebookConflictError):
        await server.delete_cell(path, ref="b")
    # Cell A untouched.
    assert [c["id"] for c in client.notebook["cells"]] == ["a"]


# ---------------------------------------------------------------------------
# run_code: finally cleanup behavior
# ---------------------------------------------------------------------------


class _DummyCtx:
    async def report_progress(self, *args, **kwargs) -> None:
        return None


@pytest.mark.asyncio
async def test_run_code_deletes_temp_cell_on_normal_completion(monkeypatch) -> None:
    session, client, path = _seed([_code_cell("a", "x = 1")])

    async def _fake_run_cell(session, ref, *, timeout_s=120.0, progress=None,
                             restart_on_kernel_death=False):
        return execution.CellResult(status="ok", execution_count=42, output_count=0)

    monkeypatch.setattr(execution, "run_cell", _fake_run_cell)

    res = await server.run_code(path, source="print('hi')", ctx=_DummyCtx(),
                                persist_as_cell=False, timeout_s=5.0)
    assert res.persisted is False
    # Only the original cell should remain.
    assert [c["id"] for c in client.notebook["cells"]] == ["a"]


@pytest.mark.asyncio
async def test_run_code_keeps_temp_cell_when_kernel_still_running(monkeypatch) -> None:
    """If the cell timed out and the kernel is still executing, deleting the
    cell would lose the follow-up outputs the kernel is about to emit.
    """
    session, client, path = _seed([_code_cell("a", "x = 1")])

    async def _fake_run_cell(session, ref, *, timeout_s=120.0, progress=None,
                             restart_on_kernel_death=False):
        return execution.CellResult(
            status="error", execution_count=None,
            kernel_still_running=True,
            error_name="Timeout", error_value="cell exceeded timeout_s",
        )

    monkeypatch.setattr(execution, "run_cell", _fake_run_cell)

    res = await server.run_code(path, source="while True: pass", ctx=_DummyCtx(),
                                persist_as_cell=False, timeout_s=1.0)
    assert res.persisted is True, "kernel_still_running must force-persist the temp cell"
    assert res.run.kernel_still_running is True
    # Both the original and the temp cell should still be on disk.
    assert len(client.notebook["cells"]) == 2


@pytest.mark.asyncio
async def test_recent_user_activity_reports_cell_changes_since_first_call() -> None:
    """First call seeds the snapshot (no changes reported). Then the user
    inserts and edits cells; a second call surfaces the diff. The iopub
    tap is unavailable here (stub client has no kernel_channel) so
    executions stay empty and the response carries a note explaining why.
    """
    session, client, path = _seed([_code_cell("a", "x = 1")])
    # Mark the tap as absent so collect_user_activity hits the "no signal"
    # branch — the stub client doesn't implement kernel_channel.
    session.iopub_tap = None

    first = await server.recent_user_activity(path)
    assert first.cell_changes == []
    assert "iopub tap unavailable" in first.note

    # User inserts a new cell and edits the original from VS Code.
    client.notebook["cells"].append(_code_cell("b", "y = 2"))
    client.notebook["cells"][0]["source"] = "x = 99"

    second = await server.recent_user_activity(path)
    kinds = {(c.kind, c.cell_id) for c in second.cell_changes}
    assert ("added", "b") in kinds
    assert ("edited", "a") in kinds

    # A third immediate call sees no further changes — the second call
    # advanced the last-seen snapshot.
    third = await server.recent_user_activity(path)
    assert third.cell_changes == []


@pytest.mark.asyncio
async def test_run_code_keeps_temp_cell_when_run_raises(monkeypatch) -> None:
    """If the execution path raises (kernel died, websocket error, ...),
    the finally block should still attempt cleanup but skip the delete —
    keeping the cell visible so the user can see what happened.
    """
    session, client, path = _seed([_code_cell("a", "x = 1")])

    async def _fake_run_cell(session, ref, *, timeout_s=120.0, progress=None,
                             restart_on_kernel_death=False):
        raise RuntimeError("kernel died mid-execution")

    monkeypatch.setattr(execution, "run_cell", _fake_run_cell)

    with pytest.raises(RuntimeError):
        await server.run_code(path, source="x", ctx=_DummyCtx(),
                              persist_as_cell=False, timeout_s=5.0)
    # Temp cell kept so the user can see the half-finished state.
    assert len(client.notebook["cells"]) == 2
