"""Unit tests for NotebookSession behavior that doesn't require a real kernel.

We exercise maybe_rejoin's candidate-ranking logic and rebind validation
against a stub JupyterClient. This keeps the dangerous collaboration paths
covered even on hosts where Jupyter Server can't start (containers running
as root without --allow-root, hosts without ipykernel, CI).
"""

from __future__ import annotations

import pytest

from mcp_jupyter_driver.errors import CellNotFoundError, NotebookConflictError
from mcp_jupyter_driver.session import (
    NotebookSession,
    RebindOutcome,
    is_vscode_synthetic_path,
)


class _StubClient:
    def __init__(self, sessions: list[dict], alive: set[str] | None = None) -> None:
        self._sessions = sessions
        self._alive = alive if alive is not None else {s["kernel"]["id"] for s in sessions}

    async def list_sessions(self) -> list[dict]:
        return list(self._sessions)

    async def get_kernel(self, kernel_id: str) -> dict:
        if kernel_id not in self._alive:
            raise RuntimeError("dead")
        return {"id": kernel_id, "execution_state": "idle"}


def _make_session(
    *,
    server_relative: str = "project_a/analysis.ipynb",
    session_id: str = "claude-sess",
    kernel_id: str = "claude-kernel",
    client: _StubClient | None = None,
) -> NotebookSession:
    return NotebookSession(
        canonical="/project_a/analysis.ipynb",
        server_relative=server_relative,
        session_id=session_id,
        kernel_id=kernel_id,
        kernel_name="python3",
        client=client,  # type: ignore[arg-type]
    )


def _sess(sid: str, kid: str, path: str, name: str = "python3") -> dict:
    return {"id": sid, "kernel": {"id": kid, "name": name}, "path": path}


# ---------------------------------------------------------------------------
# maybe_rejoin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_rejoin_does_not_hijack_same_basename_in_other_dir() -> None:
    """Two notebooks share basename `analysis.ipynb` across directories.
    Claude is bound to project_a/analysis.ipynb and that session is alive.
    A user session for project_b/analysis.ipynb must NOT be picked up.
    """
    client = _StubClient(
        sessions=[
            _sess("a-sess", "claude-kernel", "project_a/analysis.ipynb"),
            _sess("b-sess", "other-kernel", "project_b/analysis.ipynb"),
        ]
    )
    session = _make_session(client=client)
    rejoined = await session.maybe_rejoin()
    assert rejoined is False
    assert session.kernel_id == "claude-kernel"


@pytest.mark.asyncio
async def test_maybe_rejoin_prefers_non_ours_at_same_exact_path() -> None:
    """When VS Code opens the exact same path with a different kernel,
    auto-rejoin should hop onto it — that's the co-editing intent.
    """
    client = _StubClient(
        sessions=[
            _sess("a-sess", "claude-kernel", "project_a/analysis.ipynb"),
            _sess("vsc-sess", "vsc-kernel", "project_a/analysis.ipynb"),
        ]
    )
    session = _make_session(client=client)
    rejoined = await session.maybe_rejoin()
    assert rejoined is True
    assert session.kernel_id == "vsc-kernel"
    assert session.session_id == "vsc-sess"


@pytest.mark.asyncio
async def test_maybe_rejoin_synthetic_vscode_path() -> None:
    """VS Code synthetic path matches when there's no exact-path candidate."""
    client = _StubClient(
        sessions=[
            _sess("vsc-sess", "vsc-kernel", "analysis-jvsc-aaa-bbb.ipynb"),
        ]
    )
    session = _make_session(
        server_relative="analysis.ipynb",
        kernel_id="claude-kernel",
        client=client,
    )
    # Our own kernel doesn't have a server-side session in this stub. Make
    # it "dead" so we don't insist on staying.
    client._alive = {"vsc-kernel"}
    rejoined = await session.maybe_rejoin()
    assert rejoined is True
    assert session.kernel_id == "vsc-kernel"


@pytest.mark.asyncio
async def test_maybe_rejoin_basename_only_with_single_candidate() -> None:
    """Basename match is used ONLY when exactly one alive candidate exists."""
    client = _StubClient(
        sessions=[
            _sess("vsc-sess", "vsc-kernel", "userwork.ipynb"),
        ]
    )
    session = _make_session(
        # server_relative is some absolute path; sessions list uses
        # workspace-relative encoding. basename matches.
        server_relative="some/where/userwork.ipynb",
        kernel_id="claude-kernel",
        client=client,
    )
    client._alive = {"vsc-kernel"}
    rejoined = await session.maybe_rejoin()
    assert rejoined is True
    assert session.kernel_id == "vsc-kernel"


@pytest.mark.asyncio
async def test_maybe_rejoin_basename_skipped_when_ambiguous() -> None:
    """Two basename candidates → don't pick, since they're in different dirs."""
    client = _StubClient(
        sessions=[
            _sess("a-sess", "k-a", "alpha/file.ipynb"),
            _sess("b-sess", "k-b", "beta/file.ipynb"),
        ]
    )
    session = _make_session(
        server_relative="gamma/file.ipynb",
        kernel_id="claude-kernel",
        client=client,
    )
    client._alive = {"k-a", "k-b"}
    rejoined = await session.maybe_rejoin()
    assert rejoined is False
    assert session.kernel_id == "claude-kernel"


@pytest.mark.asyncio
async def test_pinned_session_is_never_rejoined() -> None:
    client = _StubClient(
        sessions=[
            _sess("vsc-sess", "vsc-kernel", "project_a/analysis.ipynb"),
        ]
    )
    session = _make_session(client=client)
    session.pinned = True
    rejoined = await session.maybe_rejoin()
    assert rejoined is False
    assert session.kernel_id == "claude-kernel"


# ---------------------------------------------------------------------------
# rebind_to_kernel validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rebind_rejects_empty_target() -> None:
    client = _StubClient(sessions=[_sess("s1", "k1", "p.ipynb")])
    session = _make_session(client=client)
    outcome = await session.rebind_to_kernel("")
    assert isinstance(outcome, RebindOutcome)
    assert outcome.ok is False
    assert outcome.reason == "empty"


@pytest.mark.asyncio
async def test_rebind_rejects_short_prefix() -> None:
    client = _StubClient(sessions=[_sess("s1", "abcdefghij1234567890", "p.ipynb")])
    session = _make_session(client=client)
    outcome = await session.rebind_to_kernel("abcd")  # only 4 chars
    assert outcome.ok is False
    assert outcome.reason == "too_short"


@pytest.mark.asyncio
async def test_rebind_ambiguous_prefix_returns_candidates() -> None:
    """Two kernels sharing an 8-char prefix should be reported, not silently picked."""
    client = _StubClient(
        sessions=[
            _sess("s1", "deadbeef-aaaa", "a.ipynb"),
            _sess("s2", "deadbeef-bbbb", "b.ipynb"),
        ]
    )
    session = _make_session(client=client)
    outcome = await session.rebind_to_kernel("deadbeef")
    assert outcome.ok is False
    assert outcome.reason == "ambiguous"
    assert len(outcome.candidates) == 2


@pytest.mark.asyncio
async def test_rebind_exact_match_pins() -> None:
    client = _StubClient(
        sessions=[
            _sess("s-target", "k-target-uuid-stuff", "x.ipynb"),
        ]
    )
    session = _make_session(client=client)
    outcome = await session.rebind_to_kernel("k-target-uuid-stuff")
    assert outcome.ok is True
    assert session.kernel_id == "k-target-uuid-stuff"
    assert session.pinned is True


@pytest.mark.asyncio
async def test_rebind_to_dead_kernel_fails() -> None:
    client = _StubClient(
        sessions=[_sess("s1", "dead-kernel-id-12345", "p.ipynb")],
        alive=set(),  # nothing is alive
    )
    session = _make_session(client=client)
    outcome = await session.rebind_to_kernel("dead-kernel-id-12345")
    assert outcome.ok is False
    assert outcome.reason == "dead"
    assert session.pinned is False  # didn't pin after failure


# ---------------------------------------------------------------------------
# close ownership
# ---------------------------------------------------------------------------


class _DeleteTrackingClient(_StubClient):
    def __init__(self, sessions: list[dict]) -> None:
        super().__init__(sessions)
        self.deleted: list[str] = []

    async def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)


@pytest.mark.asyncio
async def test_close_does_not_delete_unowned_session() -> None:
    """After auto-rejoin onto VS Code's session, close_notebook must not
    delete that session — doing so would shut down the user's kernel.
    """
    client = _DeleteTrackingClient(
        sessions=[_sess("vsc-sess", "vsc-kernel", "p.ipynb")]
    )
    session = _make_session(
        client=client,
        session_id="claude-sess",
        kernel_id="claude-kernel",
    )
    # Simulate auto-rejoin onto an external session.
    session.session_id = "vsc-sess"
    session.kernel_id = "vsc-kernel"
    # owned_session_ids still contains only claude-sess. We don't add the
    # rejoined session to owned.
    await session.close(shutdown_kernel=True)
    assert "vsc-sess" not in client.deleted
    # We will, however, clean up our own original session if it still
    # exists on the server.
    assert "claude-sess" in client.deleted


@pytest.mark.asyncio
async def test_close_deletes_owned_session() -> None:
    client = _DeleteTrackingClient(
        sessions=[_sess("claude-sess", "claude-kernel", "p.ipynb")]
    )
    session = _make_session(client=client)
    await session.close(shutdown_kernel=True)
    assert client.deleted == ["claude-sess"]


# ---------------------------------------------------------------------------
# patch_cell strict-id behavior
# ---------------------------------------------------------------------------


class _NotebookFakeClient:
    """A stub that simulates the Contents API in memory."""

    def __init__(self, notebook: dict) -> None:
        self.notebook = notebook
        self.last_written: dict | None = None

    async def read_notebook(self, path: str) -> dict:
        # Hand back a deep-ish copy so test mutation can't leak into our state.
        import copy
        return copy.deepcopy(self.notebook)

    async def write_notebook(self, path: str, nb: dict) -> dict:
        self.notebook = nb
        self.last_written = nb
        return {}


def _nb(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


@pytest.mark.asyncio
async def test_patch_cell_raises_when_cell_id_disappeared() -> None:
    """If Claude was working on cell id=abc and the user deleted it, we MUST
    raise — not silently fall back to fallback_index and clobber a different
    cell.
    """
    client = _NotebookFakeClient(_nb([
        {"id": "other", "cell_type": "code", "source": "y = 2", "outputs": [], "execution_count": None},
    ]))
    session = _make_session(client=client)  # type: ignore[arg-type]

    with pytest.raises(CellNotFoundError):
        await session.patch_cell(
            cell_id="abc",            # disappeared
            fallback_index=0,         # would otherwise hit "other"
            mutate=lambda c: c.update(outputs=[{"output_type": "stream", "name": "stdout", "text": "X"}]),
        )
    # "other" must be untouched.
    assert client.notebook["cells"][0]["outputs"] == []


@pytest.mark.asyncio
async def test_patch_cell_uses_fallback_index_only_when_no_id() -> None:
    client = _NotebookFakeClient(_nb([
        {"cell_type": "code", "source": "y = 2", "outputs": [], "execution_count": None},
    ]))
    session = _make_session(client=client)  # type: ignore[arg-type]

    async def _mut(c: dict) -> None:
        c["source"] = "y = 99"

    await session.patch_cell(
        cell_id=None,
        fallback_index=0,
        mutate=lambda c: c.update(source="y = 99"),
    )
    assert client.notebook["cells"][0]["source"] == "y = 99"


# ---------------------------------------------------------------------------
# VS Code synthetic path regex
# ---------------------------------------------------------------------------


def test_vscode_synthetic_path_matches_real_format() -> None:
    assert is_vscode_synthetic_path(
        "analysis-jvsc-deadbeef-cafef00d.ipynb", "analysis"
    )


def test_vscode_synthetic_path_rejects_garbage_suffix() -> None:
    assert not is_vscode_synthetic_path(
        "analysis-jvsc-anything.ipynb", "analysis"
    )


def test_vscode_synthetic_path_rejects_non_ipynb_extension() -> None:
    assert not is_vscode_synthetic_path(
        "analysis-jvsc-deadbeef-cafef00d.txt", "analysis"
    )


def test_vscode_synthetic_path_rejects_wrong_stem() -> None:
    assert not is_vscode_synthetic_path(
        "other-jvsc-deadbeef-cafef00d.ipynb", "analysis"
    )


# ---------------------------------------------------------------------------
# Dead exact-path must not block a live synthetic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# mutate_notebook_fresh: the sync-before-mutate primitive
# ---------------------------------------------------------------------------


def _full_session_with_nb(nb: dict) -> tuple[NotebookSession, _NotebookFakeClient]:
    client = _NotebookFakeClient(nb)
    session = _make_session(client=client)  # type: ignore[arg-type]
    return session, client


@pytest.mark.asyncio
async def test_mutate_notebook_fresh_basic_write() -> None:
    session, client = _full_session_with_nb(_nb([
        {"id": "a", "cell_type": "code", "source": "x = 1", "outputs": [], "execution_count": None},
    ]))

    def _mut(nb: dict) -> None:
        nb["cells"][0]["source"] = "x = 99"

    out = await session.mutate_notebook_fresh(_mut, operation_name="test")
    assert out["cells"][0]["source"] == "x = 99"
    assert client.notebook["cells"][0]["source"] == "x = 99"


@pytest.mark.asyncio
async def test_mutate_notebook_fresh_missing_expected_cell_raises_conflict() -> None:
    """The target cell vanished between caller plan and our fresh read."""
    session, client = _full_session_with_nb(_nb([
        {"id": "other", "cell_type": "code", "source": "y = 2", "outputs": [], "execution_count": None},
    ]))

    with pytest.raises(NotebookConflictError):
        await session.mutate_notebook_fresh(
            lambda nb: None,
            expected_cell_id="vanished",
            operation_name="edit_cell",
        )
    # And not a write either.
    assert client.last_written is None


@pytest.mark.asyncio
async def test_mutate_notebook_fresh_expected_source_mismatch_raises_conflict() -> None:
    """Cell still exists, but its source changed since the caller resolved it."""
    session, client = _full_session_with_nb(_nb([
        {"id": "a", "cell_type": "code", "source": "x = 2", "outputs": [], "execution_count": None},
    ]))

    with pytest.raises(NotebookConflictError):
        await session.mutate_notebook_fresh(
            lambda nb: None,
            expected_cell_id="a",
            expected_source="x = 1",  # caller saw "x = 1" but it's now "x = 2"
            operation_name="edit_cell",
        )
    assert client.last_written is None


@pytest.mark.asyncio
async def test_mutate_notebook_fresh_preserves_concurrent_user_cell() -> None:
    """Between the caller planning the edit and the mutator running, the
    user added a brand-new cell in VS Code. The fresh read picks it up; the
    mutator only touches the target cell, and the user's cell survives.
    """
    session, client = _full_session_with_nb(_nb([
        {"id": "a", "cell_type": "code", "source": "x = 1", "outputs": [], "execution_count": None},
    ]))

    # Simulate "user added a cell from VS Code" by mutating the stub state
    # before we call. (The helper reads fresh, so it sees this state.)
    client.notebook["cells"].append(
        {"id": "user-new", "cell_type": "code", "source": "u = 99", "outputs": [], "execution_count": None}
    )

    def _edit_a(nb: dict) -> None:
        for c in nb["cells"]:
            if c.get("id") == "a":
                c["source"] = "x = 42"

    await session.mutate_notebook_fresh(
        _edit_a, expected_cell_id="a", operation_name="edit_cell",
    )
    ids = [c["id"] for c in client.notebook["cells"]]
    assert ids == ["a", "user-new"], "user-added cell must be preserved"
    sources = {c["id"]: c["source"] for c in client.notebook["cells"]}
    assert sources == {"a": "x = 42", "user-new": "u = 99"}


@pytest.mark.asyncio
async def test_clear_all_outputs_pattern_does_not_revert_source_edits() -> None:
    """The clear-all mutator only zeroes outputs/execution_count. Source
    edits the user made in VS Code between our read and the mutator running
    are read freshly and preserved.
    """
    session, client = _full_session_with_nb(_nb([
        {"id": "a", "cell_type": "code", "source": "x = 1",
         "outputs": [{"output_type": "stream", "name": "stdout", "text": "old"}],
         "execution_count": 1},
    ]))

    # User edits the source from VS Code right before our clear runs.
    client.notebook["cells"][0]["source"] = "x = 2  # user edit"

    def _clear_all(nb: dict) -> None:
        for c in nb.get("cells") or []:
            if c.get("cell_type") == "code" and c.get("outputs"):
                c["outputs"] = []
                c["execution_count"] = None

    await session.mutate_notebook_fresh(
        _clear_all, operation_name="clear_cell_outputs(all)"
    )
    cell = client.notebook["cells"][0]
    assert cell["source"] == "x = 2  # user edit"
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


@pytest.mark.asyncio
async def test_dead_exact_path_does_not_block_live_synthetic() -> None:
    """If our exact-path session is dead but a live VS Code synthetic exists,
    auto-rejoin should switch to the synthetic — the previous logic returned
    False just because *any* exact candidate existed.
    """
    client = _StubClient(
        sessions=[
            _sess("dead-sess", "dead-kernel", "project/analysis.ipynb"),
            _sess("vsc-sess", "vsc-kernel", "analysis-jvsc-deadbeef-cafef00d.ipynb"),
        ],
        alive={"vsc-kernel"},  # exact-path's kernel is dead
    )
    session = _make_session(
        server_relative="project/analysis.ipynb",
        session_id="dead-sess",
        kernel_id="dead-kernel",
        client=client,
    )
    rejoined = await session.maybe_rejoin()
    assert rejoined is True
    assert session.kernel_id == "vsc-kernel"
