"""Unit tests for NotebookSession behavior that doesn't require a real kernel.

We exercise maybe_rejoin's candidate-ranking logic and rebind validation
against a stub JupyterClient. This keeps the dangerous collaboration paths
covered even on hosts where Jupyter Server can't start (containers running
as root without --allow-root, hosts without ipykernel, CI).
"""

from __future__ import annotations

import pytest

from mcp_jupyter_driver.errors import CellNotFoundError
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
