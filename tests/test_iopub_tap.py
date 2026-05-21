"""Unit tests for the iopub tap's message dispatch.

We bypass the WebSocket entirely and feed synthetic iopub messages to
``IopubTap._handle`` to verify Claude-vs-user attribution, execution
lifecycle (open → outputs → status idle), and output summarization caps.
"""

from __future__ import annotations

import pytest

from mcp_jupyter_driver.iopub_tap import IopubTap
from mcp_jupyter_driver.session import NotebookSession, canonical_key


class _StubClient:
    async def list_sessions(self):
        return []


def _make_session(*, session_id: str = "claude-sess") -> NotebookSession:
    path = "/test/work.ipynb"
    return NotebookSession(
        canonical=canonical_key(path),
        server_relative="test/work.ipynb",
        session_id=session_id,
        kernel_id="kernel-xyz",
        kernel_name="python3",
        client=_StubClient(),  # type: ignore[arg-type]
    )


def _iopub(msg_type: str, parent_msg_id: str, parent_session: str, content: dict) -> dict:
    return {
        "channel": "iopub",
        "msg_type": msg_type,
        "parent_header": {"msg_id": parent_msg_id, "session": parent_session},
        "content": content,
    }


def test_claude_run_records_execution_with_by_claude_true() -> None:
    session = _make_session(session_id="claude-sess")
    tap = IopubTap(session)

    tap._handle(_iopub("execute_input", "m1", "claude-sess", {"code": "x = 1", "execution_count": 7}))
    tap._handle(_iopub("stream", "m1", "claude-sess", {"name": "stdout", "text": "hello\n"}))
    tap._handle(_iopub("status", "m1", "claude-sess", {"execution_state": "idle"}))

    execs = tap.recent_executions()
    assert len(execs) == 1
    ex = execs[0]
    assert ex.by_claude is True
    assert ex.code == "x = 1"
    assert ex.execution_count == 7
    assert ex.status == "ok"
    assert ex.outputs[0]["type"] == "stream"
    assert "hello" in ex.outputs[0]["text"]


def test_user_run_recorded_with_by_claude_false() -> None:
    session = _make_session(session_id="claude-sess")
    tap = IopubTap(session)

    tap._handle(_iopub("execute_input", "m2", "vscode-sess", {"code": "y = 2"}))
    tap._handle(_iopub("status", "m2", "vscode-sess", {"execution_state": "idle"}))

    execs = tap.recent_executions()
    assert len(execs) == 1
    assert execs[0].by_claude is False
    assert execs[0].code == "y = 2"


def test_historical_session_ids_attribute_old_runs_to_claude() -> None:
    """After a rejoin, old executions in the in-flight log must still be
    attributed to Claude even though session_id changed."""
    session = _make_session(session_id="new-sess")
    session.historical_session_ids.add("old-sess")
    tap = IopubTap(session)

    tap._handle(_iopub("execute_input", "m3", "old-sess", {"code": "z = 3"}))
    tap._handle(_iopub("status", "m3", "old-sess", {"execution_state": "idle"}))

    assert tap.recent_executions()[0].by_claude is True


def test_error_output_marks_execution_status_error() -> None:
    session = _make_session()
    tap = IopubTap(session)
    pid = "m4"
    tap._handle(_iopub("execute_input", pid, session.session_id, {"code": "1/0"}))
    tap._handle(_iopub("error", pid, session.session_id, {
        "ename": "ZeroDivisionError",
        "evalue": "division by zero",
        "traceback": ["..."],
    }))
    tap._handle(_iopub("status", pid, session.session_id, {"execution_state": "idle"}))

    ex = tap.recent_executions()[0]
    assert ex.status == "error"
    assert ex.error_name == "ZeroDivisionError"
    assert ex.error_value == "division by zero"


def test_image_data_bundle_elided_to_keep_buffer_small() -> None:
    session = _make_session()
    tap = IopubTap(session)
    pid = "m5"
    tap._handle(_iopub("execute_input", pid, session.session_id, {"code": "plt.show()"}))
    tap._handle(_iopub("display_data", pid, session.session_id, {
        "data": {
            "image/png": "AAAA" * 50_000,  # 200KB pretend
            "text/plain": "<Figure>",
        },
    }))
    tap._handle(_iopub("status", pid, session.session_id, {"execution_state": "idle"}))

    out = tap.recent_executions()[0].outputs[0]
    assert out["data"]["image/png"] == "<elided>"
    assert out["data"]["text/plain"] == "<Figure>"


def test_outputs_with_no_matching_execute_input_are_ignored() -> None:
    """If a stream arrives without a preceding execute_input (e.g. tap
    started mid-execution), we drop it instead of crashing."""
    session = _make_session()
    tap = IopubTap(session)
    tap._handle(_iopub("stream", "unknown-msg", session.session_id, {"name": "stdout", "text": "x"}))
    assert tap.recent_executions() == []


def test_non_iopub_messages_ignored() -> None:
    session = _make_session()
    tap = IopubTap(session)
    tap._handle({"channel": "shell", "msg_type": "execute_reply", "parent_header": {}, "content": {}})
    assert tap.recent_executions() == []


def test_recent_executions_filters_by_since() -> None:
    session = _make_session()
    tap = IopubTap(session)
    pid = "m6"
    tap._handle(_iopub("execute_input", pid, session.session_id, {"code": "a = 1"}))
    tap._handle(_iopub("status", pid, session.session_id, {"execution_state": "idle"}))

    ex = tap.recent_executions()[0]
    # Far-future cutoff returns nothing.
    assert tap.recent_executions(since=ex.finished_at + 1) == []
    # Cutoff at exactly its started_at still includes the execution.
    assert len(tap.recent_executions(since=ex.started_at)) == 1


@pytest.mark.asyncio
async def test_stop_when_never_started_is_no_op() -> None:
    """Closing a notebook session whose tap never got an event loop must
    not crash. NotebookSession.close calls iopub_tap.stop() — verify it's
    safe even when start() was never called."""
    session = _make_session()
    tap = IopubTap(session)
    await tap.stop()  # should not raise
