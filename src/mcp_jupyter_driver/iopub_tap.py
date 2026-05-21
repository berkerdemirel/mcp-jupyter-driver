"""Persistent iopub subscriber per kernel: live awareness for Claude.

`run_cell` opens a kernel WebSocket only for the lifetime of one execution.
That leaves a blind spot: when the user runs a cell from VS Code, the MCP
never sees it. The next time Claude calls a tool, it has no idea what the
user did — only what changed in the .ipynb (and outputs alone are an
ambiguous signal).

This module owns a long-lived, *read-only* iopub WebSocket per active
kernel and keeps a bounded ring buffer of recent executions. iopub is a
broadcast channel — every connected client sees every execution, including
the ones VS Code triggers — so the tap is the cleanest way to give Claude
push-style awareness without polling or interfering with anyone's runs.

Output bodies are summarized aggressively (text truncated, images
dropped) so the buffer stays small even on data-heavy notebooks.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import NotebookSession


# Max preview length per output and per execution captured into the buffer.
# These are deliberately smaller than execution.py's caps because the tap is
# meant for "what happened recently" awareness, not faithful reproduction.
_OUTPUT_PREVIEW_CHARS = 2_000
_EXEC_OUTPUT_PREVIEW_CHARS = 8_000

# Default ring-buffer size in completed executions.
_DEFAULT_BUFFER_SIZE = 200

# Backoff bounds for the reconnect loop after a transient error.
_RECONNECT_BACKOFF_INITIAL_S = 0.5
_RECONNECT_BACKOFF_MAX_S = 5.0


@dataclass
class IopubExecution:
    """One observed kernel execution, regardless of who triggered it."""

    started_at: float
    parent_session: str
    parent_msg_id: str
    # True if the originating session_id matched any of NotebookSession's
    # historical session_ids at capture time. False means "the user (or
    # another client) ran this."
    by_claude: bool
    code: str | None = None
    execution_count: int | None = None
    outputs: list[dict] = field(default_factory=list)
    status: str = "in_progress"  # in_progress | ok | error
    error_name: str | None = None
    error_value: str | None = None
    finished_at: float | None = None


def _summarize_text(text: str | list, limit: int = _OUTPUT_PREVIEW_CHARS) -> str:
    s = "".join(x for x in text if isinstance(x, str)) if isinstance(text, list) else text
    if not isinstance(s, str):
        return ""
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _summarize_data_bundle(data: dict) -> dict:
    """Strip binary MIME and truncate text MIME so output bundles are cheap.

    Image bytes can be large; we keep only the MIME marker, not the body.
    """
    out: dict[str, str] = {}
    for mime, value in (data or {}).items():
        if mime.startswith("image/") or mime.startswith("application/vnd."):
            out[mime] = "<elided>"
            continue
        if isinstance(value, list):
            value = "".join(x for x in value if isinstance(x, str))
        if isinstance(value, str):
            out[mime] = _summarize_text(value)
    return out


def _summarize_output(msg_type: str, content: dict) -> dict | None:
    """Map a raw iopub payload to a small dict suitable for awareness logs.

    Returns None if the message isn't an output we want to record.
    """
    if msg_type == "stream":
        return {
            "type": "stream",
            "name": content.get("name"),
            "text": _summarize_text(content.get("text", "")),
        }
    if msg_type in ("display_data", "execute_result", "update_display_data"):
        return {
            "type": msg_type,
            "data": _summarize_data_bundle(content.get("data") or {}),
            "execution_count": content.get("execution_count"),
        }
    if msg_type == "error":
        return {
            "type": "error",
            "ename": content.get("ename"),
            "evalue": content.get("evalue"),
        }
    return None


class IopubTap:
    """Long-lived iopub listener attached to a ``NotebookSession``.

    Created and started lazily by ``NotebookSession.open``; stopped by
    ``NotebookSession.close``. When ``session.kernel_id`` changes (auto-rejoin,
    rebind, restart), the tap closes the current socket and reconnects to
    the new kernel on the next loop iteration.
    """

    def __init__(
        self,
        session: "NotebookSession",
        *,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
    ) -> None:
        self._session = session
        self._buffer: deque[IopubExecution] = deque(maxlen=buffer_size)
        # In-flight executions keyed by parent_msg_id.
        self._in_flight: dict[str, IopubExecution] = {}
        self._stop: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Spawn the background listen-and-reconnect task."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="iopub-tap")

    async def stop(self) -> None:
        """Signal stop and wait for the background task to finish."""
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                # Best-effort cleanup; never let a tap shutdown block close.
                pass
            self._task = None

    # ---- readers ---------------------------------------------------------

    def recent_executions(
        self, *, since: float | None = None, include_in_flight: bool = True
    ) -> list[IopubExecution]:
        """All recorded executions newer than ``since`` (epoch seconds).

        Order: oldest first. Includes still-running executions by default so
        callers can see what's currently happening.
        """
        out: list[IopubExecution] = []
        for ex in self._buffer:
            if since is None or ex.started_at >= since:
                out.append(ex)
        if include_in_flight:
            for ex in self._in_flight.values():
                if since is None or ex.started_at >= since:
                    out.append(ex)
        out.sort(key=lambda e: e.started_at)
        return out

    # ---- internals -------------------------------------------------------

    async def _run(self) -> None:
        backoff = _RECONNECT_BACKOFF_INITIAL_S
        while not self._stop.is_set():
            kid = self._session.kernel_id
            tap_session_id = f"mcp-iopub-tap-{kid[:8]}"
            try:
                async with self._session.client.kernel_channel(
                    kid, tap_session_id
                ) as ch:
                    backoff = _RECONNECT_BACKOFF_INITIAL_S
                    try:
                        while not self._stop.is_set():
                            # Reconnect if the session's kernel binding moved
                            # under us (auto-rejoin, rebind, restart).
                            if self._session.kernel_id != kid:
                                break
                            try:
                                msg = await ch.recv(timeout=0.5)
                            except asyncio.TimeoutError:
                                continue
                            try:
                                self._handle(msg)
                            except Exception:
                                # One malformed message must not kill the tap.
                                continue
                    finally:
                        # If we're leaving this kernel (rebind/restart/error)
                        # any still-pending in-flight executions will never
                        # receive their ``status: idle`` — finalize them now
                        # so they don't leak forever in ``_in_flight``.
                        self._finalize_in_flight(reason="disconnected")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Connection error, kernel restart, etc. Back off and retry.
                self._finalize_in_flight(reason="disconnected")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX_S)

    def _finalize_in_flight(self, *, reason: str) -> None:
        """Move all currently in-flight executions into the ring buffer with
        a synthetic finish time. Used when the iopub connection drops and we
        know we will never see the matching ``status: idle`` for those
        executions."""
        if not self._in_flight:
            return
        now = time.time()
        for ex in list(self._in_flight.values()):
            if ex.status == "in_progress":
                ex.status = reason
            ex.finished_at = now
            self._buffer.append(ex)
        self._in_flight.clear()

    def _handle(self, msg: dict) -> None:
        if msg.get("channel") != "iopub":
            return
        mt = msg.get("msg_type") or (msg.get("header") or {}).get("msg_type")
        parent = msg.get("parent_header") or {}
        parent_msg_id = parent.get("msg_id") or ""
        parent_session = parent.get("session") or ""
        content = msg.get("content", {}) or {}

        # execute_input opens a new execution record.
        if mt == "execute_input":
            ex = IopubExecution(
                started_at=time.time(),
                parent_session=parent_session,
                parent_msg_id=parent_msg_id,
                by_claude=self._is_claude_session(parent_session),
                code=content.get("code"),
                execution_count=content.get("execution_count"),
            )
            self._in_flight[parent_msg_id] = ex
            return

        # status idle finalizes the matching execution.
        if mt == "status" and content.get("execution_state") == "idle":
            ex = self._in_flight.pop(parent_msg_id, None)
            if ex is None:
                return
            if ex.status == "in_progress":
                ex.status = "ok"
            ex.finished_at = time.time()
            self._buffer.append(ex)
            return

        # Outputs: append to the matching in-flight execution.
        ex = self._in_flight.get(parent_msg_id)
        if ex is None:
            return
        summary = _summarize_output(mt or "", content)
        if summary is None:
            return
        if summary.get("type") == "error":
            ex.status = "error"
            ex.error_name = summary.get("ename")
            ex.error_value = summary.get("evalue")
        # Cap the per-execution output payload so a chatty cell doesn't
        # balloon the ring buffer.
        if self._exec_size(ex) < _EXEC_OUTPUT_PREVIEW_CHARS:
            ex.outputs.append(summary)

    def _exec_size(self, ex: IopubExecution) -> int:
        total = 0
        for o in ex.outputs:
            if "text" in o and isinstance(o["text"], str):
                total += len(o["text"])
            elif "data" in o and isinstance(o["data"], dict):
                total += sum(len(v) for v in o["data"].values() if isinstance(v, str))
            elif "evalue" in o and isinstance(o["evalue"], str):
                total += len(o["evalue"])
        return total

    def _is_claude_session(self, parent_session: str) -> bool:
        """A parent_session is Claude's if it matches any session_id the
        NotebookSession has bound (now or historically). The historical set
        is tracked on the session itself."""
        if not parent_session:
            return False
        if parent_session == self._session.session_id:
            return True
        return parent_session in getattr(
            self._session, "historical_session_ids", set()
        )
