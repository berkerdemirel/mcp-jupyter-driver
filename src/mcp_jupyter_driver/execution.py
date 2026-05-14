"""Cell execution via the Jupyter Server's kernel WebSocket.

Both Claude (this module) and the user's editor connect to the same kernel
through the server. Outputs come back on iopub; we apply them to the
in-memory notebook and PUT the updated notebook through the Contents API so
the editor's notebook view sees the result.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import nbformat
from nbformat.notebooknode import NotebookNode

from . import widgets
from .errors import KernelDiedError
from .session import NotebookSession, resolve_cell_index

ProgressCb = Callable[[float, float | None, str], Awaitable[None]]

# Per-recv wait timeout; on timeout we check kernel liveness.
WS_TIMEOUT_S = 0.5
# Debounce window for writing the notebook back during streaming output.
WRITE_DEBOUNCE_S = 0.2

# Per-output / per-cell size caps. Images & widget MIME are not capped.
MAX_OUTPUT_BYTES = 1 * 1024 * 1024
MAX_CELL_BYTES = 5 * 1024 * 1024
_TRUNCATED = "\n…[mcp-jupyter-driver: output truncated]\n"


@dataclass
class CellResult:
    status: str  # "ok" | "error" | "kernel_died"
    execution_count: int | None
    output_count: int = 0
    has_widget: bool = False
    error_name: str | None = None
    error_value: str | None = None
    error_traceback: list[str] = field(default_factory=list)
    truncated: bool = False
    interactive_input: bool = False
    kernel_restarted: bool = False
    # Set when we returned because the timeout fired but the kernel is still
    # running. Callers should call interrupt_kernel or accept that subsequent
    # outputs may keep arriving on the live kernel.
    kernel_still_running: bool = False


# ---- output helpers (pure, unit-testable) -----------------------------------


def _truncate_str(s: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if isinstance(s, list):
        s = "".join(x for x in s if isinstance(x, str))
    if not isinstance(s, str):
        return s
    if len(s) <= limit:
        return s
    return s[: limit - len(_TRUNCATED)] + _TRUNCATED


# MIME types that don't count toward the cell-size cap. Binary images can be
# large but they're the whole point of displaying a plot, and widget MIME is
# small JSON but uncapping it keeps interactive widgets faithful.
_EXEMPT_MIME_PREFIXES: tuple[str, ...] = ("image/",)
_EXEMPT_MIME_TYPES = {
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
}


def _is_exempt_mime(mime: str) -> bool:
    if mime in _EXEMPT_MIME_TYPES:
        return True
    for pref in _EXEMPT_MIME_PREFIXES:
        if mime.startswith(pref):
            return True
    return False


def _output_size(out: NotebookNode) -> int:
    otype = out.get("output_type")
    if otype == "stream":
        text = out.get("text", "")
        return len("".join(text) if isinstance(text, list) else text)
    if otype in ("display_data", "execute_result", "update_display_data"):
        total = 0
        for k, v in (out.get("data") or {}).items():
            if _is_exempt_mime(k):
                continue
            if isinstance(v, list):
                total += sum(len(x) for x in v if isinstance(x, str))
            elif isinstance(v, str):
                total += len(v)
        return total
    if otype == "error":
        return sum(len(t) for t in (out.get("traceback") or []))
    return 0


def _cap_output(out: NotebookNode) -> bool:
    trimmed = False
    otype = out.get("output_type")
    if otype == "stream":
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if isinstance(text, str) and len(text) > MAX_OUTPUT_BYTES:
            trimmed = True
        out["text"] = _truncate_str(text)
    elif otype in ("display_data", "execute_result", "update_display_data"):
        data = out.get("data") or {}
        for k, v in list(data.items()):
            if k.startswith("text/") or k == "application/json":
                if isinstance(v, list):
                    v = "".join(x for x in v if isinstance(x, str))
                if isinstance(v, str):
                    if len(v) > MAX_OUTPUT_BYTES:
                        trimmed = True
                    data[k] = _truncate_str(v)
    return trimmed


def _coalesce_stream(outputs: list[NotebookNode], new_out: NotebookNode) -> bool:
    if not outputs:
        return False
    prev = outputs[-1]
    if (
        prev.get("output_type") == "stream"
        and new_out.get("output_type") == "stream"
        and prev.get("name") == new_out.get("name")
    ):
        prev_text = prev.get("text", "")
        if isinstance(prev_text, list):
            prev_text = "".join(prev_text)
        new_text = new_out.get("text", "")
        if isinstance(new_text, list):
            new_text = "".join(new_text)
        merged = prev_text + new_text
        truncated = False
        if len(merged) > MAX_OUTPUT_BYTES:
            merged = _truncate_str(merged)
            truncated = True
        prev["text"] = merged
        if truncated:
            prev["_mcp_truncated"] = True
        return True
    return False


def _display_id_of(msg: dict) -> str | None:
    """display_data/update_display_data carry a display_id in transient.

    We use it to find prior outputs to update — nbformat.output_from_msg
    doesn't recognize update_display_data on its own.
    """
    transient = (msg.get("content", {}) or {}).get("transient") or {}
    did = transient.get("display_id")
    return did if isinstance(did, str) and did else None


# ---- main entrypoint --------------------------------------------------------


async def run_cell(
    session: NotebookSession,
    ref: int | str,
    *,
    timeout_s: float = 120.0,
    progress: ProgressCb | None = None,
    restart_on_kernel_death: bool = False,
) -> CellResult:
    """Execute a cell against the shared kernel and write outputs back.

    Writes are always re-read-then-patch by cell id (falling back to index
    only when no id is present) so a concurrent VS Code edit during the run
    can't be clobbered by a stale full-notebook PUT.
    """
    async with session.exec_lock:
        # If VS Code is using a different kernel for this notebook (or our
        # kernel died), switch to a live one before we run.
        await session.maybe_rejoin()
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            return CellResult(status="ok", execution_count=None, output_count=0)

        cell_id = cell.get("id")
        src = cell.get("source", "")
        source = "".join(src) if isinstance(src, list) else (src or "")

        # Working copy: list of outputs we'll patch into the on-disk cell.
        # We never PUT the whole notebook from here — only the target cell.
        outputs: list[NotebookNode] = []
        # Initial clear is strict: if the cell already disappeared (user
        # deleted it before we even started), fail loudly.
        await _flush_cell(
            session, cell_id, idx,
            outputs=[], execution_count=None, best_effort=False,
        )

        try:
            async with session.client.kernel_channel(
                session.kernel_id, session.session_id
            ) as ch:
                msg_id = await ch.send(
                    "execute_request",
                    {
                        "code": source,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": True,
                        "stop_on_error": True,
                    },
                )

                result = await _consume(
                    ch=ch,
                    msg_id=msg_id,
                    session=session,
                    cell_id=cell_id,
                    fallback_index=idx,
                    outputs=outputs,
                    timeout_s=timeout_s,
                    progress=progress,
                )
                if result.has_widget:
                    await _snapshot_widgets(session, ch)
        except KernelDiedError:
            if restart_on_kernel_death:
                try:
                    await session.client.restart_kernel(session.kernel_id)
                except Exception:
                    raise
                return CellResult(
                    status="kernel_died",
                    execution_count=None,
                    kernel_restarted=True,
                )
            raise

        # Final flush is strict — if this fails, run_cell must raise rather
        # than return success while the file on disk is missing outputs.
        await _flush_cell(
            session,
            cell_id,
            idx,
            outputs=outputs,
            execution_count=result.execution_count,
            best_effort=False,
        )
        return result


async def _flush_cell(
    session: NotebookSession,
    cell_id: str | None,
    fallback_index: int,
    *,
    outputs: list[NotebookNode],
    execution_count: int | None,
    best_effort: bool = False,
) -> None:
    """Read fresh, patch only the target cell's outputs+execution_count, write.

    Streaming flushes pass ``best_effort=True`` — transient server hiccups
    during long output runs shouldn't fail the cell. The initial and final
    flushes pass ``best_effort=False`` so the caller actually sees missing
    cells / write failures instead of returning ok=True on a silent miss.
    """

    def _apply(c: dict) -> None:
        if c.get("cell_type") != "code":
            return
        c["outputs"] = list(outputs)
        c["execution_count"] = execution_count

    try:
        await session.patch_cell(
            cell_id=cell_id, fallback_index=fallback_index, mutate=_apply
        )
    except Exception:
        if best_effort:
            return
        raise


async def _consume(
    *,
    ch,
    msg_id: str,
    session: NotebookSession,
    cell_id: str | None,
    fallback_index: int,
    outputs: list[NotebookNode],
    timeout_s: float,
    progress: ProgressCb | None,
) -> CellResult:
    result = CellResult(status="ok", execution_count=None)
    deadline = asyncio.get_event_loop().time() + timeout_s
    cell_bytes = 0
    last_write = 0.0
    truncation_marker_added = False
    saw_idle = False
    shell_reply_seen = False
    # display_id -> index in outputs (for update_display_data).
    display_ids: dict[str, int] = {}
    # If clear_output(wait=True) was requested, clear before the next output.
    clear_pending = False

    def _do_clear() -> None:
        nonlocal cell_bytes, truncation_marker_added
        outputs.clear()
        display_ids.clear()
        cell_bytes = 0
        truncation_marker_added = False

    while not (saw_idle and shell_reply_seen):
        if asyncio.get_event_loop().time() > deadline:
            outputs.append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=f"\n[mcp-jupyter-driver] cell timed out after {timeout_s:.0f}s; "
                    f"kernel is still running — call interrupt_kernel to stop it.\n",
                )
            )
            result.status = "error"
            result.error_name = "Timeout"
            result.error_value = "cell execution exceeded timeout_s"
            result.kernel_still_running = True
            result.output_count = len(outputs)
            return result

        try:
            msg = await ch.recv(timeout=WS_TIMEOUT_S)
        except asyncio.TimeoutError:
            try:
                await session.client.get_kernel(session.kernel_id)
            except Exception:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="error",
                        ename="KernelDied",
                        evalue="kernel exited during execution",
                        traceback=[],
                    )
                )
                result.status = "kernel_died"
                result.error_name = "KernelDied"
                result.error_value = "kernel exited during execution"
                result.output_count = len(outputs)
                await _flush_cell(
                    session, cell_id, fallback_index,
                    outputs=outputs, execution_count=result.execution_count,
                    best_effort=True,
                )
                raise KernelDiedError(session.canonical)
            continue

        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue

        channel = msg.get("channel")
        mt = msg.get("msg_type") or (msg.get("header") or {}).get("msg_type")
        content = msg.get("content", {})

        if channel == "stdin":
            if mt == "input_request":
                try:
                    await ch.send(
                        "input_reply", {"value": ""}, channel="stdin"
                    )
                except Exception:
                    pass
                result.interactive_input = True
            continue

        if channel == "shell":
            if mt == "execute_reply":
                shell_reply_seen = True
                if content.get("execution_count") is not None:
                    result.execution_count = content["execution_count"]
                # Use shell reply as fallback when iopub error was missed.
                if content.get("status") == "error" and result.status != "error":
                    result.status = "error"
                    result.error_name = content.get("ename")
                    result.error_value = content.get("evalue")
                    result.error_traceback = list(content.get("traceback") or [])
            continue

        # channel == "iopub" (or unspecified)
        if mt == "status":
            if content.get("execution_state") == "idle":
                saw_idle = True
            continue

        if mt == "execute_input":
            count = content.get("execution_count")
            if count is not None:
                result.execution_count = count
            continue

        if mt in ("comm_open", "comm_msg", "comm_close"):
            continue

        if mt == "clear_output":
            if content.get("wait"):
                clear_pending = True
            else:
                _do_clear()
                clear_pending = False
            continue

        if mt == "update_display_data":
            did = _display_id_of(msg)
            if did is None or did not in display_ids:
                # No prior matching display — nothing to do.
                continue
            try:
                refreshed = nbformat.v4.new_output(
                    output_type="display_data",
                    data=dict(content.get("data") or {}),
                    metadata=dict(content.get("metadata") or {}),
                )
            except Exception:
                continue
            if _cap_output(refreshed):
                result.truncated = True
            target_idx = display_ids[did]
            prev = outputs[target_idx]
            cell_bytes = max(0, cell_bytes - _output_size(prev))
            outputs[target_idx] = refreshed
            cell_bytes += _output_size(refreshed)
            now = time.monotonic()
            if now - last_write >= WRITE_DEBOUNCE_S:
                last_write = now
                await _flush_cell(
                    session, cell_id, fallback_index,
                    outputs=outputs, execution_count=result.execution_count,
                    best_effort=True,
                )
            continue

        if mt in ("stream", "display_data", "execute_result", "error"):
            # Honor pending clear (wait=True) right before adding the next output.
            if clear_pending:
                _do_clear()
                clear_pending = False

            try:
                out = nbformat.v4.output_from_msg(msg)
            except Exception:
                continue

            if cell_bytes >= MAX_CELL_BYTES:
                if not truncation_marker_added:
                    outputs.append(
                        nbformat.v4.new_output(
                            output_type="stream", name="stderr", text=_TRUNCATED
                        )
                    )
                    truncation_marker_added = True
                    result.truncated = True
                # Drop the over-cap output entirely so we honor the documented cap.
                continue

            if _cap_output(out):
                result.truncated = True
            this_size = _output_size(out)
            if cell_bytes + this_size > MAX_CELL_BYTES:
                # Don't append an output that would push us over. Insert the
                # truncation marker instead and stop accepting more.
                if not truncation_marker_added:
                    outputs.append(
                        nbformat.v4.new_output(
                            output_type="stream", name="stderr", text=_TRUNCATED
                        )
                    )
                    truncation_marker_added = True
                result.truncated = True
                continue

            if mt == "stream":
                if not _coalesce_stream(outputs, out):
                    outputs.append(out)
                # After (possible) coalesce, the merged last output may have
                # been capped — reflect that in result.truncated.
                if outputs and outputs[-1].pop("_mcp_truncated", False):
                    result.truncated = True
            else:
                outputs.append(out)
                if mt == "display_data":
                    did = _display_id_of(msg)
                    if did is not None:
                        display_ids[did] = len(outputs) - 1

            cell_bytes += this_size

            if mt == "error":
                result.status = "error"
                result.error_name = content.get("ename")
                result.error_value = content.get("evalue")
                result.error_traceback = list(content.get("traceback") or [])

            if widgets.outputs_contain_widget([out]):
                result.has_widget = True

            result.output_count = len(outputs)

            now = time.monotonic()
            if now - last_write >= WRITE_DEBOUNCE_S:
                last_write = now
                await _flush_cell(
                    session, cell_id, fallback_index,
                    outputs=outputs, execution_count=result.execution_count,
                    best_effort=True,
                )

            if progress is not None:
                try:
                    await progress(
                        float(result.output_count), None,
                        f"{mt} ({result.output_count})",
                    )
                except Exception:
                    pass
            continue

    result.output_count = len(outputs)
    return result


async def _snapshot_widgets(session: NotebookSession, ch) -> None:
    """Run silent helper on the existing channel; install state into nb.metadata.

    Reads the latest notebook back from the server before writing so a
    concurrent VS Code edit doesn't get clobbered.
    """
    msg_id = await ch.send(
        "execute_request",
        {
            "code": widgets.snapshot_request_code(),
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
    )
    collected: list[str] = []
    deadline = asyncio.get_event_loop().time() + 5.0
    saw_idle = False
    shell_reply_seen = False
    while not (saw_idle and shell_reply_seen):
        if asyncio.get_event_loop().time() > deadline:
            return
        try:
            msg = await ch.recv(timeout=WS_TIMEOUT_S)
        except asyncio.TimeoutError:
            continue
        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue
        mt = msg.get("msg_type")
        content = msg.get("content", {})
        if msg.get("channel") == "shell" and mt == "execute_reply":
            shell_reply_seen = True
            continue
        if mt == "stream" and content.get("name") == "stdout":
            text = content.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            collected.append(text)
        elif mt == "status" and content.get("execution_state") == "idle":
            saw_idle = True

    snapshot = widgets.parse_snapshot_stdout("".join(collected))
    if snapshot is None:
        return
    fresh = await session.read_notebook()
    widgets.install_widget_state(fresh.setdefault("metadata", {}), snapshot)
    await session.write_notebook(fresh)
