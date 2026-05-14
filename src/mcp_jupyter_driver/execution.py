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


# ---- output helpers (pure, unit-testable) -----------------------------------


def _truncate_str(s: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if isinstance(s, list):
        s = "".join(x for x in s if isinstance(x, str))
    if not isinstance(s, str):
        return s
    if len(s) <= limit:
        return s
    return s[: limit - len(_TRUNCATED)] + _TRUNCATED


def _output_size(out: NotebookNode) -> int:
    otype = out.get("output_type")
    if otype == "stream":
        text = out.get("text", "")
        return len("".join(text) if isinstance(text, list) else text)
    if otype in ("display_data", "execute_result", "update_display_data"):
        total = 0
        for v in (out.get("data") or {}).values():
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
        prev["text"] = prev_text + new_text
        return True
    return False


# ---- main entrypoint --------------------------------------------------------


async def run_cell(
    session: NotebookSession,
    ref: int | str,
    *,
    timeout_s: float = 120.0,
    progress: ProgressCb | None = None,
    restart_on_kernel_death: bool = False,
) -> CellResult:
    """Execute a cell against the shared kernel and write outputs back."""
    async with session.exec_lock:
        # If VS Code is using a different kernel for this notebook (or our
        # kernel died), switch to a live one before we run.
        await session.maybe_rejoin()
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            return CellResult(status="ok", execution_count=None, output_count=0)

        cell["outputs"] = []
        cell["execution_count"] = None
        await session.write_notebook(nb)  # editor shows "running"

        try:
            async with session.client.kernel_channel(
                session.kernel_id, session.session_id
            ) as ch:
                msg_id = await ch.send(
                    "execute_request",
                    {
                        "code": cell.get("source", ""),
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
                    nb=nb,
                    cell=cell,
                    timeout_s=timeout_s,
                    progress=progress,
                )
                if result.has_widget:
                    await _snapshot_widgets(ch, nb)
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

        # Final write — outputs and execution_count now in place.
        await session.write_notebook(nb)
        return result


async def _consume(
    *,
    ch,
    msg_id: str,
    session: NotebookSession,
    nb: dict,
    cell: dict,
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

    while not (saw_idle and shell_reply_seen):
        if asyncio.get_event_loop().time() > deadline:
            cell["outputs"].append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=f"\n[mcp-jupyter-driver] cell timed out after {timeout_s:.0f}s; "
                    f"call interrupt_kernel to stop the kernel.\n",
                )
            )
            result.status = "error"
            result.error_name = "Timeout"
            result.error_value = "cell execution exceeded timeout_s"
            return result

        try:
            msg = await ch.recv(timeout=WS_TIMEOUT_S)
        except asyncio.TimeoutError:
            try:
                await session.client.get_kernel(session.kernel_id)
            except Exception:
                cell["outputs"].append(
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
                await session.write_notebook(nb)
                raise KernelDiedError(session.canonical)
            continue

        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue

        channel = msg.get("channel")
        mt = msg.get("msg_type") or (msg.get("header") or {}).get("msg_type")
        content = msg.get("content", {})

        if channel == "stdin":
            if mt == "input_request":
                # Auto-reply empty so the cell doesn't hang.
                try:
                    await ch.send(
                        "input_reply", {"value": ""}, channel="stdin"
                    )
                except Exception:
                    pass
                # We have to surface this to the caller; set a marker by
                # appending a stderr note. The caller checks result.interactive_input.
                result.interactive_input = True
            continue

        if channel == "shell":
            if mt == "execute_reply":
                shell_reply_seen = True
                if content.get("execution_count") is not None:
                    result.execution_count = content["execution_count"]
                    cell["execution_count"] = content["execution_count"]
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
                cell["execution_count"] = count
            continue

        if mt in ("comm_open", "comm_msg", "comm_close"):
            continue

        if mt in (
            "stream",
            "display_data",
            "execute_result",
            "error",
            "update_display_data",
        ):
            try:
                out = nbformat.v4.output_from_msg(msg)
            except Exception:
                continue

            if cell_bytes >= MAX_CELL_BYTES:
                if not truncation_marker_added:
                    cell["outputs"].append(
                        nbformat.v4.new_output(
                            output_type="stream", name="stderr", text=_TRUNCATED
                        )
                    )
                    truncation_marker_added = True
                    result.truncated = True
                continue

            if _cap_output(out):
                result.truncated = True
            this_size = _output_size(out)
            if cell_bytes + this_size > MAX_CELL_BYTES:
                result.truncated = True

            if mt == "stream":
                if not _coalesce_stream(cell["outputs"], out):
                    cell["outputs"].append(out)
            else:
                cell["outputs"].append(out)

            cell_bytes += this_size

            if mt == "error":
                result.status = "error"
                result.error_name = content.get("ename")
                result.error_value = content.get("evalue")
                result.error_traceback = list(content.get("traceback") or [])

            if widgets.outputs_contain_widget([out]):
                result.has_widget = True

            result.output_count = len(cell["outputs"])

            now = time.monotonic()
            if now - last_write >= WRITE_DEBOUNCE_S:
                last_write = now
                try:
                    await session.write_notebook(nb)
                except Exception:
                    pass

            if progress is not None:
                try:
                    await progress(
                        float(result.output_count), None,
                        f"{mt} ({result.output_count})",
                    )
                except Exception:
                    pass
            continue

    return result


async def _snapshot_widgets(ch, nb: dict) -> None:
    """Run silent helper on the existing channel; install state into nb.metadata."""
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
    if snapshot is not None:
        widgets.install_widget_state(nb.setdefault("metadata", {}), snapshot)
