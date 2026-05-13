"""Cell execution: send to kernel, consume iopub, write outputs back live.

We treat the in-memory `nb` as the source of truth and the .ipynb file as a
projection of it (re-written atomically + debounced as outputs arrive). The
debounced writer collapses bursts; we flush before returning so the editor
always sees the final state when the tool call ends.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import nbformat
from nbformat.notebooknode import NotebookNode

from . import widgets
from .errors import InteractiveInputError, KernelDiedError
from .session import NotebookSession

ProgressCb = Callable[[float, float | None, str], Awaitable[None]]
# Per-message wait timeout while a cell is running; if a single iopub msg
# doesn't arrive within this window we check whether the kernel is alive.
IOPUB_TIMEOUT_S = 0.5

# Output size caps. Individual outputs (stream text, traceback, single MIME
# values) are truncated at MAX_OUTPUT_BYTES with a marker. Once a cell's
# total accumulated payload exceeds MAX_CELL_BYTES, further outputs are
# dropped and a single truncation marker is appended.
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


def _truncate_str(s: str, limit: int = MAX_OUTPUT_BYTES) -> str:
    if isinstance(s, list):
        s = "".join(s)
    if not isinstance(s, str):
        return s
    if len(s) <= limit:
        return s
    return s[: limit - len(_TRUNCATED)] + _TRUNCATED


def _output_size(out: NotebookNode) -> int:
    """Cheap size estimate of an output's payload (chars)."""
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
    """Trim oversize fields in a single output in place. Returns True if any
    field had to be truncated.
    """
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
    """If the previous output is a stream of the same name, merge text into it."""
    if not outputs:
        return False
    prev = outputs[-1]
    if (
        prev.get("output_type") == "stream"
        and new_out.get("output_type") == "stream"
        and prev.get("name") == new_out.get("name")
    ):
        prev["text"] = prev.get("text", "") + new_out.get("text", "")
        return True
    return False


async def run_cell(
    session: NotebookSession,
    ref: int | str,
    *,
    timeout_s: float = 120.0,
    progress: ProgressCb | None = None,
    restart_on_kernel_death: bool = False,
) -> CellResult:
    """Execute one cell and stream outputs back to the .ipynb file.

    If the kernel dies mid-execution and `restart_on_kernel_death` is True,
    the kernel is restarted before returning (result.kernel_restarted=True).
    Otherwise raises KernelDiedError.
    """
    async with session.exec_lock:
        await session.assert_alive()
        cell_index = session.resolve_cell_index(ref)
        cell = session.nb.cells[cell_index]
        if cell.get("cell_type") != "code":
            return CellResult(status="ok", execution_count=None, output_count=0)

        cell["outputs"] = []
        cell["execution_count"] = None
        session.writer.schedule()

        kc = session.kc
        msg_id = kc.execute(cell.get("source", ""), store_history=True)

        stdin_seen: list[bool] = [False]
        stdin_task = asyncio.create_task(_handle_stdin(session, msg_id, stdin_seen))
        try:
            result = await _consume_iopub(
                session=session,
                cell=cell,
                msg_id=msg_id,
                deadline=asyncio.get_event_loop().time() + timeout_s,
                progress=progress,
            )
        except KernelDiedError:
            if restart_on_kernel_death:
                await session.km.restart_kernel(now=True)
                result = CellResult(
                    status="kernel_died",
                    execution_count=None,
                    kernel_restarted=True,
                )
            else:
                raise
        finally:
            stdin_task.cancel()
            try:
                await stdin_task
            except (asyncio.CancelledError, Exception):
                pass
        if stdin_seen[0]:
            result.interactive_input = True

        # Try to get the execution_count from the shell reply.
        try:
            shell_reply = await asyncio.wait_for(
                _shell_reply_for(kc, msg_id), timeout=2.0
            )
            count = shell_reply.get("content", {}).get("execution_count")
            if count is not None:
                result.execution_count = count
                cell["execution_count"] = count
        except asyncio.TimeoutError:
            pass

        # Widget-state snapshot if this cell produced a widget view.
        if result.has_widget:
            await _snapshot_widget_state(session)

        await session.writer.flush()
        return result


async def _handle_stdin(session: NotebookSession, msg_id: str, seen: list[bool]) -> None:
    """Auto-reply empty string to input_request so a cell that called input()
    doesn't hang forever. Flips `seen[0]` so the caller can surface this.
    """
    kc = session.kc
    while True:
        try:
            msg = await kc.get_stdin_msg()
        except Exception:
            await asyncio.sleep(0.1)
            continue
        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue
        if msg.get("msg_type") == "input_request":
            seen[0] = True
            try:
                kc.input("")
            except Exception:
                pass


async def _consume_iopub(
    *,
    session: NotebookSession,
    cell: NotebookNode,
    msg_id: str,
    deadline: float,
    progress: ProgressCb | None,
) -> CellResult:
    result = CellResult(status="ok", execution_count=None)
    kc = session.kc
    loop = asyncio.get_event_loop()
    saw_idle_for_us = False
    cell_bytes = 0
    truncation_marker_added = False

    while not saw_idle_for_us:
        if loop.time() > deadline:
            # Soft timeout: don't kill the kernel, just stop streaming. Caller
            # can interrupt explicitly if they want.
            cell["outputs"].append(
                nbformat.v4.new_output(
                    output_type="stream",
                    name="stderr",
                    text=f"\n[mcp-jupyter-driver] cell timed out after {deadline:.0f}s; "
                    f"call interrupt_kernel to stop the kernel.\n",
                )
            )
            result.status = "error"
            result.error_name = "Timeout"
            result.error_value = "cell execution exceeded timeout_s"
            return result

        try:
            msg = await asyncio.wait_for(kc.get_iopub_msg(), timeout=IOPUB_TIMEOUT_S)
        except asyncio.TimeoutError:
            if not await session.km.is_alive():
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
                await session.writer.flush()
                raise KernelDiedError(str(session.path))
            continue

        parent = (msg.get("parent_header") or {}).get("msg_id")
        if parent != msg_id:
            # Not ours (e.g. silent kernel-side helper from another caller).
            continue

        msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
        content = msg.get("content", {})

        if msg_type == "status":
            if content.get("execution_state") == "idle":
                saw_idle_for_us = True
            continue

        if msg_type == "execute_input":
            count = content.get("execution_count")
            if count is not None:
                cell["execution_count"] = count
                result.execution_count = count
            continue

        if msg_type in ("comm_open", "comm_msg", "comm_close"):
            _record_comm(session, msg_type, content)
            continue

        if msg_type in ("stream", "display_data", "execute_result", "error", "update_display_data"):
            try:
                out = nbformat.v4.output_from_msg(msg)
            except Exception:
                continue

            if cell_bytes >= MAX_CELL_BYTES:
                # Already past the cap; drop further outputs after appending
                # exactly one marker.
                if not truncation_marker_added:
                    cell["outputs"].append(
                        nbformat.v4.new_output(
                            output_type="stream", name="stderr", text=_TRUNCATED
                        )
                    )
                    truncation_marker_added = True
                    result.truncated = True
                    session.writer.schedule()
                continue

            if _cap_output(out):
                result.truncated = True
            this_size = _output_size(out)
            if cell_bytes + this_size > MAX_CELL_BYTES:
                result.truncated = True

            if msg_type == "stream":
                if not _coalesce_stream(cell["outputs"], out):
                    cell["outputs"].append(out)
            else:
                cell["outputs"].append(out)

            cell_bytes += this_size

            if msg_type == "error":
                result.status = "error"
                result.error_name = content.get("ename")
                result.error_value = content.get("evalue")
                result.error_traceback = list(content.get("traceback") or [])

            if widgets.outputs_contain_widget([out]):
                result.has_widget = True

            result.output_count = len(cell["outputs"])
            session.writer.schedule()
            if progress is not None:
                try:
                    await progress(
                        float(result.output_count),
                        None,
                        f"{msg_type} ({result.output_count} outputs)",
                    )
                except Exception:
                    pass
            continue

        # Unknown message type: ignore silently to stay forward-compatible.

    return result


async def _shell_reply_for(kc: Any, msg_id: str) -> dict[str, Any]:
    """Drain shell channel until we see a reply with our parent msg_id."""
    while True:
        msg = await kc.get_shell_msg()
        if (msg.get("parent_header") or {}).get("msg_id") == msg_id:
            return msg


def _record_comm(session: NotebookSession, msg_type: str, content: dict[str, Any]) -> None:
    comm_id = content.get("comm_id")
    if not comm_id:
        return
    if msg_type == "comm_open":
        session.widget_comms[comm_id] = {
            "target_name": content.get("target_name"),
            "data": content.get("data"),
        }
    elif msg_type == "comm_msg":
        entry = session.widget_comms.setdefault(comm_id, {})
        entry["last_data"] = content.get("data")
    elif msg_type == "comm_close":
        session.widget_comms.pop(comm_id, None)


async def _snapshot_widget_state(session: NotebookSession) -> None:
    """Ask the kernel for its current ipywidgets manager state, write to nb.metadata."""
    kc = session.kc
    # NOTE: silent=True suppresses iopub stream output entirely, which would
    # hide our sentinel-wrapped stdout. Use store_history=False so the
    # helper doesn't pollute the user's history.
    msg_id = kc.execute(
        widgets.snapshot_request_code(),
        silent=False,
        store_history=False,
        allow_stdin=False,
    )
    collected: list[str] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + 5.0
    while True:
        if loop.time() > deadline:
            return
        try:
            msg = await asyncio.wait_for(kc.get_iopub_msg(), timeout=IOPUB_TIMEOUT_S)
        except asyncio.TimeoutError:
            if not await session.km.is_alive():
                return
            continue
        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue
        msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
        content = msg.get("content", {})
        if msg_type == "stream" and content.get("name") == "stdout":
            collected.append(content.get("text", ""))
        elif msg_type == "status" and content.get("execution_state") == "idle":
            break

    snapshot = widgets.parse_snapshot_stdout("".join(collected))
    if snapshot is not None:
        widgets.install_widget_state(session.nb.metadata, snapshot)
        session.writer.schedule()
