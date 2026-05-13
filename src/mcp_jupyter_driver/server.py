"""FastMCP server: thin @mcp.tool wrappers around the session/execution layer.

v1 walking-skeleton surface:
  - open_notebook / close_notebook / list_open_notebooks
  - list_cells / get_cell
  - run_cell (streaming progress)
  - kernel_status / interrupt_kernel / restart_kernel
"""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from . import execution, inspection, registry
from .session import NotebookSession

mcp = FastMCP("mcp-jupyter-driver")


# ----- response models -------------------------------------------------------


class NotebookHandle(BaseModel):
    path: str
    cell_count: int
    kernel_state: str
    busy: bool = False


class CellSummary(BaseModel):
    index: int
    cell_id: str | None
    type: Literal["code", "markdown", "raw"]
    source_preview: str
    exec_count: int | None
    output_count: int


class CellDetail(BaseModel):
    index: int
    cell_id: str | None
    type: Literal["code", "markdown", "raw"]
    source: str
    execution_count: int | None
    outputs: list[dict]


class RunResult(BaseModel):
    status: Literal["ok", "error", "kernel_died"]
    execution_count: int | None
    output_count: int
    has_widget: bool
    error_name: str | None = None
    error_value: str | None = None
    error_traceback: list[str] = Field(default_factory=list)
    truncated: bool = False
    interactive_input: bool = False
    kernel_restarted: bool = False


class OkResult(BaseModel):
    ok: bool = True


class CellRef(BaseModel):
    index: int
    cell_id: str | None = None


class ClearedResult(BaseModel):
    cleared_count: int


class RunCodeResult(BaseModel):
    cell_index: int
    cell_id: str | None
    persisted: bool
    run: "RunResult"


class VariableSummary(BaseModel):
    name: str
    type: str
    size_hint: str = ""
    repr_preview: str = ""


class VariableDetail(BaseModel):
    found: bool
    name: str
    type: str | None = None
    repr: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    columns: list[str] | None = None
    dtypes_per_column: dict[str, str] | None = None
    head: dict[str, list] | None = None
    length: int | None = None


class CompletionResult(BaseModel):
    matches: list[str]
    cursor_start: int
    cursor_end: int


# ----- helpers ---------------------------------------------------------------


async def _kernel_state_label(session: NotebookSession) -> str:
    try:
        alive = await session.km.is_alive()
    except Exception:
        return "unknown"
    return "alive" if alive else "dead"


def _source_preview(source: str, limit: int = 80) -> str:
    if not source:
        return ""
    flat = source.replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _summarize_cell(index: int, cell) -> CellSummary:
    return CellSummary(
        index=index,
        cell_id=cell.get("id"),
        type=cell.get("cell_type", "code"),
        source_preview=_source_preview(cell.get("source", "")),
        exec_count=cell.get("execution_count"),
        output_count=len(cell.get("outputs") or []),
    )


# ----- lifecycle -------------------------------------------------------------


@mcp.tool()
async def open_notebook(
    path: str, create_if_missing: bool = False
) -> NotebookHandle:
    """Open a notebook and start its kernel. Returns a handle.

    The kernel stays alive across tool calls in this Claude Code session
    until you call close_notebook or restart_kernel.
    """
    session = await registry.open_session(path, create_if_missing=create_if_missing)
    return NotebookHandle(
        path=str(session.path),
        cell_count=len(session.nb.cells),
        kernel_state=await _kernel_state_label(session),
        busy=session.exec_lock.locked(),
    )


@mcp.tool()
async def close_notebook(path: str) -> OkResult:
    """Shutdown the kernel and forget this notebook."""
    await registry.close_session(path)
    return OkResult()


@mcp.tool()
async def list_open_notebooks() -> list[NotebookHandle]:
    """List every notebook currently open in this session."""
    out: list[NotebookHandle] = []
    for s in registry.list_sessions():
        out.append(
            NotebookHandle(
                path=str(s.path),
                cell_count=len(s.nb.cells),
                kernel_state=await _kernel_state_label(s),
                busy=s.exec_lock.locked(),
            )
        )
    return out


# ----- cells -----------------------------------------------------------------


@mcp.tool()
async def list_cells(path: str) -> list[CellSummary]:
    """List cells in the open notebook: index, id, type, source preview, exec count."""
    session = registry.get_session(path)
    return [_summarize_cell(i, c) for i, c in enumerate(session.nb.cells)]


@mcp.tool()
async def get_cell(path: str, ref: int | str) -> CellDetail:
    """Full content of one cell, including outputs. `ref` is an index or cell id."""
    session = registry.get_session(path)
    idx = session.resolve_cell_index(ref)
    cell = session.nb.cells[idx]
    return CellDetail(
        index=idx,
        cell_id=cell.get("id"),
        type=cell.get("cell_type", "code"),
        source=cell.get("source", ""),
        execution_count=cell.get("execution_count"),
        outputs=list(cell.get("outputs") or []),
    )


# ----- cell editing ----------------------------------------------------------


@mcp.tool()
async def add_cell(
    path: str,
    cell_type: Literal["code", "markdown", "raw"],
    source: str,
    index: int | None = None,
) -> CellRef:
    """Insert a new cell. `index=None` appends to the end."""
    session = registry.get_session(path)
    async with session.exec_lock:
        idx, cid = session.add_cell(cell_type, source, index)
        session.writer.schedule()
        await session.writer.flush()
    return CellRef(index=idx, cell_id=cid)


@mcp.tool()
async def edit_cell(path: str, ref: int | str, source: str) -> CellRef:
    """Replace the source of an existing cell. Clears outputs/exec_count for code cells."""
    session = registry.get_session(path)
    async with session.exec_lock:
        idx, cid = session.edit_cell(ref, source)
        session.writer.schedule()
        await session.writer.flush()
    return CellRef(index=idx, cell_id=cid)


@mcp.tool()
async def delete_cell(path: str, ref: int | str) -> OkResult:
    """Remove a cell by index or id."""
    session = registry.get_session(path)
    async with session.exec_lock:
        session.delete_cell(ref)
        session.writer.schedule()
        await session.writer.flush()
    return OkResult()


@mcp.tool()
async def move_cell(path: str, from_ref: int | str, to_index: int) -> OkResult:
    """Reorder a cell to a new position."""
    session = registry.get_session(path)
    async with session.exec_lock:
        session.move_cell(from_ref, to_index)
        session.writer.schedule()
        await session.writer.flush()
    return OkResult()


@mcp.tool()
async def clear_cell_outputs(path: str, ref: int | str | None = None) -> ClearedResult:
    """Clear outputs and execution_count for one cell, or all code cells if ref is None."""
    session = registry.get_session(path)
    async with session.exec_lock:
        cleared = session.clear_outputs(ref)
        session.writer.schedule()
        await session.writer.flush()
    return ClearedResult(cleared_count=cleared)


# ----- execution -------------------------------------------------------------


def _run_result_to_model(result: execution.CellResult) -> "RunResult":
    return RunResult(
        status=result.status,
        execution_count=result.execution_count,
        output_count=result.output_count,
        has_widget=result.has_widget,
        error_name=result.error_name,
        error_value=result.error_value,
        error_traceback=result.error_traceback,
        truncated=result.truncated,
        interactive_input=result.interactive_input,
        kernel_restarted=result.kernel_restarted,
    )


@mcp.tool()
async def run_cell(
    path: str,
    ref: int | str,
    ctx: Context,
    timeout_s: float = 120.0,
    restart_on_kernel_death: bool = False,
) -> RunResult:
    """Execute a cell against this notebook's persistent kernel.

    Outputs stream into the .ipynb file as they're produced, so the user
    sees updates live in their editor. Returns a summary; the full outputs
    are in the file (or fetch them via get_cell).

    `restart_on_kernel_death`: if the kernel dies mid-execution, restart it
    and return a summary with `kernel_restarted=True` instead of erroring.
    """
    session = registry.get_session(path)

    async def _progress(progress: float, total: float | None, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
        except Exception:
            pass

    result = await execution.run_cell(
        session,
        ref,
        timeout_s=timeout_s,
        progress=_progress,
        restart_on_kernel_death=restart_on_kernel_death,
    )
    return _run_result_to_model(result)


@mcp.tool()
async def run_code(
    path: str,
    source: str,
    ctx: Context,
    persist_as_cell: bool = False,
    timeout_s: float = 120.0,
) -> RunCodeResult:
    """Append a code cell, run it, optionally remove it after.

    If `persist_as_cell` is False the cell is removed after execution — useful
    for one-off probes that you don't want to leave in the notebook.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        idx, cid = session.add_cell("code", source, None)
        session.writer.schedule()

    async def _progress(progress: float, total: float | None, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
        except Exception:
            pass

    result = await execution.run_cell(
        session, idx, timeout_s=timeout_s, progress=_progress
    )

    if not persist_as_cell:
        async with session.exec_lock:
            try:
                session.delete_cell(cid or idx)
            except Exception:
                # If indices shifted (no other writer should have, but be defensive),
                # leave the cell in place rather than deleting the wrong one.
                pass
            session.writer.schedule()
            await session.writer.flush()

    return RunCodeResult(
        cell_index=idx,
        cell_id=cid,
        persisted=persist_as_cell,
        run=_run_result_to_model(result),
    )


# ----- introspection ---------------------------------------------------------


@mcp.tool()
async def list_variables(
    path: str, include_private: bool = False
) -> list[VariableSummary]:
    """List user-defined variables in the live kernel.

    Hides `_mcp_*` helper names and dunders. `include_private` toggles
    single-underscore names.
    """
    session = registry.get_session(path)
    raw = await inspection.list_variables(session, include_private=include_private)
    return [VariableSummary(**v) for v in raw]


@mcp.tool()
async def inspect_variable(
    path: str, name: str, max_repr_len: int = 2000
) -> VariableDetail:
    """Inspect one variable in the live kernel.

    For pandas DataFrames you'll get columns, dtypes, and a 5-row head; for
    numpy/torch arrays you'll get shape + dtype.
    """
    session = registry.get_session(path)
    raw = await inspection.inspect_variable(
        session, name, max_repr_len=max_repr_len
    )
    return VariableDetail(**raw)


@mcp.tool()
async def complete(path: str, source: str, cursor_pos: int) -> CompletionResult:
    """Kernel-driven code completion at `cursor_pos` inside `source`."""
    session = registry.get_session(path)
    result = await inspection.complete(session, source, cursor_pos)
    return CompletionResult(**result)


# ----- kernel control --------------------------------------------------------


@mcp.tool()
async def kernel_status(path: str) -> NotebookHandle:
    """Report the kernel state for an open notebook."""
    session = registry.get_session(path)
    return NotebookHandle(
        path=str(session.path),
        cell_count=len(session.nb.cells),
        kernel_state=await _kernel_state_label(session),
        busy=session.exec_lock.locked(),
    )


@mcp.tool()
async def interrupt_kernel(path: str) -> OkResult:
    """Send SIGINT to the kernel (e.g. to stop a runaway cell)."""
    session = registry.get_session(path)
    await session.km.interrupt_kernel()
    return OkResult()


@mcp.tool()
async def restart_kernel(path: str, clear_outputs: bool = False) -> NotebookHandle:
    """Restart the kernel. If clear_outputs is true, also wipe cell outputs."""
    session = registry.get_session(path)
    async with session.exec_lock:
        await session.km.restart_kernel(now=False)
        # client stays valid across restart in jupyter_client v8
        if clear_outputs:
            for cell in session.nb.cells:
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
        session.writer.schedule()
        await session.writer.flush()
    return NotebookHandle(
        path=str(session.path),
        cell_count=len(session.nb.cells),
        kernel_state=await _kernel_state_label(session),
        busy=False,
    )
