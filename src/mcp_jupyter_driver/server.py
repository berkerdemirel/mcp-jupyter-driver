"""FastMCP server: tool surface over the Jupyter-Server-hosted notebooks.

All notebook reads/writes go through the Jupyter Server's Contents API, and
all kernel work runs against kernels the same server manages. Your editor
(VS Code → "Existing Jupyter Server") points at the same server and shares
both the kernel and the file with Claude.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import nbformat
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from . import execution, inspection, registry
from .jserver import get_or_start_server, stop_server
from .session import NotebookSession, resolve_cell_index


@asynccontextmanager
async def _lifespan(app) -> AsyncIterator[None]:
    """Shutdown hook: close open sessions and tear down the Jupyter Server."""
    try:
        yield
    finally:
        try:
            await registry.close_all()
        finally:
            await stop_server()


mcp = FastMCP("mcp-jupyter-driver", lifespan=_lifespan)


# ----- response models -------------------------------------------------------


class ServerInfo(BaseModel):
    url: str
    token: str
    url_with_token: str
    notes: str = (
        "VS Code: kernel picker → \"Select Another Kernel...\" → \"Existing "
        "Jupyter Server...\" → paste `url_with_token` into the URL box (one "
        "string, token already embedded). Give it any nickname. Then pick a "
        "kernel from this server in the notebook."
    )


class NotebookHandle(BaseModel):
    path: str
    cell_count: int
    kernel_id: str
    kernel_name: str
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
    run: RunResult


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


class JupyterSessionInfo(BaseModel):
    session_id: str
    kernel_id: str
    kernel_name: str
    path: str
    kernel_state: str
    is_claudes: bool = False


class RebindResult(BaseModel):
    rebound: bool
    new_kernel_id: str | None = None
    new_session_id: str | None = None
    note: str = ""


# ----- helpers ---------------------------------------------------------------


def _src_preview(source: str, limit: int = 80) -> str:
    if not source:
        return ""
    flat = source.replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _summarize_cell(index: int, cell: dict) -> CellSummary:
    return CellSummary(
        index=index,
        cell_id=cell.get("id"),
        type=cell.get("cell_type", "code"),
        source_preview=_src_preview(_str(cell.get("source", ""))),
        exec_count=cell.get("execution_count"),
        output_count=len(cell.get("outputs") or []),
    )


def _str(source) -> str:
    """nbformat stores source as either str or list-of-str."""
    if isinstance(source, list):
        return "".join(source)
    return source or ""


async def _to_handle(session: NotebookSession, nb: dict | None = None) -> NotebookHandle:
    if nb is None:
        nb = await session.read_notebook()
    return NotebookHandle(
        path=session.canonical,
        cell_count=len(nb.get("cells") or []),
        kernel_id=session.kernel_id,
        kernel_name=session.kernel_name,
        kernel_state=await session.kernel_state(),
        busy=session.exec_lock.locked(),
    )


# ----- server info -----------------------------------------------------------


@mcp.tool()
async def jupyter_server_info() -> ServerInfo:
    """Return the URL + token of the Jupyter Server this MCP is hosting.

    Paste `url_with_token` into VS Code's "Existing Jupyter Server" dialog —
    it's a single string with the token embedded as a query parameter, which
    is what VS Code (and Jupyter clients in general) expect.
    """
    srv = await get_or_start_server()
    return ServerInfo(
        url=srv.url,
        token=srv.token,
        url_with_token=f"{srv.url}/?token={srv.token}",
    )


# ----- lifecycle -------------------------------------------------------------


@mcp.tool()
async def open_notebook(
    path: str,
    create_if_missing: bool = False,
    kernel_name: str = "python3",
) -> NotebookHandle:
    """Open a notebook and start a server-managed kernel session for it.

    `kernel_name` is a kernelspec name (e.g. "python3", or a user-registered
    name like "myenv"). List them via list_kernelspecs.
    """
    session = await registry.open_session(
        path, create_if_missing=create_if_missing, kernel_name=kernel_name
    )
    return await _to_handle(session)


@mcp.tool()
async def close_notebook(path: str, shutdown_kernel: bool = True) -> OkResult:
    """End this MCP's session for the notebook. By default shuts down the kernel."""
    await registry.close_session(path, shutdown_kernel=shutdown_kernel)
    return OkResult()


@mcp.tool()
async def list_open_notebooks() -> list[NotebookHandle]:
    """All notebooks opened in this Claude Code session."""
    out: list[NotebookHandle] = []
    for s in registry.list_sessions():
        out.append(await _to_handle(s))
    return out


@mcp.tool()
async def list_kernelspecs() -> list[str]:
    """Available kernelspec names on this Jupyter Server.

    Register your conda/uv environment as a kernel via
    `python -m ipykernel install --user --name myenv` and it will appear here.
    """
    client = await registry.get_client()
    data = await client.list_kernelspecs()
    return sorted((data.get("kernelspecs") or {}).keys())


@mcp.tool()
async def list_jupyter_sessions(path: str | None = None) -> list[JupyterSessionInfo]:
    """List every session/kernel currently running on the local Jupyter Server.

    If `path` is given, only sessions for that notebook path. Useful for
    diagnosing kernel-sharing issues: if you and Claude see different
    kernel_ids for the same notebook, you're not sharing state. Fix with
    `rebind_kernel` or by picking the running kernel in VS Code's picker.
    """
    client = await registry.get_client()
    sessions = await client.list_sessions()
    out: list[JupyterSessionInfo] = []
    # Build a map of (path -> claude's bound kernel_id) so we can mark our own.
    claudes_bindings: dict[str, str] = {
        s.server_relative: s.kernel_id for s in registry.list_sessions()
    }
    for s in sessions:
        sess_path = s.get("path") or ""
        if path is not None:
            from .session import server_path

            if sess_path != server_path(path):
                continue
        kid = s["kernel"]["id"]
        state = s["kernel"].get("execution_state", "unknown")
        out.append(
            JupyterSessionInfo(
                session_id=s["id"],
                kernel_id=kid,
                kernel_name=s["kernel"]["name"],
                path=sess_path,
                kernel_state=state,
                is_claudes=claudes_bindings.get(sess_path) == kid,
            )
        )
    return out


@mcp.tool()
async def rebind_kernel(path: str, target: str) -> RebindResult:
    """Switch Claude's notebook binding to a specific kernel/session.

    `target` can be a session_id, a kernel_id, or a kernel_id prefix (8 chars
    is plenty). Use this after `list_jupyter_sessions` reveals that you and
    Claude are on different kernels for the same notebook.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        old_kid = session.kernel_id
        ok = await session.rebind_to_kernel(target)
    if not ok:
        return RebindResult(
            rebound=False,
            note=f"No session found matching {target!r}. Call list_jupyter_sessions to see what's available.",
        )
    if session.kernel_id == old_kid:
        return RebindResult(
            rebound=False,
            new_kernel_id=session.kernel_id,
            new_session_id=session.session_id,
            note="Already bound to this kernel.",
        )
    return RebindResult(
        rebound=True,
        new_kernel_id=session.kernel_id,
        new_session_id=session.session_id,
        note=f"Switched from {old_kid[:8]} to {session.kernel_id[:8]}. Variables from the previous kernel are gone; you're now seeing the shared kernel's state.",
    )


@mcp.tool()
async def refresh_notebook(path: str) -> NotebookHandle:
    """Force-refresh: re-read the notebook from disk via the Jupyter Server.

    Every tool already reads fresh on entry; this is a no-op convenience tool
    for confirming server-side state. Returns the latest handle.
    """
    session = registry.get_session(path)
    return await _to_handle(session)


# ----- cells -----------------------------------------------------------------


@mcp.tool()
async def list_cells(path: str) -> list[CellSummary]:
    """List cells with index, id, type, source preview, exec count."""
    session = registry.get_session(path)
    nb = await session.read_notebook()
    return [_summarize_cell(i, c) for i, c in enumerate(nb.get("cells") or [])]


@mcp.tool()
async def get_cell(path: str, ref: int | str) -> CellDetail:
    """Full content of one cell, including outputs. `ref` is an index or cell id."""
    session = registry.get_session(path)
    nb = await session.read_notebook()
    idx = resolve_cell_index(nb, ref)
    cell = (nb.get("cells") or [])[idx]
    return CellDetail(
        index=idx,
        cell_id=cell.get("id"),
        type=cell.get("cell_type", "code"),
        source=_str(cell.get("source", "")),
        execution_count=cell.get("execution_count"),
        outputs=list(cell.get("outputs") or []),
    )


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
        nb = await session.read_notebook()
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source)
        else:
            cell = nbformat.v4.new_raw_cell(source=source)
        cells = nb.setdefault("cells", [])
        if index is None or index >= len(cells):
            cells.append(cell)
            idx = len(cells) - 1
        else:
            idx = max(0, index)
            cells.insert(idx, cell)
        await session.write_notebook(nb)
    return CellRef(index=idx, cell_id=cell.get("id"))


@mcp.tool()
async def edit_cell(path: str, ref: int | str, source: str) -> CellRef:
    """Replace a cell's source. Clears outputs and exec_count for code cells."""
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        cell = nb["cells"][idx]
        cell["source"] = source
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        await session.write_notebook(nb)
    return CellRef(index=idx, cell_id=cell.get("id"))


@mcp.tool()
async def delete_cell(path: str, ref: int | str) -> OkResult:
    """Remove a cell by index or id."""
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        del nb["cells"][idx]
        await session.write_notebook(nb)
    return OkResult()


@mcp.tool()
async def move_cell(path: str, from_ref: int | str, to_index: int) -> OkResult:
    """Reorder a cell."""
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, from_ref)
        cell = nb["cells"].pop(idx)
        to_idx = max(0, min(to_index, len(nb["cells"])))
        nb["cells"].insert(to_idx, cell)
        await session.write_notebook(nb)
    return OkResult()


@mcp.tool()
async def clear_cell_outputs(
    path: str, ref: int | str | None = None
) -> ClearedResult:
    """Clear outputs for one cell, or all code cells if ref is None."""
    session = registry.get_session(path)
    cleared = 0
    async with session.exec_lock:
        nb = await session.read_notebook()
        if ref is None:
            for cell in nb.get("cells") or []:
                if cell.get("cell_type") == "code" and cell.get("outputs"):
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cleared += 1
        else:
            idx = resolve_cell_index(nb, ref)
            cell = nb["cells"][idx]
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                cell["outputs"] = []
                cell["execution_count"] = None
                cleared = 1
        await session.write_notebook(nb)
    return ClearedResult(cleared_count=cleared)


# ----- execution -------------------------------------------------------------


def _run_result_to_model(result: execution.CellResult) -> RunResult:
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
    """Execute a cell against the shared kernel.

    Outputs stream into the .ipynb (via the Jupyter Server) so VS Code sees
    them live. Variables persist across calls because the kernel is shared.
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
    """Append a code cell, run it, optionally remove it after."""
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        cell = nbformat.v4.new_code_cell(source=source)
        nb["cells"].append(cell)
        idx = len(nb["cells"]) - 1
        cell_id = cell.get("id")
        await session.write_notebook(nb)

    async def _progress(progress: float, total: float | None, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
        except Exception:
            pass

    result = await execution.run_cell(
        session, cell_id or idx, timeout_s=timeout_s, progress=_progress
    )

    if not persist_as_cell:
        async with session.exec_lock:
            nb = await session.read_notebook()
            try:
                ridx = resolve_cell_index(nb, cell_id or idx)
                del nb["cells"][ridx]
                await session.write_notebook(nb)
            except Exception:
                pass

    return RunCodeResult(
        cell_index=idx,
        cell_id=cell_id,
        persisted=persist_as_cell,
        run=_run_result_to_model(result),
    )


# ----- introspection ---------------------------------------------------------


@mcp.tool()
async def list_variables(
    path: str, include_private: bool = False
) -> list[VariableSummary]:
    """List user-defined variables in the live (shared) kernel."""
    session = registry.get_session(path)
    raw = await inspection.list_variables(session, include_private=include_private)
    return [VariableSummary(**v) for v in raw]


@mcp.tool()
async def inspect_variable(
    path: str, name: str, max_repr_len: int = 2000
) -> VariableDetail:
    """Inspect one variable in the live kernel.

    Picks up shape+dtype for numpy/torch arrays, columns+head for pandas
    DataFrames.
    """
    session = registry.get_session(path)
    raw = await inspection.inspect_variable(
        session, name, max_repr_len=max_repr_len
    )
    return VariableDetail(**raw)


@mcp.tool()
async def complete(path: str, source: str, cursor_pos: int) -> CompletionResult:
    """Kernel-driven completion."""
    session = registry.get_session(path)
    return CompletionResult(**(await inspection.complete(session, source, cursor_pos)))


# ----- kernel control --------------------------------------------------------


@mcp.tool()
async def kernel_status(path: str) -> NotebookHandle:
    """Status of the shared kernel for this notebook."""
    session = registry.get_session(path)
    return await _to_handle(session)


@mcp.tool()
async def interrupt_kernel(path: str) -> OkResult:
    """SIGINT the shared kernel (stop a runaway cell)."""
    session = registry.get_session(path)
    await session.client.interrupt_kernel(session.kernel_id)
    return OkResult()


@mcp.tool()
async def restart_kernel(path: str, clear_outputs: bool = False) -> NotebookHandle:
    """Restart the shared kernel. If clear_outputs is true, wipe cell outputs too."""
    session = registry.get_session(path)
    async with session.exec_lock:
        await session.client.restart_kernel(session.kernel_id)
        if clear_outputs:
            nb = await session.read_notebook()
            for cell in nb.get("cells") or []:
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            await session.write_notebook(nb)
    return await _to_handle(session)
