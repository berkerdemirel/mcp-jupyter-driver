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
from .errors import CellNotFoundError, NotebookConflictError
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
    # True if the cell timed out but the kernel is still executing. The next
    # call will queue behind it — call interrupt_kernel first if you don't
    # want to wait.
    kernel_still_running: bool = False


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
    owned_by_claude: bool = False
    # How this session would be discovered if ``path`` was queried via
    # auto-rejoin: "exact_path", "vscode_synthetic", "basename", or
    # "no_match". Only set when the caller passed ``path`` to
    # ``list_jupyter_sessions``.
    match_reason: str = ""


class RebindResult(BaseModel):
    rebound: bool
    new_kernel_id: str | None = None
    new_session_id: str | None = None
    note: str = ""
    candidates: list[dict] = Field(default_factory=list)


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

    When ``path`` is given, every session is returned (no filtering) with a
    ``match_reason`` field describing how auto-rejoin sees it:

    - ``exact_path`` — server path equals the resolved ``server_relative``.
    - ``vscode_synthetic`` — VS Code's ``<stem>-jvsc-<uuid>-<uuid>.ipynb``.
    - ``basename`` — only basenames match (cross-directory; auto-rejoin
      uses this fallback only when it's unique).
    - ``no_match`` — auto-rejoin would ignore this session for ``path``.

    ``is_claudes`` is set by matching the MCP-side binding's session_id or
    kernel_id, not the path, so it stays correct after ``rebind_kernel``.
    ``owned_by_claude`` is set only for sessions Claude created (and may
    safely shut down).
    """
    client = await registry.get_client()
    sessions = await client.list_sessions()

    bound_session_ids = {s.session_id for s in registry.list_sessions()}
    bound_kernel_ids = {s.kernel_id for s in registry.list_sessions()}
    owned_session_ids: set[str] = set()
    for s in registry.list_sessions():
        owned_session_ids |= s.owned_session_ids

    match_reasons: dict[str, str] = {}
    if path is not None:
        from pathlib import Path as _P
        from .session import is_vscode_synthetic_path, server_path

        target = server_path(path)
        basename = _P(target).name
        stem = _P(basename).stem
        for s in sessions:
            sess_path = s.get("path") or ""
            name = _P(sess_path).name
            if sess_path == target:
                match_reasons[s["id"]] = "exact_path"
            elif is_vscode_synthetic_path(name, stem):
                match_reasons[s["id"]] = "vscode_synthetic"
            elif name == basename:
                match_reasons[s["id"]] = "basename"
            else:
                match_reasons[s["id"]] = "no_match"

    out: list[JupyterSessionInfo] = []
    for s in sessions:
        sess_path = s.get("path") or ""
        kid = s["kernel"]["id"]
        state = s["kernel"].get("execution_state", "unknown")
        sid = s["id"]
        out.append(
            JupyterSessionInfo(
                session_id=sid,
                kernel_id=kid,
                kernel_name=s["kernel"]["name"],
                path=sess_path,
                kernel_state=state,
                is_claudes=sid in bound_session_ids or kid in bound_kernel_ids,
                owned_by_claude=sid in owned_session_ids,
                match_reason=match_reasons.get(sid, ""),
            )
        )
    return out


@mcp.tool()
async def rebind_kernel(path: str, target: str) -> RebindResult:
    """Switch Claude's notebook binding to a specific kernel/session, and pin.

    `target` must be an exact session_id, an exact kernel_id, or a kernel_id
    prefix of at least 8 characters. Empty / overly-short prefixes are
    rejected, and an ambiguous match returns the candidate list so the caller
    can disambiguate. After a successful rebind, auto-rejoin will not undo
    your choice — call `unpin_kernel` to let it reconsider, or
    `rebind_kernel` again to pick a different one.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        old_kid = session.kernel_id
        outcome = await session.rebind_to_kernel(target)
    if not outcome.ok:
        notes = {
            "empty": "target must be a non-empty session_id, kernel_id, or kernel_id prefix.",
            "too_short": outcome.detail or "target prefix is too short; pass at least 8 characters.",
            "not_found": f"No session found matching {target!r}. Call list_jupyter_sessions to see what's available.",
            "ambiguous": f"Target {target!r} matched multiple sessions; pass a more specific id (see candidates).",
            "dead": "Target kernel exists but isn't alive. Restart it from VS Code or pick a different one.",
        }
        return RebindResult(
            rebound=False,
            note=notes.get(outcome.reason, ""),
            candidates=outcome.candidates,
        )
    if session.kernel_id == old_kid:
        return RebindResult(
            rebound=False,
            new_kernel_id=session.kernel_id,
            new_session_id=session.session_id,
            note="Already bound to this kernel (now pinned).",
        )
    return RebindResult(
        rebound=True,
        new_kernel_id=session.kernel_id,
        new_session_id=session.session_id,
        note=f"Switched from {old_kid[:8]} to {session.kernel_id[:8]} and pinned. Auto-rejoin will no longer move this binding until you unpin or rebind again.",
    )


@mcp.tool()
async def unpin_kernel(path: str) -> OkResult:
    """Allow auto-rejoin to move this notebook's kernel binding again.

    After `rebind_kernel`, the binding is pinned (auto-rejoin no-ops). Call
    this to re-enable the auto-rejoin behavior (e.g. so Claude can pick up
    a fresh kernel the user just attached in VS Code).
    """
    session = registry.get_session(path)
    session.unpin()
    return OkResult()


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
    """Insert a new cell. ``index=None`` appends to the end.

    The insertion happens on the freshest server-side notebook state — if
    the user added cells in VS Code since the last MCP read, those cells
    are preserved and the new cell lands relative to that fresher state.
    """
    session = registry.get_session(path)
    placed: dict = {}

    def _insert(nb: dict) -> None:
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source)
        else:
            cell = nbformat.v4.new_raw_cell(source=source)
        cells = nb.setdefault("cells", [])
        if index is None or index >= len(cells):
            cells.append(cell)
            placed["idx"] = len(cells) - 1
        else:
            i = max(0, index)
            cells.insert(i, cell)
            placed["idx"] = i
        placed["cell"] = cell

    async with session.exec_lock:
        await session.mutate_notebook_fresh(_insert, operation_name="add_cell")
    return CellRef(index=placed["idx"], cell_id=placed["cell"].get("id"))


@mcp.tool()
async def edit_cell(path: str, ref: int | str, source: str) -> CellRef:
    """Replace a cell's source. Clears outputs and exec_count for code cells.

    Resolves ``ref`` to a stable cell id from a fresh read, then re-reads
    immediately before writing and locates by id (raising
    ``NotebookConflictError`` if the cell has disappeared). A concurrent VS
    Code reorder can't make us edit the wrong cell, and a concurrent
    deletion produces a clear conflict instead of writing.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        cell_id = nb["cells"][idx].get("id")

        def _do(fresh_nb: dict) -> None:
            cells = fresh_nb.get("cells") or []
            target = None
            if cell_id is not None:
                for c in cells:
                    if c.get("id") == cell_id:
                        target = c
                        break
            else:
                if 0 <= idx < len(cells):
                    target = cells[idx]
            if target is None:
                raise CellNotFoundError(cell_id if cell_id is not None else idx)
            target["source"] = source
            if target.get("cell_type") == "code":
                target["outputs"] = []
                target["execution_count"] = None

        await session.mutate_notebook_fresh(
            _do,
            expected_cell_id=cell_id,
            operation_name="edit_cell",
        )
    return CellRef(index=idx, cell_id=cell_id)


@mcp.tool()
async def delete_cell(path: str, ref: int | str) -> OkResult:
    """Remove a cell by index or id, located by stable id at write time.

    Raises ``NotebookConflictError`` (catchable as such by the caller, or
    as ``CellNotFoundError`` for the specific "cell is gone" subclass) if
    the cell has disappeared between the ref resolution and the write.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, ref)
        cell_id = nb["cells"][idx].get("id")
        await session.delete_cell_by_id(cell_id, idx)
    return OkResult()


@mcp.tool()
async def move_cell(path: str, from_ref: int | str, to_index: int) -> OkResult:
    """Reorder a cell. Target is identified by stable id at write time."""
    session = registry.get_session(path)
    async with session.exec_lock:
        nb = await session.read_notebook()
        idx = resolve_cell_index(nb, from_ref)
        cell_id = nb["cells"][idx].get("id")
        await session.move_cell_by_id(cell_id, idx, to_index)
    return OkResult()


@mcp.tool()
async def clear_cell_outputs(
    path: str, ref: int | str | None = None
) -> ClearedResult:
    """Clear outputs for one cell, or all code cells if ref is None.

    Per-cell clears patch by stable cell id; clear-all also re-reads
    immediately before writing so concurrent VS Code source edits aren't
    reverted — we only ever zero the ``outputs`` and ``execution_count``
    fields on the freshest server-side cell objects.
    """
    session = registry.get_session(path)
    if ref is not None:
        async with session.exec_lock:
            nb = await session.read_notebook()
            idx = resolve_cell_index(nb, ref)
            target = nb["cells"][idx]
            cell_id = target.get("id")
            if target.get("cell_type") != "code" or not target.get("outputs"):
                return ClearedResult(cleared_count=0)

            def _apply(c: dict) -> None:
                c["outputs"] = []
                c["execution_count"] = None

            await session.patch_cell(
                cell_id=cell_id,
                fallback_index=idx if cell_id is None else None,
                mutate=_apply,
            )
        return ClearedResult(cleared_count=1)

    counter: dict[str, int] = {"n": 0}

    def _clear_all(nb: dict) -> None:
        for cell in nb.get("cells") or []:
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                cell["outputs"] = []
                cell["execution_count"] = None
                counter["n"] += 1

    async with session.exec_lock:
        await session.mutate_notebook_fresh(
            _clear_all, operation_name="clear_cell_outputs(all)"
        )
    return ClearedResult(cleared_count=counter["n"])


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
        kernel_still_running=result.kernel_still_running,
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
    """Append a code cell, run it, optionally remove it after.

    Cleanup of the temporary cell runs in a ``finally`` so it happens even
    if the run raises. The cleanup is skipped (the cell is kept visible) if
    the run timed out with ``kernel_still_running=True`` — deleting the
    cell while the kernel is still writing outputs to it would lose the
    follow-up output. ``RunCodeResult.persisted`` reflects whether the
    cell ended up persisted on disk so the caller can tell.
    """
    session = registry.get_session(path)
    placed: dict = {}

    def _append(nb: dict) -> None:
        cell = nbformat.v4.new_code_cell(source=source)
        nb.setdefault("cells", []).append(cell)
        placed["cell"] = cell
        placed["idx"] = len(nb["cells"]) - 1

    async with session.exec_lock:
        await session.mutate_notebook_fresh(_append, operation_name="run_code_append")
    cell_id = placed["cell"].get("id")
    idx = placed["idx"]

    async def _progress(progress: float, total: float | None, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
        except Exception:
            pass

    result: execution.CellResult | None = None
    try:
        result = await execution.run_cell(
            session, cell_id or idx, timeout_s=timeout_s, progress=_progress
        )
    finally:
        # Only delete the temp cell if we got a clean result and the kernel
        # isn't still running against it.
        safe_to_delete = (
            not persist_as_cell
            and result is not None
            and not result.kernel_still_running
        )
        if safe_to_delete:
            async with session.exec_lock:
                try:
                    await session.delete_cell_by_id(cell_id, idx)
                except NotebookConflictError:
                    # User already removed the cell; nothing to clean up.
                    pass

    final_persisted = persist_as_cell or (
        result is not None and result.kernel_still_running
    )
    return RunCodeResult(
        cell_index=idx,
        cell_id=cell_id,
        persisted=final_persisted,
        run=_run_result_to_model(result) if result is not None else _run_result_to_model(
            execution.CellResult(status="error", execution_count=None)
        ),
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
    """Status of the shared kernel for this notebook.

    Calls ``maybe_rejoin`` first so the reported binding reflects what the
    next ``run_cell`` would use — without this, status could report Claude's
    stored kernel right before execution silently switches to VS Code's.
    """
    session = registry.get_session(path)
    try:
        await session.maybe_rejoin()
    except Exception:
        # Auto-rejoin is best-effort; never block status on its failure.
        pass
    return await _to_handle(session)


@mcp.tool()
async def interrupt_kernel(path: str) -> OkResult:
    """SIGINT the shared kernel (stop a runaway cell)."""
    session = registry.get_session(path)
    await session.client.interrupt_kernel(session.kernel_id)
    return OkResult()


@mcp.tool()
async def restart_kernel(path: str, clear_outputs: bool = False) -> NotebookHandle:
    """Restart the shared kernel and refresh local binding.

    Also unpins the binding so auto-rejoin can move it again if the user
    re-attaches a different kernel in VS Code; restart is a clean slate.
    If ``clear_outputs`` is true, wipe cell outputs too.
    """
    session = registry.get_session(path)
    async with session.exec_lock:
        info = await session.client.restart_kernel(session.kernel_id)
        # Restart usually returns the same kernel id, but pick up whatever
        # the server reports rather than trusting our cached value.
        if isinstance(info, dict):
            kid = info.get("id")
            if isinstance(kid, str) and kid:
                session.kernel_id = kid
            kname = info.get("name")
            if isinstance(kname, str) and kname:
                session.kernel_name = kname
        session.unpin()
        if clear_outputs:
            def _clear(nb: dict) -> None:
                for cell in nb.get("cells") or []:
                    if cell.get("cell_type") == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None

            await session.mutate_notebook_fresh(
                _clear, operation_name="restart_kernel(clear_outputs)"
            )
    return await _to_handle(session)
