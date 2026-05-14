# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                            # install deps (Python ≥ 3.10, uv required)
uv run pytest                                      # full suite (helpers + Jupyter-Server-backed integration)
uv run pytest tests/test_execution_helpers.py      # fast unit tests only — no kernel spawned
uv run pytest tests/test_integration_kernel.py -k name_of_test   # single integration test
uv run python -m mcp_jupyter_driver --self-check   # imports everything, prints registered tool names, exits
uv run python -m mcp_jupyter_driver                # run the MCP server on stdio (this is what Claude Code invokes)
```

Integration tests spawn a real `jupyter server` subprocess per test. They are slower and depend on `jupyter`, `ipykernel`, and `websockets` being on the venv path — `uv sync` covers all three.

## Architecture

This MCP server is unusual in that **it does not edit `.ipynb` files directly** and **it does not own kernels**. It supervises a single `jupyter server` subprocess that owns both, and exposes a FastMCP tool surface that delegates every file read/write and every cell execution to that server. The user's editor (VS Code → "Existing Jupyter Server") connects to the same server, so Claude and the user share one kernel per notebook and one writer for the file. This is the whole point — variables Claude sets are visible to the user, and vice versa, because they are literally the same kernel process.

Read in roughly this order to understand the system end-to-end:

1. `jserver.py` — `JupyterServer` class supervises the `jupyter server` subprocess (one per MCP process, module-level singleton via `get_or_start_server`). Picks a port (cached → `MCP_JUPYTER_PORT` → random), generates/persists a token in `~/.cache/mcp-jupyter-driver/`. Token and port are stable across MCP restarts so VS Code's saved "Existing Jupyter Server" entry keeps working.
2. `client.py` — `JupyterClient` wraps the server's REST APIs (Contents, Sessions, Kernels) and the kernel WebSocket. All notebook I/O goes through `read_notebook` / `write_notebook` (Contents API) — never direct file I/O.
3. `session.py` — `NotebookSession` is one per notebook path. It holds the `session_id`/`kernel_id` binding and an `exec_lock` that serializes Claude-driven mutations per notebook. Two methods drive the kernel-sharing behavior:
   - `maybe_rejoin()` — called before each execution/introspection tool. If a *live* "user" session exists for this notebook (matched by exact path, basename, or VS Code's `<stem>-jvsc-...` synthetic path) and its kernel differs from ours, switch to it. This is what makes Claude follow VS Code's kernel.
   - `rebind_kernel(target)` + `pinned` flag — explicit override that suppresses `maybe_rejoin`. Cleared by `unpin_kernel`. Don't remove the pinning behavior without thinking through it; users rely on `rebind_kernel` to override auto-rejoin's heuristics.
4. `registry.py` — module-level `dict[canonical_path -> NotebookSession]`. Also lazily owns the `JupyterClient` singleton.
5. `execution.py` — runs a cell over the kernel WebSocket, applies iopub messages to an in-memory notebook copy, and writes back through the Contents API (debounced during streaming output). Enforces per-output (1 MB) and per-cell (5 MB) byte caps; images and widget MIMEs are exempt. Auto-replies `""` to `input_request` so notebooks asking for stdin don't hang.
6. `inspection.py` — `list_variables` / `inspect_variable` / `complete`. Critically these read the **live kernel**, not the notebook file, so variables persist after a cell is deleted.
7. `widgets.py` — captures widget state into `nb.metadata.widgets` after cells that produce widget MIME, so VS Code/JupyterLab can rehydrate them.
8. `server.py` — the FastMCP `mcp` object, response Pydantic models, and the `@mcp.tool()` surface. Tools are thin: validate args, look up the session, dispatch into the modules above, return Pydantic models. New tools go here.

### Invariants to preserve

- **Single writer.** Only the Jupyter Server writes the `.ipynb`. Never `open(path, "w")` on a notebook file from MCP code — go through `session.write_notebook` (Contents API). Direct file writes would race the server.
- **Sync before mutate.** Every structural notebook mutation MUST go through `session.mutate_notebook_fresh(...)` (or, for the cell-output hot path, `session.patch_cell` / `delete_cell_by_id` / `move_cell_by_id`, which are thin wrappers over it). The helper re-reads the notebook from the Jupyter Server immediately before writing, optionally verifies `expected_cell_id` / `expected_source`, then writes. If a precondition fails, it raises `NotebookConflictError` (a `CellNotFoundError` is the specific "cell is gone" subclass) — **never** silently fall back to writing at the original index. New structural tools should resolve their `ref` to a stable cell id once, pass that id as `expected_cell_id`, and let the helper handle the re-read.
- **Auto-rejoin runs before kernel-touching tools.** Execution and introspection paths call `session.maybe_rejoin()` so Claude tracks whichever kernel VS Code is currently using. Don't bypass it on new kernel-facing tools unless you have a reason — and respect `session.pinned`.
- **`exec_lock` for any tool that writes the notebook.** Cell mutations (`add_cell`, `edit_cell`, …) and `run_cell` take `session.exec_lock` so concurrent *MCP* tool calls don't interleave partial writes. `exec_lock` does NOT protect against VS Code — that's what `mutate_notebook_fresh` is for.
- **`run_code` cleanup is gated by safety.** The temp cell is removed in a `finally` block, but only when the run completed normally and `kernel_still_running` is false. If the run timed out with the kernel still executing, or the run raised, the cell stays visible so the user can see what happened — and `RunCodeResult.persisted` reflects that.
- **VS Code does not auto-reload the notebook view.** Variable sharing through the kernel works regardless, but to *see* Claude's structural edits in VS Code, the user must run "File: Revert File". Document this when surfacing new editing tools; don't try to "fix" it with file-watch hacks.
- **Output caps live in `execution.py`.** `MAX_OUTPUT_BYTES` (1 MB) and `MAX_CELL_BYTES` (5 MB) are intentional. The trailing `_TRUNCATED` marker is what the `truncated=True` flag on `RunResult` corresponds to.

### Cache and environment

- `~/.cache/mcp-jupyter-driver/token` (mode 0600) and `connection.json` are persisted between launches. Delete `token` to rotate.
- `MCP_JUPYTER_CACHE_DIR` overrides the cache dir; `MCP_JUPYTER_PORT` overrides the preferred port (default 17077).
- The server runs with `root_dir=/`, so Contents API paths are absolute paths with the leading `/` stripped (`session.server_path`). Keep this in mind when constructing paths for new tools.
