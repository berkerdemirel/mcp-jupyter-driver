# mcp-jupyter-driver

An MCP server that lets Claude and you **co-edit a Jupyter notebook against
the same kernel**. Variables you set are visible to Claude; cells Claude
runs are visible to you. Variables persist in the kernel even if the cell
that defined them is deleted, because the kernel is shared and decoupled
from the notebook file.

## Notebook view refresh after Claude writes

Cell outputs and structural edits Claude makes go through the Jupyter
Server's Contents API and are written to disk. VS Code's notebook editor
holds its own in-memory copy and does **not** auto-reload on external file
changes — so to *see* Claude's updates, run **"File: Revert File"** (or
"Notebook: Revert") from the command palette.

(Variable sharing through the kernel works regardless — that's the more
fundamental property of the shared-server architecture, below. Use
`list_variables` / `inspect_variable` to see what's in the kernel without
needing the notebook UI in sync.)

`jupyter-collaboration` would give true live sync via Y.js, but at present
its WebSocket flow conflicts with VS Code's Jupyter extension's cell
execution path, leaving cells stuck. We may revisit when VS Code's
collaboration support matures.

## How it works

The MCP supervises a `jupyter server` subprocess on localhost. Both Claude
(through the MCP) and your editor (VS Code → "Existing Jupyter Server")
connect to that single server, share its kernels, and read/write the same
`.ipynb` via the server's Contents API. There is exactly one writer for the
file (the server), so there are no save conflicts.

Architecturally:

```
                  ┌──────────────────────┐
   Claude  ───►   │   mcp-jupyter-driver │
                  │   (MCP server)        │
                  └─────────┬─────────────┘
                            │ REST + WebSocket
                            ▼
                  ┌──────────────────────┐
                  │   jupyter server      │  ◄── VS Code
                  │   (subprocess)        │      "Existing Jupyter Server"
                  │   - Contents API      │
                  │   - Kernels API       │
                  │   - WebSocket channels│
                  └─────────┬─────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │   ipykernel(s)        │
                  └──────────────────────┘
```

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> mcp-jupyter-driver
cd mcp-jupyter-driver
uv sync
uv run python -m mcp_jupyter_driver --self-check
```

Register with Claude Code — add to `~/.claude.json` under the right
`mcpServers` object (project-scoped or user-scoped):

```json
{
  "jupyter": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "--directory", "/absolute/path/to/mcp-jupyter-driver",
      "run", "python", "-m", "mcp_jupyter_driver"
    ]
  }
}
```

Restart Claude Code. `/mcp` should list `jupyter` with all tools.

## Use your real Python environment for the kernel

By default the kernel runs in the `python3` kernelspec, which is the MCP's
own uv-managed venv (minimal packages). To run your data-science stack
(pandas, torch, sklearn, etc.) install `ipykernel` in *your* env and register
it once:

```bash
# from inside your conda/uv/venv environment
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "myenv"
```

Then when opening a notebook from Claude, pass `kernel_name="myenv"`:

> "Open `/path/to/work.ipynb` with kernel `myenv`."

List available kernels via the `list_kernelspecs` tool.

## Connect VS Code to the same server (one-time setup)

1. Start a Claude Code session with this MCP enabled.
2. Ask Claude to call `jupyter_server_info`. It returns `url_with_token`,
   a single string of the form `http://127.0.0.1:<port>/?token=<token>`.
3. In VS Code, open the `.ipynb` you want to work on (notebook view).
4. Top right, click the kernel picker → **"Select Another Kernel..."** →
   **"Existing Jupyter Server..."** → paste `url_with_token` into the URL
   box (one string — the token is embedded as a query parameter, which is
   what VS Code expects). Give the server any nickname.
5. After it connects, you'll see kernels listed. Pick the one Claude is
   using (or pick any — VS Code will route the notebook to a kernel from
   this server, and `open_notebook` from Claude will reuse the same session
   if you pass the same path).

If VS Code warns about an insecure server, that means the URL is missing
the `?token=...` suffix. Make sure you pasted `url_with_token`, not just
`url`.

After this, runs you trigger from VS Code's UI and runs Claude triggers via
the MCP both hit the same kernel. Variables flow between you.

The URL+token are also written to `~/.cache/mcp-jupyter-driver/connection.json`
for convenience.

### URL+token are stable across MCP restarts

The token is persisted to `~/.cache/mcp-jupyter-driver/token` (mode 0600) and
reused on every launch. The port is preferred-then-cached: each launch first
tries the prior port, then `MCP_JUPYTER_PORT` (default 17077), then a random
free port — and writes the actually-bound port back to the cache. So once
you've added the server to VS Code's "Existing Jupyter Server" list, the
same entry keeps working across Claude restarts.

To rotate the token (e.g., if it ever leaks):

```bash
rm -f ~/.cache/mcp-jupyter-driver/token
# next MCP launch generates a fresh one; you'll re-paste once into VS Code.
```

Override via env vars if you need to:

- `MCP_JUPYTER_CACHE_DIR` — where token + connection.json live (default `~/.cache/mcp-jupyter-driver`).
- `MCP_JUPYTER_PORT` — preferred port to try first (default `17077`).

## Tool surface

**Server / lifecycle**

| Tool | What it does |
|---|---|
| `jupyter_server_info()` | Returns the URL + token for the local Jupyter Server. Paste into VS Code's "Existing Jupyter Server" dialog. |
| `list_kernelspecs()` | Kernelspec names available on the server. |
| `open_notebook(path, create_if_missing=False, kernel_name="python3")` | Open a notebook + bind a session/kernel. Uses the server's Contents API. |
| `close_notebook(path, shutdown_kernel=True)` | Close the session. |
| `list_open_notebooks()` | All notebooks open in this MCP session. |
| `refresh_notebook(path)` | Force-refresh handle (every tool already re-reads on entry). |

**Cell editing** (all go through the server's Contents API)

| Tool | What it does |
|---|---|
| `add_cell(path, cell_type, source, index=None)` | Insert a new code/markdown/raw cell. |
| `edit_cell(path, ref, source)` | Replace a cell's source. Clears outputs for code cells. |
| `delete_cell(path, ref)` | Remove a cell. |
| `move_cell(path, from_ref, to_index)` | Reorder. |
| `clear_cell_outputs(path, ref=None)` | Clear outputs (one cell or all). |
| `list_cells(path)` | Index, id, type, source preview, exec count. |
| `get_cell(path, ref)` | Full source + outputs. |

**Execution** (over kernel WebSocket — shared with VS Code)

| Tool | What it does |
|---|---|
| `run_cell(path, ref, timeout_s=120, restart_on_kernel_death=False)` | Execute a cell. Outputs stream into the file via the server. |
| `run_code(path, source, persist_as_cell=False, timeout_s=120)` | Append-and-run; optionally remove the cell after. |

**Introspection** (live kernel state, decoupled from notebook contents)

| Tool | What it does |
|---|---|
| `list_variables(path, include_private=False)` | User variables in the live kernel. **Survives cell deletion** — variables live in the kernel, not the notebook. |
| `inspect_variable(path, name, max_repr_len=2000)` | Deep inspect (pandas: columns/dtypes/head; numpy: shape/dtype). |
| `complete(path, source, cursor_pos)` | Kernel-driven completion. |

**Kernel control**

| Tool | What it does |
|---|---|
| `kernel_status(path)` | Is the kernel alive? Busy? |
| `interrupt_kernel(path)` | SIGINT the kernel. |
| `restart_kernel(path, clear_outputs=False)` | Restart. |

**Kernel sharing diagnostics**

| Tool | What it does |
|---|---|
| `list_jupyter_sessions(path=None)` | All sessions/kernels on the local server. `is_claudes` flags the one Claude is bound to. Use this when you suspect you and Claude are on different kernels. |
| `rebind_kernel(path, target)` | Point Claude's notebook session at a specific kernel. `target` can be a kernel_id, session_id, or kernel_id prefix (8 chars). |

Claude also **auto-rejoins** your kernel before each `run_cell` / `list_variables` /
etc. if it detects a live session for the same notebook (matched by exact
path or by basename, since VS Code and Jupyter Server sometimes disagree on
path encoding).

## A typical workflow

1. In VS Code: open your notebook in notebook view.
2. In a Claude Code session (with this MCP active), ask:
   > Get the Jupyter server info.
   Claude calls `jupyter_server_info` and tells you the URL+token.
3. In VS Code: connect to "Existing Jupyter Server" with that URL+token. Pick
   a kernel (or your registered `myenv`).
4. From Claude, open the same notebook:
   > Open `/path/to/work.ipynb` with kernel `myenv`.
5. Drive it together. Examples:
   - You add a cell that loads an image, run it from VS Code.
   - Ask Claude: *"Add a cell that shows the histogram of that image."* —
     Claude reads the live notebook, adds a cell, runs it against the same
     kernel.
   - You add a cell that flips the image, run it. The variable update lives
     in the shared kernel; ask Claude to inspect it.
   - Delete a cell in VS Code. Variables it created stay alive. Claude can
     still use them.

## Failure-mode behavior

- **Per-output cap 1 MB, per-cell total 5 MB** — text/JSON outputs over the cap
  get truncated with a marker. Images and widget MIME aren't capped.
- **`input()` doesn't hang** — the MCP auto-replies with `""` and sets
  `interactive_input=True` on the result.
- **Kernel death** — `run_cell(restart_on_kernel_death=True)` recovers; otherwise raises.
- **ipywidgets** — the widget MIME is preserved and a state snapshot is
  written into `nb.metadata.widgets` after each widget-producing cell so
  VS Code/JupyterLab can render the live widget.

## Development

```bash
uv sync
uv run pytest          # 20 tests (helpers + server-backed integration)
uv run python -m mcp_jupyter_driver --self-check
```

## Security

The Jupyter Server we host listens on `127.0.0.1` only and uses a random
token per launch. Anyone with the token has full Contents-API access to your
filesystem and can execute arbitrary code through the kernel. Don't share
the token. Don't run this as root.
