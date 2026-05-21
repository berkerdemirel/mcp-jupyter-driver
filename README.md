# mcp-jupyter-driver

An MCP server that lets Claude and you **co-edit a Jupyter notebook against
the same kernel**. Variables you set are visible to Claude; cells Claude
runs are visible to you. Variables persist in the kernel even if the cell
that defined them is deleted, because the kernel is shared and decoupled
from the notebook file.

## Notebook view refresh after Claude writes

Cell outputs and structural edits Claude makes go through the Jupyter
Server's Contents API and are written to disk. VS Code's notebook editor
auto-reloads from disk **whenever its in-memory copy isn't dirty** — so
in practice, just save your work (`Ctrl/Cmd+S`) after running cells or
typing in a cell, and Claude's subsequent writes appear automatically.

The only state where reload pauses is "your in-memory copy has unsaved
changes" — VS Code rightly won't clobber unsaved work. Save (or undo)
and the next external write from Claude appears.

For users who don't want to remember to save, the **companion VS Code
extension** in `vscode-extension/` watches `.ipynb` files and, on
external change, reverts clean notebooks automatically or surfaces a
status-bar nudge ("Claude updated `<name>` — click to reload") when
your copy is dirty. It's plain JavaScript with no build step — symlink
the folder into `~/.vscode/extensions/` and restart VS Code. See
[`vscode-extension/README.md`](vscode-extension/README.md) for details.

(Variable sharing through the kernel works regardless of the notebook
view — that's the more fundamental property of the shared-server
architecture, below. Use `list_variables` / `inspect_variable` to see
what's in the kernel without needing the notebook UI in sync.)

`jupyter-collaboration` would give true CRDT-based live sync via Y.js,
but at present its WebSocket flow conflicts with VS Code's Jupyter
extension's cell execution path, leaving cells stuck. The save-then-
reload behavior plus the optional companion extension cover the same
ground without the Yjs dependency.

## Claude sees what you do (live awareness)

The MCP keeps a long-lived **iopub subscriber** on each notebook's shared
kernel — read-only, so it doesn't interfere with anyone's runs — and a
running snapshot of the notebook structure. When you run a cell from VS
Code or edit/delete/move cells, Claude can ask for a summary:

```
recent_user_activity(path) → UserActivity {
  cell_changes: [added | removed | edited | moved …],
  executions:   [{ code, outputs_preview, by_claude=False, … }, …],
  …
}
```

The execution log is attributed via the originating Jupyter session id —
runs Claude triggered show `by_claude=True`, runs you triggered show
`by_claude=False`. By default the tool only returns your runs (pass
`include_claude=True` to also see Claude's). This is the channel for "I
want to check things in the notebook without prompting Claude, but I
want Claude to be aware of what I checked." Claude can poll this between
prompts to stay synced.

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

The kernel choice is owned by VS Code in the recommended workflow:
register your environment as a kernelspec once, pick it in VS Code's
kernel picker, and Claude inherits it via `find_existing_session_for_path`
on the next `open_notebook` call.

```bash
# from inside your conda/uv/venv environment, one time
pip install ipykernel
python -m ipykernel install --user --name myenv --display-name "myenv"
```

Then in VS Code: open the notebook → kernel picker (top right) → pick
`myenv` under the running server. Now ask Claude to open the same
notebook — Claude attaches to that kernel automatically. `list_kernelspecs`
lists what's registered if you want to confirm.

If you instead open the notebook from Claude *first* (Flow A — no VS
Code session exists yet), pass `kernel_name="myenv"` on `open_notebook`
to make Claude start `myenv` itself; auto-rejoin will still converge
once you connect VS Code.

## Connect VS Code to the supervised Jupyter Server (one-time setup)

The Jupyter Server here is shared — neither "Claude's" nor "VS Code's".
The MCP supervises it as a subprocess; Claude and your editor are both
clients of it via standard Jupyter REST + WebSocket. Bootstrapping
happens via Claude because the MCP starts the server on first tool
call, but once it's running both sides connect on equal footing.

1. Launch Claude Code with this MCP enabled. The MCP starts the
   supervised Jupyter Server on first tool call.
2. Ask Claude to call `jupyter_server_info`. It returns `url_with_token`,
   a single string of the form `http://127.0.0.1:<port>/?token=<token>`.
3. In VS Code, open the `.ipynb` you want to work on (notebook view).
4. Top right, click the kernel picker → **"Select Another Kernel..."** →
   **"Existing Jupyter Server..."** → paste `url_with_token` into the URL
   box (one string — the token is embedded as a query parameter, which
   is what VS Code expects). Give the server any nickname.
5. After it connects, pick the kernel you want from the picker
   (your registered `myenv`, or any other kernelspec on this server).
   The next time you ask Claude to `open_notebook` on this path, Claude
   attaches to the session you just created — same kernel, shared
   variables.

If VS Code warns about an insecure server, that means the URL is missing
the `?token=...` suffix. Make sure you pasted `url_with_token`, not just
`url`.

After this, runs you trigger from VS Code's UI and runs Claude triggers
via the MCP both hit the same kernel. Variables flow between you.

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
| `close_notebook(path, shutdown_kernel=True)` | Close Claude's binding for this notebook. With `shutdown_kernel=True` we only DELETE sessions Claude created — sessions VS Code (or any other client) owns are left alive, so closing Claude's notebook can never shut down the user's kernel. |
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

**Awareness** (push-style: what the user has been doing)

| Tool | What it does |
|---|---|
| `recent_user_activity(path, since=None, include_claude=False)` | Iopub-tap'd executions plus cell-level diff since the last call. Use this between prompts so Claude picks up what you did in VS Code without you having to narrate it. |

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

Claude **attaches to your kernel automatically** in two places:

- **At `open_notebook` time**, `find_existing_session_for_path` looks for a
  live session for this notebook using three tiers — exact path, VS Code's
  synthetic `<stem>-jvsc-<uuid>-<uuid>.ipynb`, and unique-basename fallback
  — and attaches with `owns_session=False` (so `close_notebook` never
  shuts your kernel down).
- **Before every kernel-touching tool** (`run_cell`, `list_variables`,
  `kernel_status`, etc.), `maybe_rejoin` re-checks the same tiers and
  switches if a better match exists. Once attached to a user-owned
  session, the binding is **sticky** — subsequent `maybe_rejoin` calls
  won't bounce back to Claude's original throwaway kernel even though
  it's still alive at the exact path.

Path matching normalizes leading slashes on both sides, and Claude's own
session is excluded from the tier lists so it can't gate the fallbacks.

## A typical workflow (VS Code-first, recommended)

This order avoids any transient "we're on different kernels" mismatch —
Claude attaches to your kernel from the very first call.

1. In Claude Code (with this MCP active), ask:
   > Get the Jupyter server info.
   Claude calls `jupyter_server_info` and tells you the URL+token. **Don't
   open the notebook from Claude yet.**
2. In VS Code: connect to "Existing Jupyter Server" with the URL+token
   (saved across MCP restarts, so this is one-time setup). Open your
   notebook in notebook view, pick a kernel.
3. *Now* from Claude, open the same notebook:
   > Open `/path/to/work.ipynb`.

   `find_existing_session_for_path` lands on VS Code's session
   immediately — same kernel, same PID, shared variables.
4. Drive it together. Examples:
   - You add a cell that loads an image, run it from VS Code.
   - Ask Claude: *"Add a cell that shows the histogram of that image."* —
     Claude reads the live notebook, adds a cell, runs it against the same
     kernel.
   - You add a cell that flips the image, run it. The variable update lives
     in the shared kernel; ask Claude to inspect it.
   - Delete a cell in VS Code. Variables it created stay alive. Claude can
     still use them.

## Concurrent edits work — both sides land

In practice you and Claude can edit the same notebook freely. The
layered safeguards make races a non-issue:

- **Sync before mutate.** Every structural notebook mutation
  (`add_cell`, `edit_cell`, `delete_cell`, `move_cell`,
  `clear_cell_outputs`, `run_code`'s temp-cell path,
  `restart_kernel(clear_outputs=True)`, and the widget metadata
  install) re-reads the notebook from the Jupyter Server immediately
  before writing, and applies the edit on the freshest server-side
  state.
- **Cell-id targeting.** Mutations locate the target by stable
  `cell.id`, not by index — so a reorder you did in VS Code can't
  make Claude edit the wrong cell, and a delete produces a clean
  `NotebookConflictError` rather than mis-editing.
- **Optimistic mtime precondition + bounded retry.** The fresh re-read
  captures the server's `last_modified`; the PUT carries it as an
  `If-Unmodified-Since`-style check. If you saved in the tiny window
  between Claude's read and PUT, `mutate_notebook_fresh` re-reads and
  re-applies the mutator (up to three attempts) instead of surfacing
  a race. Persistent contention eventually raises
  `ConcurrentWriteError`, but you won't hit that under normal use.

Concretely:

- Edit cell A in VS Code while Claude edits cell B — both edits land.
- Delete a cell Claude is about to edit — Claude's call fails clearly
  ("target cell no longer exists") instead of editing the wrong cell.
- Save the notebook while Claude is flushing streaming outputs —
  Claude's mutation retries against your new state rather than
  clobbering you.
- `clear_cell_outputs` (all-cells mode) and
  `restart_kernel(clear_outputs=True)` only zero `outputs` /
  `execution_count` on the freshest cells, so source edits you made
  in VS Code are preserved.

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
uv run pytest          # all tests (helpers + Jupyter-Server-backed integration)
uv run pytest tests/test_execution_helpers.py tests/test_session_helpers.py \
              tests/test_widgets.py tests/test_iopub_tap.py \
              tests/test_awareness.py tests/test_conflict_tools.py
                       # kernel-free unit tests only — fast, no jupyter server required
uv run python -m mcp_jupyter_driver --self-check
```

## Security

The Jupyter Server we host listens on `127.0.0.1` only and uses a random
token per launch. Anyone with the token has full Contents-API access to your
filesystem and can execute arbitrary code through the kernel. Don't share
the token. Avoid running this as root outside containers/CI — `jupyter
server` will refuse without `--allow-root`, which we pass automatically when
we genuinely are root.
