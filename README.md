# mcp-jupyter-driver

An MCP server that lets Claude drive Jupyter notebooks with a **persistent
kernel**, while you watch the notebook update **live in your editor** (VS Code,
JupyterLab, etc.).

## Why

Claude's default notebook handling re-runs everything or shells out to a
script — throwing away the main value of a notebook: a kernel that stays
warm so heavy setup (loading data, training, fitting) happens once and dozens
of cheap exploratory cells (filter, plot, summary, prompt iteration) reuse
the in-memory state.

This MCP gives Claude human-level access to a notebook (open / run a cell /
add a cell / inspect state / restart) while keeping the `.ipynb` file on disk
as the live UI: every output stream is written back atomically as the cell
runs, so the editor sees updates as they happen. No web UI to install.

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo> mcp-jupyter-driver
cd mcp-jupyter-driver
uv sync
uv run python -m mcp_jupyter_driver --self-check
```

Register with Claude Code — add to `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "jupyter": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/mcp-jupyter-driver",
        "run", "python", "-m", "mcp_jupyter_driver"
      ]
    }
  }
}
```

Then restart Claude Code. `claude mcp` should list `jupyter`.

## Tool surface

**Lifecycle**

| Tool | What it does |
|---|---|
| `open_notebook(path, create_if_missing=False)` | Load a `.ipynb` and start a kernel for it. |
| `close_notebook(path)` | Shutdown the kernel and forget the notebook. |
| `list_open_notebooks()` | All notebooks open in this session. |

**Cell editing**

| Tool | What it does |
|---|---|
| `add_cell(path, cell_type, source, index=None)` | Insert a new code/markdown/raw cell. Returns its index and id. |
| `edit_cell(path, ref, source)` | Replace a cell's source. Clears outputs for code cells. |
| `delete_cell(path, ref)` | Remove a cell by index or id. |
| `move_cell(path, from_ref, to_index)` | Reorder a cell. |
| `clear_cell_outputs(path, ref=None)` | Clear outputs (one cell or all code cells). |
| `list_cells(path)` | Cells with index, id, type, source preview, exec count. |
| `get_cell(path, ref)` | Full source + outputs of one cell. |

**Execution**

| Tool | What it does |
|---|---|
| `run_cell(path, ref, timeout_s=120, restart_on_kernel_death=False)` | Execute a cell. Outputs stream into the `.ipynb` live. |
| `run_code(path, source, persist_as_cell=False, timeout_s=120)` | Append-and-run; optionally remove the cell after. |

**Kernel control**

| Tool | What it does |
|---|---|
| `kernel_status(path)` | Is the kernel alive? Busy? |
| `interrupt_kernel(path)` | SIGINT the kernel (stop a runaway cell). |
| `restart_kernel(path, clear_outputs=False)` | Restart the kernel. |

**Introspection**

| Tool | What it does |
|---|---|
| `list_variables(path, include_private=False)` | User variables in the live kernel: name, type, size hint, repr preview. |
| `inspect_variable(path, name, max_repr_len=2000)` | Deep inspect one variable (pandas: columns/dtypes/head; numpy: shape/dtype). |
| `complete(path, source, cursor_pos)` | Kernel-driven completion. |

## Usage pattern

In a Claude Code session:

1. Open the `.ipynb` in VS Code (or JupyterLab) on one side. Make sure to use
   the notebook view, not the JSON view.
2. Ask Claude: *"Open `path/to/exploration.ipynb` and run the first cell."*
3. Claude calls `open_notebook` (kernel starts), then `run_cell`. The output
   appears in VS Code as it streams in.
4. Iterate: *"Add a cell that filters by date and plots it."* Cell appears,
   runs against the kernel that already has your data loaded.

The kernel stays alive across every tool call in the session. Restart it
explicitly with `restart_kernel`.

## ipywidgets

Widgets work day one (pass-through). When a cell produces an `ipywidgets`
view (e.g. `IntSlider`), the widget MIME bundle is preserved in the cell
output and the kernel-side widget state is snapshotted into
`nb.metadata.widgets` on save — so VS Code's notebook renderer shows a live,
interactive widget. Claude can create and display widgets; programmatically
driving widget state from Claude (clicking buttons, moving sliders via the
server) is not in v1.

## Known limitations (v1)

- **Claude owns the file while a notebook is open.** Manual edits to the
  `.ipynb` while a session is open may be overwritten on the next
  cell-output write. Close the notebook first if you want to hand-edit.
- **One execution at a time per kernel.** Matches Jupyter's model.
  Cross-notebook execution is parallel.
- **Large outputs are capped.** Per-output text caps at 1 MB, per-cell total
  at 5 MB; oversize content is truncated with an explicit marker. Images,
  widget MIME, and JSON bundles are left intact.
- **Cells that call `input()` won't hang.** The server auto-replies with an
  empty string and sets `interactive_input=True` on the run result so you
  know it happened. Rewrite the cell to not block on stdin.

## Development

```bash
uv sync
uv run pytest          # 26 tests (unit + kernel integration)
uv run python -m mcp_jupyter_driver --self-check
```

## Security

The MCP runs arbitrary Python code in whichever environment runs the server.
Run as your normal user; do not run elevated, and don't point it at
untrusted notebooks unless you'd already trust their code.
