# mcp-jupyter-driver: notebook sync (VS Code companion)

Auto-reverts `.ipynb` notebooks when an external writer (Claude through the
`mcp-jupyter-driver` MCP server) modifies them on disk, so structural edits
and streaming outputs appear live in your notebook editor instead of
requiring a manual "File: Revert File".

## Why this exists

VS Code's notebook editor only auto-reloads from disk when the in-memory
copy is *not dirty*. Running any cell from VS Code marks the document
dirty (an output gets appended), so every subsequent external write goes
unseen until you save or revert. The MCP server can't fix this on its own
— the dirty-state logic lives inside VS Code. This extension does the
nudge.

## What it does

1. Watches `**/*.ipynb` with a `FileSystemWatcher`.
2. On every external change, looks up the open notebook editor for that
   URI:
   - If the editor's document is **not dirty** → focuses the notebook
     and runs `workbench.action.files.revert`. Claude's edits appear
     in-place.
   - If the editor **is dirty** → shows a status-bar item
     (`Claude updated <name> — click to reload`) so you decide when to
     accept.
3. Debounces back-to-back changes (default 350 ms) so streaming output
   flushes during `run_cell` don't fight the renderer.
4. Mtime-gates: if the change event corresponds to a save you just did
   yourself, it's ignored.

## Install (no build step, no npm needed)

The extension is plain JavaScript — VS Code can load it directly from
this folder. Two ways:

**A) Symlink into the user extensions dir.** On Linux/macOS:

```bash
ln -s "$(pwd)/vscode-extension" \
      ~/.vscode/extensions/mcp-jupyter-driver-sync-0.1.0
```

(On Windows: `%USERPROFILE%\.vscode\extensions\mcp-jupyter-driver-sync-0.1.0`.)
Then restart VS Code. The extension activates the next time you open an
`.ipynb`.

**B) Run as a development extension.** Open `vscode-extension/` in VS
Code and press **F5** — this launches an Extension Development Host with
the extension loaded. Useful for iterating, no install needed.

To verify it's active: open a notebook, check the status bar — when an
external writer changes the file with your in-memory copy dirty, you'll
see `$(sync) <name>: Claude updated — click to reload`.

## Settings

- `mcpJupyterDriverSync.debounceMs` (default `350`)
- `mcpJupyterDriverSync.autoRevertWhenClean` (default `true`)

Open VS Code settings → search "mcpJupyterDriverSync".

## Optional: package as a `.vsix`

If you want a shareable artifact for other machines, you'll need npm and
`@vscode/vsce`. On this cluster:

```bash
module load nodejs/20.12.1     # brings node + npm
npm install -g @vscode/vsce    # one-time; installs into ~/.npm-global by default
vsce package                   # produces mcp-jupyter-driver-sync-0.1.0.vsix
code --install-extension mcp-jupyter-driver-sync-0.1.0.vsix
```

The symlink path above achieves the same outcome with zero tooling — use
`vsce package` only if you want a portable VSIX you can hand to a
colleague.

## What it does NOT do

- It doesn't talk to the MCP server. There's no network channel — it
  reads VS Code's filesystem events. This keeps the extension trivially
  safe and removes any coupling to the MCP transport.
- It doesn't merge edits. If your in-memory copy is dirty, the
  extension surfaces the conflict; resolving it (revert or save first)
  is up to you.
- It doesn't try to be smart about whose write triggered the change. Any
  external `.ipynb` write triggers the same logic. In practice, the only
  external writer you'll see is `mcp-jupyter-driver`.
