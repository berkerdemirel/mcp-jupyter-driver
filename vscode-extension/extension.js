// Auto-revert notebooks when an external writer (typically Claude via the
// mcp-jupyter-driver MCP server) modifies them on disk.
//
// VS Code's notebook editor only re-reads from disk when the in-memory copy
// is *not* dirty. Running even one cell from VS Code dirties the document,
// so subsequent external writes silently sit in the file without appearing
// in the editor until the user runs "File: Revert File". This extension
// nudges the revert command when an external change is detected on a clean
// notebook, and surfaces a status-bar item when the document is dirty so
// the user can see and act on the pending sync.
//
// Plain JavaScript (no build step) so it can be loaded straight from this
// folder via "Developer: Install Extension from Location..." or by copying
// into ~/.vscode/extensions/.

const vscode = require("vscode");

let watcher;
let statusItem;
const pendingByUri = new Map();
const lastHandledMtime = new Map();

function activate(context) {
  watcher = vscode.workspace.createFileSystemWatcher("**/*.ipynb");
  context.subscriptions.push(watcher);

  watcher.onDidChange(scheduleHandle, undefined, context.subscriptions);
  watcher.onDidCreate(scheduleHandle, undefined, context.subscriptions);

  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  statusItem.command = "mcpJupyterDriverSync.revertActive";
  context.subscriptions.push(statusItem);

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "mcpJupyterDriverSync.revertActive",
      async () => {
        const active = vscode.window.activeNotebookEditor;
        if (!active) return;
        await vscode.commands.executeCommand("workbench.action.files.revert");
        clearStatusItem();
      },
    ),
  );

  context.subscriptions.push(
    vscode.workspace.onDidCloseNotebookDocument((doc) => {
      const key = doc.uri.toString();
      const pending = pendingByUri.get(key);
      if (pending) {
        clearTimeout(pending.timer);
        pendingByUri.delete(key);
      }
      lastHandledMtime.delete(key);
      clearStatusItem();
    }),
  );
}

function deactivate() {
  for (const { timer } of pendingByUri.values()) clearTimeout(timer);
  pendingByUri.clear();
}

function scheduleHandle(uri) {
  const cfg = vscode.workspace.getConfiguration("mcpJupyterDriverSync");
  const debounceMs = Math.max(0, cfg.get("debounceMs", 350));
  const key = uri.toString();
  const existing = pendingByUri.get(key);
  if (existing) clearTimeout(existing.timer);
  const timer = setTimeout(() => {
    pendingByUri.delete(key);
    handleChange(uri).catch(() => {});
  }, debounceMs);
  pendingByUri.set(key, { uri, timer });
}

async function handleChange(uri) {
  const editor = vscode.window.visibleNotebookEditors.find(
    (e) => e.notebook.uri.toString() === uri.toString(),
  );
  if (!editor) return;

  try {
    const stat = await vscode.workspace.fs.stat(uri);
    const prev = lastHandledMtime.get(uri.toString());
    if (prev !== undefined && stat.mtime === prev) return;
    lastHandledMtime.set(uri.toString(), stat.mtime);
  } catch {
    // Stat may fail if the file is mid-write; treat as "act anyway."
  }

  if (editor.notebook.isDirty) {
    showDirtyStatus(uri);
    return;
  }

  const cfg = vscode.workspace.getConfiguration("mcpJupyterDriverSync");
  if (!cfg.get("autoRevertWhenClean", true)) return;

  await vscode.window.showNotebookDocument(editor.notebook, {
    preserveFocus: false,
    preview: false,
  });
  try {
    await vscode.commands.executeCommand("workbench.action.files.revert");
  } catch {
    // Transient locks; the next change event re-tries.
  }
  clearStatusItem();
}

function showDirtyStatus(uri) {
  if (!statusItem) return;
  const name = uri.path.split("/").pop() || "notebook";
  statusItem.text = `$(sync) ${name}: Claude updated — click to reload`;
  statusItem.tooltip =
    "An external writer (Claude/mcp-jupyter-driver) modified this " +
    "notebook on disk, but your in-memory copy is dirty so the editor " +
    "kept its own version. Click to revert and pick up Claude's edits.";
  statusItem.show();
}

function clearStatusItem() {
  if (statusItem) statusItem.hide();
}

module.exports = { activate, deactivate };
