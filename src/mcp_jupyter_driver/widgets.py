"""ipywidgets pass-through support.

VS Code and JupyterLab render widgets natively if:
  1. The cell output contains application/vnd.jupyter.widget-view+json with a model_id.
  2. The notebook has a widget-state snapshot at
     nb.metadata["widgets"]["application/vnd.jupyter.widget-state+json"].

We don't try to render anything ourselves. Our job is to (a) not drop widget
MIME from outputs (nbformat.v4.output_from_msg keeps it for us already) and
(b) snapshot kernel-side widget state into nb.metadata so it survives a save.

For the snapshot we run a silent kernel-side helper that returns
ipywidgets.Widget.get_manager_state() as JSON. We invoke it lazily — only
after a cell that produced widget output.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"
WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"

# Run the snapshot inside a function and tear down every name it added to
# globals afterwards — otherwise users see _mcp_widget_state_snapshot /
# _mcp_json / _mcp_sys / _mcp_sentinel leaking into the kernel namespace.
_SNAPSHOT_CODE = """
def _mcp_widget_state_snapshot_run():
    import json as _mcp_json, sys as _mcp_sys
    def _state():
        try:
            from ipywidgets import Widget
        except Exception:
            return None
        try:
            return Widget.get_manager_state(drop_defaults=True)
        except Exception as _e:
            return {"_mcp_error": repr(_e)}
    _sentinel = "\\x1emcp-widget-state\\x1e"
    _mcp_sys.stdout.write(_sentinel + _mcp_json.dumps(_state()) + _sentinel)
    _mcp_sys.stdout.flush()
try:
    _mcp_widget_state_snapshot_run()
finally:
    globals().pop('_mcp_widget_state_snapshot_run', None)
"""

_SENTINEL = "\x1emcp-widget-state\x1e"


def outputs_contain_widget(outputs: Iterable[dict]) -> bool:
    """True if any output has a widget-view MIME bundle."""
    for out in outputs:
        data = out.get("data") or {}
        if WIDGET_VIEW_MIME in data:
            return True
    return False


def snapshot_request_code() -> str:
    """Silent kernel code that prints a sentinel-wrapped JSON widget-state snapshot."""
    return _SNAPSHOT_CODE


def parse_snapshot_stdout(stream_text: str) -> dict[str, Any] | None:
    """Extract the snapshot JSON from a captured stdout stream.

    Returns the manager_state dict, or None if ipywidgets isn't installed in
    the kernel / the snapshot failed.
    """
    parts = stream_text.split(_SENTINEL)
    if len(parts) < 3:
        return None
    payload = parts[1]
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if result is None or (isinstance(result, dict) and "_mcp_error" in result):
        return None
    return result


def install_widget_state(nb_metadata: dict, manager_state: dict[str, Any]) -> None:
    """Write a snapshot into nb.metadata under the canonical key.

    Mutates nb_metadata in place.
    """
    widgets_meta = nb_metadata.setdefault("widgets", {})
    widgets_meta[WIDGET_STATE_MIME] = manager_state
