"""Variable inspection and completion against the live kernel.

Variables are listed/inspected by running a small `_mcp_`-prefixed helper in
the kernel that JSON-dumps its result to stdout between sentinels. `silent`
on execute_request unfortunately suppresses iopub streams entirely (so we
couldn't capture the stdout), so we use `silent=False, store_history=False`
instead — the helper still runs invisibly to the user because we don't
attach its output to any cell.

The helper names are all `_mcp_`-prefixed and are filtered out of variable
listings.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .errors import KernelDiedError
from .session import NotebookSession

IOPUB_TIMEOUT_S = 0.5
HELPER_TIMEOUT_S = 5.0

_SENTINEL_TAG = "mcp-helper"
_SENTINEL = f"\x1e{_SENTINEL_TAG}\x1e"


def _wrap(payload_expr: str) -> str:
    """Wrap a Python expression producing a JSON-serializable value into
    stdout-bracketed JSON the parser can recover.
    """
    return (
        "import json as _mcp_json, sys as _mcp_sys\n"
        f"_mcp_payload = {payload_expr}\n"
        f"_mcp_sys.stdout.write('{_SENTINEL}' + _mcp_json.dumps(_mcp_payload) + '{_SENTINEL}')\n"
        "_mcp_sys.stdout.flush()\n"
    )


_LIST_VARS_BODY = """
_MCP_IPYTHON_INJECTED = {
    'In', 'Out', 'get_ipython', 'exit', 'quit', 'open',
    'PS1', 'REPLHooks', 'get_last_command', 'is_wsl', 'original_ps1',
    'readline', 'platform', 'sys',
}
def _mcp_list_vars(_include_private=False):
    _mcp_out = []
    _mcp_g = dict(globals())
    for _mcp_n, _mcp_v in _mcp_g.items():
        if _mcp_n.startswith('_mcp_') or _mcp_n.startswith('__'):
            continue
        if _mcp_n in _MCP_IPYTHON_INJECTED:
            continue
        if not _include_private and _mcp_n.startswith('_'):
            continue
        try:
            _mcp_tname = type(_mcp_v).__name__
        except Exception:
            _mcp_tname = '?'
        # size hint
        _mcp_size = ''
        try:
            if hasattr(_mcp_v, 'shape'):
                _mcp_size = 'shape=' + repr(tuple(_mcp_v.shape))
            elif hasattr(_mcp_v, '__len__') and not isinstance(_mcp_v, str):
                _mcp_size = 'len=' + str(len(_mcp_v))
            elif isinstance(_mcp_v, str):
                _mcp_size = 'len=' + str(len(_mcp_v))
        except Exception:
            pass
        # repr preview
        try:
            _mcp_r = repr(_mcp_v)
        except Exception as _mcp_e:
            _mcp_r = '<repr failed: ' + repr(_mcp_e) + '>'
        if len(_mcp_r) > 200:
            _mcp_r = _mcp_r[:199] + '…'
        _mcp_out.append({'name': _mcp_n, 'type': _mcp_tname, 'size_hint': _mcp_size, 'repr_preview': _mcp_r})
    _mcp_out.sort(key=lambda d: d['name'])
    return _mcp_out
"""


def _list_vars_code(include_private: bool) -> str:
    flag = "True" if include_private else "False"
    return _LIST_VARS_BODY + _wrap(f"_mcp_list_vars({flag})")


def _inspect_var_code(name: str, max_repr_len: int) -> str:
    safe_name = json.dumps(name)
    return (
        _LIST_VARS_BODY
        + f"""
def _mcp_inspect(_name, _max_repr):
    _mcp_g = globals()
    if _name not in _mcp_g:
        return {{'found': False, 'name': _name}}
    _v = _mcp_g[_name]
    _info = {{'found': True, 'name': _name, 'type': type(_v).__name__}}
    try:
        _r = repr(_v)
    except Exception as _e:
        _r = '<repr failed: ' + repr(_e) + '>'
    if len(_r) > _max_repr:
        _r = _r[:_max_repr - 1] + '…'
    _info['repr'] = _r
    if hasattr(_v, 'shape'):
        try: _info['shape'] = list(_v.shape)
        except Exception: pass
    if hasattr(_v, 'dtype'):
        try: _info['dtype'] = str(_v.dtype)
        except Exception: pass
    if hasattr(_v, 'dtypes') and hasattr(_v, 'columns'):
        try: _info['columns'] = list(map(str, _v.columns))
        except Exception: pass
        try: _info['dtypes_per_column'] = {{str(k): str(_v.dtypes[k]) for k in _v.columns}}
        except Exception: pass
        try:
            _head = _v.head(5).to_dict(orient='list')
            # make sure values are JSON-friendly
            _head = {{str(k): [None if (isinstance(x, float) and (x != x)) else x for x in vs] for k, vs in _head.items()}}
            _info['head'] = _head
        except Exception: pass
    elif hasattr(_v, '__len__') and not isinstance(_v, (str, bytes)):
        try: _info['length'] = len(_v)
        except Exception: pass
    return _info
"""
        + _wrap(f"_mcp_inspect({safe_name}, {int(max_repr_len)})")
    )


async def _execute_capture(
    session: NotebookSession, code: str, *, timeout_s: float = HELPER_TIMEOUT_S
) -> Any:
    """Run silent-ish helper code and parse its sentinel-wrapped JSON stdout."""
    kc = session.kc
    msg_id = kc.execute(
        code, silent=False, store_history=False, allow_stdin=False
    )
    collected: list[str] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while True:
        if loop.time() > deadline:
            return None
        try:
            msg = await asyncio.wait_for(kc.get_iopub_msg(), timeout=IOPUB_TIMEOUT_S)
        except asyncio.TimeoutError:
            if not await session.km.is_alive():
                raise KernelDiedError(str(session.path))
            continue
        if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
            continue
        mtype = msg.get("msg_type")
        content = msg.get("content", {})
        if mtype == "stream" and content.get("name") == "stdout":
            collected.append(content.get("text", ""))
        elif mtype == "status" and content.get("execution_state") == "idle":
            break

    text = "".join(collected)
    parts = text.split(_SENTINEL)
    if len(parts) < 3:
        return None
    try:
        return json.loads(parts[1])
    except json.JSONDecodeError:
        return None


async def list_variables(session: NotebookSession, *, include_private: bool = False) -> list[dict]:
    async with session.exec_lock:
        result = await _execute_capture(session, _list_vars_code(include_private))
    return result if isinstance(result, list) else []


async def inspect_variable(
    session: NotebookSession, name: str, *, max_repr_len: int = 2000
) -> dict:
    async with session.exec_lock:
        result = await _execute_capture(
            session, _inspect_var_code(name, max_repr_len)
        )
    if not isinstance(result, dict):
        return {"found": False, "name": name}
    return result


async def complete(
    session: NotebookSession, code: str, cursor_pos: int
) -> dict:
    """Wrap kernel `complete_request`. Returns {matches, cursor_start, cursor_end}."""
    async with session.exec_lock:
        kc = session.kc
        msg_id = kc.complete(code, cursor_pos)
        # drain shell channel until our reply
        deadline = asyncio.get_event_loop().time() + 5.0
        while True:
            if asyncio.get_event_loop().time() > deadline:
                return {"matches": [], "cursor_start": cursor_pos, "cursor_end": cursor_pos}
            try:
                msg = await asyncio.wait_for(kc.get_shell_msg(), timeout=0.5)
            except asyncio.TimeoutError:
                if not await session.km.is_alive():
                    raise KernelDiedError(str(session.path))
                continue
            if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                continue
            content = msg.get("content", {})
            return {
                "matches": list(content.get("matches") or []),
                "cursor_start": int(content.get("cursor_start", cursor_pos)),
                "cursor_end": int(content.get("cursor_end", cursor_pos)),
            }
