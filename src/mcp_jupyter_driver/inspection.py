"""Variable inspection / completion via the kernel WebSocket."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .errors import KernelDiedError
from .session import NotebookSession

WS_TIMEOUT_S = 0.5
HELPER_TIMEOUT_S = 8.0

_SENTINEL_TAG = "mcp-helper"
_SENTINEL = f"\x1e{_SENTINEL_TAG}\x1e"


# Everything the helpers inject is wrapped in a single zero-arg function so
# the kernel-side namespace only ever gains ``_mcp_run`` (which we delete in
# the finally block). No more orphan ``_mcp_payload`` / ``_MCP_IPYTHON_INJECTED``
# / etc. in the user's globals.
_HELPER_PROLOGUE = f"""
def _mcp_run():
    import json as _mcp_json, sys as _mcp_sys
    _MCP_IPYTHON_INJECTED = {{
        'In', 'Out', 'get_ipython', 'exit', 'quit', 'open',
        'PS1', 'REPLHooks', 'get_last_command', 'is_wsl', 'original_ps1',
        'readline', 'platform', 'sys',
    }}
    def _mcp_safe(x):
        # json fallback for objects we can't otherwise encode (numpy scalars,
        # pandas Timestamps, Decimal, sets, bytes, Path, ...). Avoids the
        # whole inspection silently reporting found=False.
        try:
            import numpy as _np
            if isinstance(x, (_np.integer,)): return int(x)
            if isinstance(x, (_np.floating,)):
                v = float(x)
                return None if v != v else v
            if isinstance(x, (_np.bool_,)): return bool(x)
            if isinstance(x, _np.ndarray): return x.tolist()
        except Exception:
            pass
        if isinstance(x, (set, frozenset)): return list(x)
        if isinstance(x, (bytes, bytearray)):
            try: return x.decode('utf-8', 'replace')
            except Exception: return repr(x)
        try:
            return str(x)
        except Exception:
            return repr(x)
    def _list_vars(_include_private):
        _out = []
        _g = dict(globals())
        for _n, _v in _g.items():
            if _n.startswith('_mcp_') or _n.startswith('__'):
                continue
            if _n == '_MCP_IPYTHON_INJECTED' or _n == '_mcp_run':
                continue
            if _n in _MCP_IPYTHON_INJECTED:
                continue
            if not _include_private and _n.startswith('_'):
                continue
            try: _tname = type(_v).__name__
            except Exception: _tname = '?'
            _size = ''
            try:
                if hasattr(_v, 'shape'):
                    _size = 'shape=' + repr(tuple(_v.shape))
                elif hasattr(_v, '__len__') and not isinstance(_v, str):
                    _size = 'len=' + str(len(_v))
                elif isinstance(_v, str):
                    _size = 'len=' + str(len(_v))
            except Exception:
                pass
            try: _r = repr(_v)
            except Exception as _e: _r = '<repr failed: ' + repr(_e) + '>'
            if len(_r) > 200:
                _r = _r[:199] + '…'
            _out.append({{'name': _n, 'type': _tname, 'size_hint': _size, 'repr_preview': _r}})
        _out.sort(key=lambda d: d['name'])
        return _out
    def _inspect(_name, _max_repr):
        _g = globals()
        if _name not in _g:
            return {{'found': False, 'name': _name}}
        _v = _g[_name]
        _info = {{'found': True, 'name': _name, 'type': type(_v).__name__}}
        try: _r = repr(_v)
        except Exception as _e: _r = '<repr failed: ' + repr(_e) + '>'
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
                _head = {{str(k): [(None if (isinstance(x, float) and (x != x)) else x) for x in vs] for k, vs in _head.items()}}
                _info['head'] = _head
            except Exception: pass
        elif hasattr(_v, '__len__') and not isinstance(_v, (str, bytes)):
            try: _info['length'] = len(_v)
            except Exception: pass
        return _info
    return _list_vars, _inspect, _mcp_safe, _mcp_json, _mcp_sys
"""


def _wrap_call(payload_expr: str) -> str:
    """Run the helper, emit sentinel-wrapped JSON, then clean up globals."""
    return (
        _HELPER_PROLOGUE
        + f"""
try:
    _list_vars, _inspect, _safe, _json, _sysmod = _mcp_run()
    _payload = {payload_expr}
    _sysmod.stdout.write('{_SENTINEL}' + _json.dumps(_payload, default=_safe) + '{_SENTINEL}')
    _sysmod.stdout.flush()
finally:
    for _n in ('_mcp_run', '_list_vars', '_inspect', '_safe', '_json',
               '_sysmod', '_payload'):
        globals().pop(_n, None)
"""
    )


def _list_vars_code(include_private: bool) -> str:
    flag = "True" if include_private else "False"
    return _wrap_call(f"_list_vars({flag})")


def _inspect_var_code(name: str, max_repr_len: int) -> str:
    safe_name = json.dumps(name)
    return _wrap_call(f"_inspect({safe_name}, {int(max_repr_len)})")


async def _execute_capture(
    session: NotebookSession, code: str, *, timeout_s: float = HELPER_TIMEOUT_S
) -> Any:
    """Run helper code on the shared kernel and parse sentinel-wrapped JSON."""
    async with session.client.kernel_channel(
        session.kernel_id, session.session_id
    ) as ch:
        msg_id = await ch.send(
            "execute_request",
            {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
        )
        collected: list[str] = []
        deadline = asyncio.get_event_loop().time() + timeout_s
        saw_idle = False
        shell_reply_seen = False
        while not (saw_idle and shell_reply_seen):
            if asyncio.get_event_loop().time() > deadline:
                return None
            try:
                msg = await ch.recv(timeout=WS_TIMEOUT_S)
            except asyncio.TimeoutError:
                try:
                    await session.client.get_kernel(session.kernel_id)
                except Exception:
                    raise KernelDiedError(session.canonical)
                continue
            if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                continue
            channel = msg.get("channel")
            mt = msg.get("msg_type")
            content = msg.get("content", {})
            if channel == "shell" and mt == "execute_reply":
                shell_reply_seen = True
                continue
            if mt == "stream" and content.get("name") == "stdout":
                text = content.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                collected.append(text)
            elif mt == "status" and content.get("execution_state") == "idle":
                saw_idle = True

    text = "".join(collected)
    parts = text.split(_SENTINEL)
    if len(parts) < 3:
        return None
    try:
        return json.loads(parts[1])
    except json.JSONDecodeError:
        return None


async def list_variables(
    session: NotebookSession, *, include_private: bool = False
) -> list[dict]:
    async with session.exec_lock:
        await session.maybe_rejoin()
        result = await _execute_capture(session, _list_vars_code(include_private))
    return result if isinstance(result, list) else []


async def inspect_variable(
    session: NotebookSession, name: str, *, max_repr_len: int = 2000
) -> dict:
    async with session.exec_lock:
        await session.maybe_rejoin()
        result = await _execute_capture(
            session, _inspect_var_code(name, max_repr_len)
        )
    if not isinstance(result, dict):
        return {"found": False, "name": name}
    return result


async def complete(
    session: NotebookSession, code: str, cursor_pos: int
) -> dict:
    async with session.exec_lock:
        await session.maybe_rejoin()
        async with session.client.kernel_channel(
            session.kernel_id, session.session_id
        ) as ch:
            msg_id = await ch.send(
                "complete_request",
                {"code": code, "cursor_pos": cursor_pos},
                channel="shell",
            )
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await ch.recv(timeout=WS_TIMEOUT_S)
                except asyncio.TimeoutError:
                    continue
                if (msg.get("parent_header") or {}).get("msg_id") != msg_id:
                    continue
                if msg.get("channel") != "shell":
                    continue
                content = msg.get("content", {})
                return {
                    "matches": list(content.get("matches") or []),
                    "cursor_start": int(content.get("cursor_start", cursor_pos)),
                    "cursor_end": int(content.get("cursor_end", cursor_pos)),
                }
    return {"matches": [], "cursor_start": cursor_pos, "cursor_end": cursor_pos}
