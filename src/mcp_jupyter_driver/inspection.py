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


def _wrap(payload_expr: str) -> str:
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
