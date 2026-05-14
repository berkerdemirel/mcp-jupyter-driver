"""Module-level registry of open notebook sessions.

One MCP-side session per notebook path. The Jupyter Server is the singleton
owner of kernels and notebook files; we just track the binding here.
"""

from __future__ import annotations

import asyncio

from .client import JupyterClient
from .errors import NotebookAlreadyOpenError, NotebookNotOpenError
from .jserver import get_or_start_server
from .session import NotebookSession, canonical_key

_sessions: dict[str, NotebookSession] = {}
_client: JupyterClient | None = None
_lock = asyncio.Lock()


async def get_client() -> JupyterClient:
    global _client
    async with _lock:
        if _client is None:
            srv = await get_or_start_server()
            _client = JupyterClient(srv)
        return _client


async def open_session(
    path: str, *, create_if_missing: bool = False, kernel_name: str = "python3"
) -> NotebookSession:
    key = canonical_key(path)
    client = await get_client()
    async with _lock:
        if key in _sessions:
            raise NotebookAlreadyOpenError(key)
        session = await NotebookSession.open(
            path,
            client=client,
            create_if_missing=create_if_missing,
            kernel_name=kernel_name,
        )
        _sessions[key] = session
        return session


async def close_session(path: str, *, shutdown_kernel: bool = True) -> None:
    key = canonical_key(path)
    async with _lock:
        session = _sessions.pop(key, None)
    if session is None:
        raise NotebookNotOpenError(key)
    await session.close(shutdown_kernel=shutdown_kernel)


def get_session(path: str) -> NotebookSession:
    key = canonical_key(path)
    s = _sessions.get(key)
    if s is None:
        raise NotebookNotOpenError(key)
    return s


def list_sessions() -> list[NotebookSession]:
    return list(_sessions.values())


async def close_all() -> None:
    global _client
    async with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
        client, _client = _client, None
    for s in sessions:
        try:
            await s.close()
        except Exception:
            pass
    if client is not None:
        try:
            await client.close()
        except Exception:
            pass
