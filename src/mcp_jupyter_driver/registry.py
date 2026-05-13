"""Module-level registry of open notebooks.

Stdio MCP keeps the server process alive for the Claude Code session, so
this dict survives across all tool calls. The registry guards its own
mutation; per-session execution is serialized inside the session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .errors import NotebookAlreadyOpenError, NotebookNotOpenError
from .notebook_io import canonical_path
from .session import NotebookSession

_sessions: dict[Path, NotebookSession] = {}
_lock = asyncio.Lock()


async def open_session(path: str, *, create_if_missing: bool = False) -> NotebookSession:
    cpath = canonical_path(path)
    async with _lock:
        if cpath in _sessions:
            raise NotebookAlreadyOpenError(str(cpath))
        session = await NotebookSession.open(cpath, create_if_missing=create_if_missing)
        _sessions[cpath] = session
        return session


async def close_session(path: str) -> None:
    cpath = canonical_path(path)
    async with _lock:
        session = _sessions.pop(cpath, None)
    if session is None:
        raise NotebookNotOpenError(str(cpath))
    await session.close()


def get_session(path: str) -> NotebookSession:
    cpath = canonical_path(path)
    session = _sessions.get(cpath)
    if session is None:
        raise NotebookNotOpenError(str(cpath))
    return session


def list_sessions() -> list[NotebookSession]:
    return list(_sessions.values())


async def close_all() -> None:
    async with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for s in sessions:
        try:
            await s.close()
        except Exception:
            pass
