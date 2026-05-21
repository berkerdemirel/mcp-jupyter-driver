"""Async client for the Jupyter Server REST + WebSocket APIs.

- ContentsAPI: GET/PUT notebooks (the server owns the .ipynb on disk).
- KernelsAPI / SessionsAPI: start/stop kernels via the server. We use
  *sessions* (not raw kernels) so VS Code's Jupyter extension recognizes the
  notebook<->kernel binding and reuses our kernel when it opens the same
  notebook.
- KernelChannel: a thin async wrapper over the kernel's WebSocket channels.
  Sends execute/inspect/complete requests; consumes iopub stream.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
import websockets

from .jserver import JupyterServer


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_header(msg_type: str, session_id: str, username: str = "mcp") -> dict:
    return {
        "msg_id": secrets.token_hex(16),
        "username": username,
        "session": session_id,
        "msg_type": msg_type,
        "version": "5.3",
        "date": _ts(),
    }


def _encode_path(path: str) -> str:
    """Server contents paths are URL-encoded and relative to root_dir."""
    p = path.lstrip("/")
    return quote(p, safe="/")


class JupyterClient:
    """REST client + factory for KernelChannel sockets."""

    def __init__(self, server: JupyterServer) -> None:
        self.server = server
        self._http = httpx.AsyncClient(
            base_url=server.url, headers=server.headers, timeout=30.0
        )

    async def close(self) -> None:
        await self._http.aclose()

    # ---- Contents API ------------------------------------------------------

    async def read_notebook(self, path: str) -> dict:
        """GET notebook content. Returns the nbformat dict (.content of the response)."""
        r = await self._http.get(
            f"/api/contents/{_encode_path(path)}",
            params={"type": "notebook", "content": "1"},
        )
        r.raise_for_status()
        return r.json()["content"]

    async def read_notebook_with_meta(self, path: str) -> tuple[dict, str | None]:
        """GET notebook content + last_modified. Returns (content, last_modified).

        ``last_modified`` is the server-reported mtime string; pass it back
        through ``write_notebook(..., if_unmodified_since=...)`` to detect a
        concurrent write between read and PUT.
        """
        r = await self._http.get(
            f"/api/contents/{_encode_path(path)}",
            params={"type": "notebook", "content": "1"},
        )
        r.raise_for_status()
        body = r.json()
        return body["content"], body.get("last_modified")

    async def write_notebook(
        self,
        path: str,
        nb: dict,
        *,
        if_unmodified_since: str | None = None,
    ) -> dict:
        """PUT notebook content. Returns the updated server metadata.

        ``if_unmodified_since`` is an optimistic precondition: if provided, we
        first GET the current server-side ``last_modified`` and compare. If
        the server's value differs, raise ``ConcurrentWriteError`` instead of
        PUTting (so a concurrent editor's write isn't clobbered). The Contents
        API has no native conditional-PUT, so this is a best-effort
        client-side check — a small race window remains, but it shrinks the
        read→write race from the caller's perspective to the time between this
        method's own GET and PUT.
        """
        if if_unmodified_since is not None:
            current = await self.get_mtime(path)
            if current is not None and current != if_unmodified_since:
                from .errors import ConcurrentWriteError

                raise ConcurrentWriteError(path, if_unmodified_since, current)
        body = {"type": "notebook", "format": "json", "content": nb}
        r = await self._http.put(
            f"/api/contents/{_encode_path(path)}", json=body
        )
        r.raise_for_status()
        return r.json()

    async def notebook_exists(self, path: str) -> bool:
        r = await self._http.get(
            f"/api/contents/{_encode_path(path)}", params={"content": "0"}
        )
        return r.status_code == 200

    async def create_notebook_if_missing(self, path: str) -> bool:
        """If a notebook doesn't exist at path, create an empty one. Returns True if created."""
        if await self.notebook_exists(path):
            return False
        empty = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        await self.write_notebook(path, empty)
        return True

    async def get_mtime(self, path: str) -> str | None:
        try:
            r = await self._http.get(
                f"/api/contents/{_encode_path(path)}", params={"content": "0"}
            )
            r.raise_for_status()
            return r.json().get("last_modified")
        except Exception:
            return None

    # ---- Sessions / Kernels ------------------------------------------------

    async def start_session_for_notebook(
        self, path: str, *, kernel_name: str = "python3"
    ) -> dict:
        """Create a notebook session bound to this path + a fresh kernel.

        Returns the session dict (includes id, kernel: {id, ...}). Existing
        sessions for the same path are reused.
        """
        # Reuse if a session already exists for this path.
        r = await self._http.get("/api/sessions")
        r.raise_for_status()
        for s in r.json():
            if s.get("path") == path.lstrip("/"):
                return s

        body = {
            "kernel": {"name": kernel_name},
            "name": path.split("/")[-1],
            "path": path.lstrip("/"),
            "type": "notebook",
        }
        r = await self._http.post("/api/sessions", json=body)
        r.raise_for_status()
        return r.json()

    async def list_sessions(self) -> list[dict]:
        r = await self._http.get("/api/sessions")
        r.raise_for_status()
        return r.json()

    async def get_kernel(self, kernel_id: str) -> dict:
        r = await self._http.get(f"/api/kernels/{kernel_id}")
        r.raise_for_status()
        return r.json()

    async def interrupt_kernel(self, kernel_id: str) -> None:
        r = await self._http.post(f"/api/kernels/{kernel_id}/interrupt")
        r.raise_for_status()

    async def restart_kernel(self, kernel_id: str) -> dict:
        r = await self._http.post(f"/api/kernels/{kernel_id}/restart")
        r.raise_for_status()
        return r.json()

    async def delete_session(self, session_id: str) -> None:
        r = await self._http.delete(f"/api/sessions/{session_id}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    async def list_kernelspecs(self) -> dict:
        r = await self._http.get("/api/kernelspecs")
        r.raise_for_status()
        return r.json()

    # ---- WebSocket factory -------------------------------------------------

    @asynccontextmanager
    async def kernel_channel(self, kernel_id: str, session_id: str):
        """Yield a connected KernelChannel for this kernel."""
        ws_url = self.server.url.replace("http://", "ws://") + (
            f"/api/kernels/{kernel_id}/channels?session_id={session_id}&token={self.server.token}"
        )
        async with websockets.connect(
            ws_url,
            additional_headers=self.server.headers,
            max_size=2**26,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            yield KernelChannel(ws, session_id)


class KernelChannel:
    """Thin async wrapper over a kernel WebSocket: send + receive Jupyter msgs."""

    def __init__(self, ws: Any, session_id: str) -> None:
        self.ws = ws
        self.session_id = session_id

    async def send(
        self,
        msg_type: str,
        content: dict,
        *,
        channel: str = "shell",
        parent_header: dict | None = None,
    ) -> str:
        header = _make_header(msg_type, self.session_id)
        envelope = {
            "header": header,
            "parent_header": parent_header or {},
            "metadata": {},
            "content": content,
            "buffers": [],
            "channel": channel,
        }
        await self.ws.send(json.dumps(envelope))
        return header["msg_id"]

    async def recv(self, *, timeout: float | None = None) -> dict:
        if timeout is None:
            raw = await self.ws.recv()
        else:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        return json.loads(raw)
