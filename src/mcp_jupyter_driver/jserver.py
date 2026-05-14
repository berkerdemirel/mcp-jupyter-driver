"""Supervises a `jupyter server` subprocess.

One server per MCP process. Picks a free port, generates a random token,
spawns the server, waits for it to answer, and tears it down on shutdown.

The URL + token are written to ~/.cache/mcp-jupyter-driver/connection.json
so the user can paste them into VS Code's "Existing Jupyter Server" dialog.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import httpx

CONNECTION_CACHE_PATH = Path.home() / ".cache" / "mcp-jupyter-driver" / "connection.json"
SERVER_STARTUP_TIMEOUT_S = 30.0


def _pick_free_port() -> int:
    """Bind to port 0, grab the assigned port, close. Tiny race with another
    process grabbing the same port, but acceptable for local dev.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class JupyterServer:
    """Lifecycle wrapper around a `jupyter server` subprocess."""

    def __init__(self, *, root_dir: str | None = None) -> None:
        self.port = _pick_free_port()
        self.token = secrets.token_urlsafe(32)
        self.root_dir = root_dir or "/"
        self.proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}

    async def start(self) -> None:
        if self.proc is not None:
            return
        jupyter = shutil.which("jupyter") or "jupyter"
        cmd = [
            jupyter,
            "server",
            "--no-browser",
            f"--ServerApp.port={self.port}",
            f"--ServerApp.token={self.token}",
            f"--ServerApp.root_dir={self.root_dir}",
            "--ServerApp.disable_check_xsrf=True",
            "--ServerApp.allow_origin=*",
            "--ServerApp.password=",
            "--ServerApp.open_browser=False",
            "--ServerApp.answer_yes=True",
            # Don't accept signals from CTRL-C in the parent: we'll terminate
            # the subprocess ourselves.
        ]
        env = dict(os.environ)
        # Don't let JUPYTER_RUNTIME_DIR collisions confuse the runtime path.
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Wait for /api/status to answer.
        deadline = asyncio.get_event_loop().time() + SERVER_STARTUP_TIMEOUT_S
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_event_loop().time() < deadline:
                if self.proc.returncode is not None:
                    raise RuntimeError(
                        f"jupyter server exited during startup with code {self.proc.returncode}"
                    )
                try:
                    r = await client.get(f"{self.url}/api/status", headers=self.headers)
                    if r.status_code == 200:
                        await self._write_connection_file()
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.25)
        raise TimeoutError(
            f"jupyter server didn't answer at {self.url} within {SERVER_STARTUP_TIMEOUT_S}s"
        )

    async def stop(self) -> None:
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(Exception):
                await self._stderr_task
            self._stderr_task = None
        with contextlib.suppress(FileNotFoundError):
            CONNECTION_CACHE_PATH.unlink()

    async def _drain_stderr(self) -> None:
        """Forward server stderr to our stderr so issues surface to the user.

        Jupyter logs are sometimes verbose; we keep them visible because if
        the server misbehaves the user needs to see why.
        """
        assert self.proc is not None
        assert self.proc.stderr is not None
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    break
                sys.stderr.write("[jserver] " + line.decode("utf-8", "replace"))
                sys.stderr.flush()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _write_connection_file(self) -> None:
        CONNECTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "url": self.url,
            "token": self.token,
            "url_with_token": f"{self.url}/?token={self.token}",
            "root_dir": self.root_dir,
            "pid": self.proc.pid if self.proc else None,
        }
        await asyncio.to_thread(
            CONNECTION_CACHE_PATH.write_text, json.dumps(payload, indent=2)
        )


# ---- module-level singleton --------------------------------------------------

_singleton: JupyterServer | None = None
_lock = asyncio.Lock()


async def get_or_start_server() -> JupyterServer:
    global _singleton
    async with _lock:
        if _singleton is None:
            srv = JupyterServer()
            await srv.start()
            _singleton = srv
        return _singleton


async def stop_server() -> None:
    global _singleton
    async with _lock:
        if _singleton is not None:
            await _singleton.stop()
            _singleton = None
