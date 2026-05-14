"""Per-notebook session: server-managed kernel + serialization lock.

The Jupyter Server owns the kernel and the .ipynb file. We track the session
binding (notebook path <-> jupyter session id <-> kernel id) so we can
reuse it across tool calls and serialize Claude-driven operations per
notebook.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .client import JupyterClient
from .errors import CellNotFoundError, NotebookFileMissingError


def server_path(path: str) -> str:
    """Strip the leading slash so it's a valid Contents API path under root_dir=/."""
    return Path(path).expanduser().resolve(strict=False).as_posix().lstrip("/")


def canonical_key(path: str) -> str:
    """Stable registry key for a notebook path."""
    return Path(path).expanduser().resolve(strict=False).as_posix()


def resolve_cell_index(nb: dict, ref: int | str) -> int:
    cells = nb.get("cells") or []
    if isinstance(ref, int):
        if 0 <= ref < len(cells):
            return ref
        raise CellNotFoundError(ref)
    for i, c in enumerate(cells):
        if c.get("id") == ref:
            return i
    raise CellNotFoundError(ref)


class NotebookSession:
    def __init__(
        self,
        *,
        canonical: str,
        server_relative: str,
        session_id: str,
        kernel_id: str,
        kernel_name: str,
        client: JupyterClient,
    ) -> None:
        self.canonical = canonical
        self.server_relative = server_relative
        self.session_id = session_id
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name
        self.client = client
        self.exec_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: str,
        *,
        client: JupyterClient,
        create_if_missing: bool = False,
        kernel_name: str = "python3",
    ) -> "NotebookSession":
        canonical = canonical_key(path)
        rel = server_path(path)
        if not await client.notebook_exists(rel):
            if not create_if_missing:
                raise NotebookFileMissingError(canonical)
            await client.create_notebook_if_missing(rel)
        sess = await client.start_session_for_notebook(rel, kernel_name=kernel_name)
        return cls(
            canonical=canonical,
            server_relative=rel,
            session_id=sess["id"],
            kernel_id=sess["kernel"]["id"],
            kernel_name=sess["kernel"]["name"],
            client=client,
        )

    async def close(self, *, shutdown_kernel: bool = True) -> None:
        if shutdown_kernel:
            try:
                await self.client.delete_session(self.session_id)
            except Exception:
                pass

    async def kernel_state(self) -> str:
        try:
            k = await self.client.get_kernel(self.kernel_id)
            return k.get("execution_state", "unknown")
        except Exception:
            return "dead"

    async def is_kernel_alive(self) -> bool:
        try:
            await self.client.get_kernel(self.kernel_id)
            return True
        except Exception:
            return False

    async def maybe_rejoin(self) -> bool:
        """If a live "user" session exists for this notebook, switch to it.

        Strategy:
        - Find candidate sessions for this notebook: exact path match, plus
          basename match (VS Code and Jupyter Server sometimes disagree on
          path encoding — workspace-relative vs absolute — so a basename
          fallback catches that case).
        - Keep only those whose kernel is alive.
        - Prefer a candidate whose kernel_id differs from ours. That's the
          user's kernel (VS Code's), and the whole point of co-editing is
          that Claude joins it.
        - If our current binding is the only live one, do nothing.
        - If our kernel is dead and a live other exists, switch.
        """
        from pathlib import Path as _P

        sessions = await self.client.list_sessions()
        basename = _P(self.server_relative).name

        seen: set[str] = set()
        candidates: list[dict] = []
        for s in sessions:
            sid = s.get("id")
            if not sid or sid in seen:
                continue
            spath = s.get("path") or ""
            if spath == self.server_relative or _P(spath).name == basename:
                seen.add(sid)
                candidates.append(s)
        if not candidates:
            return False

        alive: list[dict] = []
        for s in candidates:
            try:
                await self.client.get_kernel(s["kernel"]["id"])
                alive.append(s)
            except Exception:
                pass
        if not alive:
            return False

        non_ours = [s for s in alive if s["kernel"]["id"] != self.kernel_id]
        if non_ours:
            pick = non_ours[0]
        else:
            ours_alive = any(s["kernel"]["id"] == self.kernel_id for s in alive)
            if ours_alive:
                return False
            pick = alive[0]

        if pick["kernel"]["id"] == self.kernel_id:
            return False
        self.session_id = pick["id"]
        self.kernel_id = pick["kernel"]["id"]
        self.kernel_name = pick["kernel"]["name"]
        return True

    async def rebind_to_kernel(self, target: str) -> bool:
        """Rebind to the kernel matching `target` (kernel_id, session_id, or
        kernel_id_prefix). Returns True if rebound.
        """
        sessions = await self.client.list_sessions()
        for s in sessions:
            if (
                s["id"] == target
                or s["kernel"]["id"] == target
                or s["kernel"]["id"].startswith(target)
            ):
                self.session_id = s["id"]
                self.kernel_id = s["kernel"]["id"]
                self.kernel_name = s["kernel"]["name"]
                return True
        return False

    async def read_notebook(self) -> dict:
        """Always fetch fresh from the server (source of truth)."""
        return await self.client.read_notebook(self.server_relative)

    async def write_notebook(self, nb: dict) -> None:
        await self.client.write_notebook(self.server_relative, nb)
