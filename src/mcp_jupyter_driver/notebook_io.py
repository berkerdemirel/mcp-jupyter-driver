"""Reading, writing, and debounced live-saving of .ipynb files.

Atomic writes via temp file + os.replace in the same directory, so editors
(VS Code, JupyterLab) that watch the notebook see only fully-valid states.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

import nbformat
from nbformat.notebooknode import NotebookNode

from .errors import FileBusyError, NotebookFileMissingError


def canonical_path(path: str | os.PathLike[str]) -> Path:
    """Resolve symlinks/relatives so the same notebook always maps to one key."""
    return Path(path).expanduser().resolve(strict=False)


def read_notebook(path: Path) -> NotebookNode:
    if not path.exists():
        raise NotebookFileMissingError(str(path))
    return nbformat.read(path, as_version=4)


def new_notebook() -> NotebookNode:
    return nbformat.v4.new_notebook()


def _atomic_write(path: Path, nb: NotebookNode) -> None:
    """Validate then write via temp + os.replace.

    Same-directory tmp is required for the rename to be atomic on POSIX.
    """
    nbformat.validate(nb)
    parent = path.parent
    if not parent.exists():
        raise NotebookFileMissingError(str(parent))
    tmp = parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)
        os.replace(tmp, path)
    except PermissionError as e:
        raise FileBusyError(str(path)) from e
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class DebouncedWriter:
    """Coalesces bursty writes during streaming cell execution.

    `schedule()` (re)arms a short timer; the actual write happens at most once
    per debounce window. `flush()` writes immediately and is awaited at the
    end of every tool call so the file is in its final state before we
    return to the caller.
    """

    def __init__(
        self,
        path: Path,
        nb_getter: Callable[[], NotebookNode],
        *,
        debounce_s: float = 0.15,
    ) -> None:
        self._path = path
        self._nb_getter = nb_getter
        self._debounce_s = debounce_s
        self._dirty = False
        self._timer: asyncio.TimerHandle | None = None
        self._write_lock = asyncio.Lock()
        self._closed = False

    def schedule(self) -> None:
        if self._closed:
            return
        self._dirty = True
        if self._timer is not None:
            self._timer.cancel()
        loop = asyncio.get_event_loop()
        self._timer = loop.call_later(
            self._debounce_s, lambda: asyncio.ensure_future(self._fire())
        )

    async def _fire(self) -> None:
        self._timer = None
        if not self._dirty or self._closed:
            return
        await self._write_now()

    async def flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._dirty:
            await self._write_now()

    async def _write_now(self) -> None:
        async with self._write_lock:
            self._dirty = False
            nb = self._nb_getter()
            await asyncio.to_thread(_atomic_write, self._path, nb)

    async def close(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
