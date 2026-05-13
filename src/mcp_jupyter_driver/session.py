"""Per-notebook session: kernel + in-memory notebook + locks + debounced writer.

One session per .ipynb path, lifetime managed by the registry. The session
keeps a single async kernel manager + client; we serialize all execution and
structural edits through `exec_lock` to match Jupyter's own single-runner
model. Cross-session work is fully parallel.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import nbformat
from jupyter_client.manager import AsyncKernelManager
from nbformat.notebooknode import NotebookNode

from .errors import CellNotFoundError, KernelDiedError
from .notebook_io import DebouncedWriter, new_notebook, read_notebook


class NotebookSession:
    def __init__(self, path: Path, nb: NotebookNode) -> None:
        self.path = path
        self.nb = nb
        self.km: AsyncKernelManager = AsyncKernelManager(kernel_name="python3")
        self.kc: Any = None  # AsyncKernelClient, typed loose to avoid import churn
        self.exec_lock = asyncio.Lock()
        self.writer = DebouncedWriter(self.path, lambda: self.nb)
        # Track widget comm activity for diagnostics. The authoritative state
        # snapshot is fetched via a silent kernel call in execution.py, but
        # we keep a cheap presence map here.
        self.widget_comms: dict[str, dict[str, Any]] = {}
        self.execution_count: int = 0

    @classmethod
    async def open(cls, path: Path, *, create_if_missing: bool = False) -> "NotebookSession":
        if path.exists():
            nb = read_notebook(path)
        elif create_if_missing:
            nb = new_notebook()
        else:
            from .errors import NotebookFileMissingError

            raise NotebookFileMissingError(str(path))

        session = cls(path, nb)
        await session.km.start_kernel()
        session.kc = session.km.client()
        session.kc.start_channels()
        await session.kc.wait_for_ready(timeout=60)
        # Persist immediately so a freshly-created notebook lands on disk.
        session.writer.schedule()
        await session.writer.flush()
        return session

    async def close(self) -> None:
        """Shutdown kernel and writer, flush any pending writes."""
        try:
            await self.writer.flush()
        finally:
            await self.writer.close()
            try:
                if self.kc is not None:
                    self.kc.stop_channels()
            except Exception:
                pass
            try:
                if await self.km.is_alive():
                    await self.km.shutdown_kernel(now=False)
            except Exception:
                # Best-effort: don't let a stuck kernel block close.
                try:
                    await self.km.shutdown_kernel(now=True)
                except Exception:
                    pass

    async def assert_alive(self) -> None:
        if not await self.km.is_alive():
            raise KernelDiedError(str(self.path))

    # --- cell lookup helpers ---

    def resolve_cell_index(self, ref: int | str) -> int:
        cells = self.nb.cells
        if isinstance(ref, int):
            if 0 <= ref < len(cells):
                return ref
            raise CellNotFoundError(ref)
        # treat string as cell id
        for i, cell in enumerate(cells):
            if cell.get("id") == ref:
                return i
        raise CellNotFoundError(ref)

    def cell_by_ref(self, ref: int | str) -> NotebookNode:
        return self.nb.cells[self.resolve_cell_index(ref)]

    # --- structural edits (callers must hold exec_lock) ---

    def add_cell(
        self, cell_type: str, source: str, index: int | None = None
    ) -> tuple[int, str | None]:
        if cell_type == "code":
            cell = nbformat.v4.new_code_cell(source=source)
        elif cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source=source)
        elif cell_type == "raw":
            cell = nbformat.v4.new_raw_cell(source=source)
        else:
            raise ValueError(f"unknown cell_type: {cell_type!r}")
        if index is None or index >= len(self.nb.cells):
            self.nb.cells.append(cell)
            idx = len(self.nb.cells) - 1
        else:
            idx = max(0, index)
            self.nb.cells.insert(idx, cell)
        return idx, cell.get("id")

    def edit_cell(self, ref: int | str, source: str) -> tuple[int, str | None]:
        idx = self.resolve_cell_index(ref)
        cell = self.nb.cells[idx]
        cell["source"] = source
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        return idx, cell.get("id")

    def delete_cell(self, ref: int | str) -> None:
        idx = self.resolve_cell_index(ref)
        del self.nb.cells[idx]

    def move_cell(self, from_ref: int | str, to_index: int) -> None:
        src = self.resolve_cell_index(from_ref)
        cell = self.nb.cells.pop(src)
        to_index = max(0, min(to_index, len(self.nb.cells)))
        self.nb.cells.insert(to_index, cell)

    def clear_outputs(self, ref: int | str | None) -> int:
        cleared = 0
        if ref is None:
            for cell in self.nb.cells:
                if cell.get("cell_type") == "code" and cell.get("outputs"):
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cleared += 1
        else:
            idx = self.resolve_cell_index(ref)
            cell = self.nb.cells[idx]
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                cell["outputs"] = []
                cell["execution_count"] = None
                cleared = 1
        return cleared
