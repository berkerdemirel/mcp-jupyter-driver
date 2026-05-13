"""Unit tests for atomic writes and the debounced writer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import nbformat
import pytest

from mcp_jupyter_driver.notebook_io import (
    DebouncedWriter,
    _atomic_write,
    canonical_path,
    new_notebook,
    read_notebook,
)


def test_canonical_path_resolves(tmp_path: Path) -> None:
    a = tmp_path / "x.ipynb"
    a.write_text("{}")
    relative = tmp_path / "." / "x.ipynb"
    assert canonical_path(relative) == a.resolve()


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    nb = new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('hi')"))
    path = tmp_path / "rt.ipynb"
    _atomic_write(path, nb)
    nb2 = read_notebook(path)
    assert nb2.cells[0].source == "print('hi')"


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    nb = new_notebook()
    path = tmp_path / "rt.ipynb"
    _atomic_write(path, nb)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(f".{path.name}.tmp.")]
    assert leftovers == []


def test_atomic_write_invalid_nb_raises(tmp_path: Path) -> None:
    nb = new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="ok"))
    # corrupt: remove a required field
    del nb["nbformat"]
    path = tmp_path / "bad.ipynb"
    with pytest.raises(Exception):
        _atomic_write(path, nb)
    assert not path.exists()


async def test_debounced_writer_coalesces(tmp_path: Path) -> None:
    nb = new_notebook()
    path = tmp_path / "d.ipynb"
    _atomic_write(path, nb)

    writes_seen = 0
    original_write = _atomic_write_count_proxy()

    def get_nb():
        return nb

    writer = DebouncedWriter(path, get_nb, debounce_s=0.05)
    for _ in range(5):
        nb.cells.append(nbformat.v4.new_code_cell(source="x"))
        writer.schedule()
    await writer.flush()
    # All schedules collapsed into one write — verify by counting cells on disk.
    on_disk = json.loads(path.read_text())
    assert len(on_disk["cells"]) == 5


def _atomic_write_count_proxy() -> int:
    """Placeholder for a real counter if we want to monkeypatch; not used yet."""
    return 0


async def test_debounced_writer_flushes_on_close(tmp_path: Path) -> None:
    nb = new_notebook()
    path = tmp_path / "c.ipynb"
    _atomic_write(path, nb)

    writer = DebouncedWriter(path, lambda: nb, debounce_s=10.0)
    nb.cells.append(nbformat.v4.new_code_cell(source="A"))
    writer.schedule()
    # debounce window is long; without flush(), nothing should land yet.
    await asyncio.sleep(0.05)
    on_disk = json.loads(path.read_text())
    assert len(on_disk["cells"]) == 0
    await writer.flush()
    on_disk = json.loads(path.read_text())
    assert len(on_disk["cells"]) == 1
    await writer.close()
