"""Typed exceptions surfaced through the MCP layer.

FastMCP catches uncaught exceptions and reports them to the client, so the
message string here is what Claude will see. Keep messages short and
actionable.
"""

from __future__ import annotations


class DriverError(Exception):
    """Base class for all driver errors."""


class NotebookNotOpenError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Notebook not open: {path}. Call open_notebook first.")
        self.path = path


class NotebookAlreadyOpenError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Notebook already open: {path}. Close it first or use it as-is.")
        self.path = path


class NotebookFileMissingError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Notebook file does not exist: {path}.")
        self.path = path


class CellNotFoundError(DriverError):
    def __init__(self, ref: int | str) -> None:
        super().__init__(f"No cell matching {ref!r}.")
        self.ref = ref


class KernelDiedError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Kernel for {path} is no longer alive. Call restart_kernel to recover."
        )
        self.path = path


class KernelBusyError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Kernel for {path} is busy with another execution. Wait or call "
            f"interrupt_kernel."
        )
        self.path = path


class FileBusyError(DriverError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Could not write {path} (locked or permission denied).")
        self.path = path


class InteractiveInputError(DriverError):
    def __init__(self) -> None:
        super().__init__(
            "Cell requested interactive input(); auto-replied with empty string. "
            "Rewrite the cell to not block on stdin."
        )
