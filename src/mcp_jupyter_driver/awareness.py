"""Notebook diff + user-activity assembly for the awareness channel.

Combines two signals:

- *Iopub tap* (``iopub_tap.py``): every kernel execution since some time,
  with Claude-vs-user attribution from parent_header.session.
- *Structural diff*: cell-level deltas between a snapshot the session has
  already shown to Claude and the latest server state. Catches add /
  delete / edit / move done from VS Code without an execution.

The output is a single ``UserActivity`` model that Claude can ask for via
the ``recent_user_activity`` tool (and that we may auto-attach to other
tool responses later).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .session import NotebookSession


_PREVIEW_LIMIT = 200


def _str(source) -> str:
    if isinstance(source, list):
        return "".join(x for x in source if isinstance(x, str))
    return source or ""


def _preview(s: str, limit: int = _PREVIEW_LIMIT) -> str:
    if not s:
        return ""
    flat = s.replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ---- response models --------------------------------------------------------


class CellChange(BaseModel):
    """One cell-level change observed between the last-seen state and now."""

    kind: Literal["added", "removed", "edited", "moved"]
    cell_id: str | None = None
    cell_type: str | None = None
    # For added / removed: the current (or last-seen) source.
    source_preview: str = ""
    # For edited only: before/after previews.
    old_source_preview: str = ""
    new_source_preview: str = ""
    # For added / edited: index in the current state.
    # For removed: last-known index in the previous state.
    # For moved: kept None — see from_index / to_index.
    index: int | None = None
    from_index: int | None = None
    to_index: int | None = None


class IopubExecutionInfo(BaseModel):
    started_at: float
    finished_at: float | None = None
    by_claude: bool
    parent_session: str
    code: str | None = None
    execution_count: int | None = None
    status: str
    output_count: int
    outputs_preview: list[dict] = Field(default_factory=list)
    error_name: str | None = None
    error_value: str | None = None


class UserActivity(BaseModel):
    """All activity observed since ``since`` (or session start if None)."""

    since: float | None = None
    until: float
    cell_changes: list[CellChange] = Field(default_factory=list)
    executions: list[IopubExecutionInfo] = Field(default_factory=list)
    # Human-readable note when no awareness is available (e.g. tap couldn't
    # start) so Claude knows the silence is "no signal" not "no activity."
    note: str = ""


# ---- diff -------------------------------------------------------------------


def diff_notebooks(old: dict | None, new: dict) -> list[CellChange]:
    """Cell-level deltas. ``None`` for ``old`` returns an empty list (first
    sight of the notebook is not a "change").

    Only cells with stable ids are diffed. ID-less cells fall through —
    they're rare in modern notebooks (nbformat 4.5+ assigns ids).
    Output-only differences (a cell got run) are NOT reported here; those
    come from the iopub tap.
    """
    if old is None:
        return []

    old_cells = old.get("cells") or []
    new_cells = new.get("cells") or []

    old_by_id: dict[str, tuple[int, dict]] = {}
    for i, c in enumerate(old_cells):
        cid = c.get("id")
        if isinstance(cid, str) and cid:
            old_by_id[cid] = (i, c)

    new_by_id: dict[str, tuple[int, dict]] = {}
    for i, c in enumerate(new_cells):
        cid = c.get("id")
        if isinstance(cid, str) and cid:
            new_by_id[cid] = (i, c)

    changes: list[CellChange] = []

    for cid, (idx, c) in old_by_id.items():
        if cid not in new_by_id:
            changes.append(
                CellChange(
                    kind="removed",
                    cell_id=cid,
                    cell_type=c.get("cell_type"),
                    source_preview=_preview(_str(c.get("source", ""))),
                    index=idx,
                )
            )

    for cid, (idx, c) in new_by_id.items():
        if cid not in old_by_id:
            changes.append(
                CellChange(
                    kind="added",
                    cell_id=cid,
                    cell_type=c.get("cell_type"),
                    source_preview=_preview(_str(c.get("source", ""))),
                    index=idx,
                )
            )

    for cid, (new_idx, new_c) in new_by_id.items():
        if cid not in old_by_id:
            continue
        old_idx, old_c = old_by_id[cid]
        old_src = _str(old_c.get("source", ""))
        new_src = _str(new_c.get("source", ""))
        if old_src != new_src:
            changes.append(
                CellChange(
                    kind="edited",
                    cell_id=cid,
                    cell_type=new_c.get("cell_type"),
                    old_source_preview=_preview(old_src),
                    new_source_preview=_preview(new_src),
                    index=new_idx,
                )
            )
        elif old_idx != new_idx:
            changes.append(
                CellChange(
                    kind="moved",
                    cell_id=cid,
                    cell_type=new_c.get("cell_type"),
                    from_index=old_idx,
                    to_index=new_idx,
                )
            )

    return changes


# ---- assembly ---------------------------------------------------------------


def _to_exec_info(ex) -> IopubExecutionInfo:
    return IopubExecutionInfo(
        started_at=ex.started_at,
        finished_at=ex.finished_at,
        by_claude=ex.by_claude,
        parent_session=ex.parent_session,
        code=ex.code,
        execution_count=ex.execution_count,
        status=ex.status,
        output_count=len(ex.outputs),
        outputs_preview=list(ex.outputs),
        error_name=ex.error_name,
        error_value=ex.error_value,
    )


async def collect_user_activity(
    session: "NotebookSession",
    *,
    since: float | None = None,
    include_claude: bool = False,
    update_snapshot: bool = True,
) -> UserActivity:
    """Snapshot iopub events and notebook diff into a single response.

    - ``since``: epoch seconds. Only events newer than this are returned.
      Defaults to "since the last call to this function" by virtue of the
      caller updating ``session.last_seen_notebook`` between calls (the
      cell-diff is naturally bounded by snapshot). For iopub events,
      ``since`` controls the cutoff explicitly.
    - ``include_claude``: by default we filter executions to those NOT
      attributed to Claude — the use case is "what did the user do." Pass
      True to include Claude's runs (useful when Claude wants to confirm
      its own recent activity).
    - ``update_snapshot``: when True, refresh
      ``session.last_seen_notebook`` to the state we just diffed against,
      so the next call only reports further changes.
    """
    until = time.time()
    fresh: dict = await session.read_notebook()
    cell_changes = diff_notebooks(session.last_seen_notebook, fresh)
    if update_snapshot:
        session.last_seen_notebook = fresh

    note = ""
    executions: list[IopubExecutionInfo] = []
    if session.iopub_tap is not None:
        raw = session.iopub_tap.recent_executions(since=since)
        for ex in raw:
            if not include_claude and ex.by_claude:
                continue
            executions.append(_to_exec_info(ex))
    else:
        note = (
            "iopub tap unavailable for this session — execution awareness "
            "is disabled. Cell-level changes (add/delete/edit/move) are "
            "still reported."
        )

    return UserActivity(
        since=since,
        until=until,
        cell_changes=cell_changes,
        executions=executions,
        note=note,
    )
