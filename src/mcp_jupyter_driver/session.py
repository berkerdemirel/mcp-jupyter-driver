"""Per-notebook session: server-managed kernel + serialization lock.

The Jupyter Server owns the kernel and the .ipynb file. We track the session
binding (notebook path <-> jupyter session id <-> kernel id) so we can
reuse it across tool calls and serialize Claude-driven operations per
notebook.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .client import JupyterClient
from .errors import (
    CellNotFoundError,
    ConcurrentWriteError,
    NotebookConflictError,
    NotebookFileMissingError,
)


# Bounded retry count for ``mutate_notebook_fresh`` on ``ConcurrentWriteError``.
# Three total attempts gives the user one save-storm bounce without burning
# tool budget on persistent races. Each retry re-reads then re-applies the
# mutator on the freshest server-side state, so retries remain correct as
# long as the mutator is a function of the input notebook.
_MUTATE_RETRY_ATTEMPTS = 3


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _looks_like_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(s))


# The suffix the Jupyter extension appends after "<stem>-jvsc-" is two
# UUID-like hex sequences joined by hyphens, followed by ".ipynb". We don't
# want to accept arbitrary "<stem>-jvsc-anything.ipynb" because random files
# could shadow the synthetic path.
_VSCODE_SYNTHETIC_SUFFIX_RE = re.compile(r"^[0-9a-fA-F][0-9a-fA-F\-]*\.ipynb$")


def is_vscode_synthetic_path(name: str, stem: str) -> bool:
    """True if ``name`` matches VS Code's ``<stem>-jvsc-<uuid>-<uuid>.ipynb``."""
    prefix = f"{stem}-jvsc-"
    if not name.startswith(prefix):
        return False
    return bool(_VSCODE_SYNTHETIC_SUFFIX_RE.match(name[len(prefix):]))


@dataclass
class RebindOutcome:
    ok: bool
    reason: Literal[
        "ok", "empty", "too_short", "not_found", "ambiguous", "dead"
    ] = "ok"
    new_session_id: str | None = None
    new_kernel_id: str | None = None
    detail: str = ""
    candidates: list[dict] = field(default_factory=list)


def server_path(path: str) -> str:
    """Strip the leading slash so it's a valid Contents API path under root_dir=/."""
    return Path(path).expanduser().resolve(strict=False).as_posix().lstrip("/")


def canonical_key(path: str) -> str:
    """Stable registry key for a notebook path."""
    return Path(path).expanduser().resolve(strict=False).as_posix()


def _norm_server_path(p: str) -> str:
    """Strip leading slash for direct comparison with our ``server_relative``.

    The Jupyter Server may or may not include the leading slash depending on
    version and how the session was created. Comparing without normalizing
    misses real matches and causes auto-rejoin to silently no-op.
    """
    return (p or "").lstrip("/")


async def find_existing_session_for_path(
    client: JupyterClient,
    server_relative: str,
    *,
    exclude_session_id: str | None = None,
) -> dict | None:
    """Return a live Jupyter session whose path matches ``server_relative``,
    using the same three-tier matching as ``maybe_rejoin``.

    Tier order:

    1. ``exact`` — server-side session path equals ``server_relative``.
    2. ``vscode_synthetic`` — path matches VS Code's
       ``<stem>-jvsc-<uuid>-<uuid>.ipynb`` encoding.
    3. ``basename`` — same filename only, and only when uniquely matched
       (to avoid silently cross-directory-hopping between notebooks that
       happen to share a name).

    Liveness is checked per candidate before selection. Returns ``None``
    when no live candidate matches.

    This is the same logic ``maybe_rejoin`` uses to follow the user's
    kernel after the fact — applied at open time so Claude attaches to
    VS Code's already-running kernel from the very first tool call,
    instead of starting a parallel kernel and only converging on the
    next tool call.
    """
    sessions = await client.list_sessions()
    target = _norm_server_path(server_relative)
    basename = Path(target).name
    stem = Path(basename).stem

    seen: set[str] = set()
    exact: list[dict] = []
    synthetic: list[dict] = []
    basename_only: list[dict] = []
    for s in sessions:
        sid = s.get("id")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        # Exclude our own session so it doesn't poison the tier lists
        # (and gate basename fallback) when we're called from maybe_rejoin
        # while already bound to a same-path session.
        if exclude_session_id is not None and sid == exclude_session_id:
            continue
        spath = _norm_server_path(s.get("path") or "")
        spath_name = Path(spath).name
        if spath == target:
            exact.append(s)
        elif is_vscode_synthetic_path(spath_name, stem):
            synthetic.append(s)
        elif spath_name == basename:
            basename_only.append(s)

    async def _first_alive(lst: list[dict]) -> dict | None:
        for s in lst:
            try:
                await client.get_kernel(s["kernel"]["id"])
                return s
            except Exception:
                continue
        return None

    picked = await _first_alive(exact)
    if picked is not None:
        return picked
    picked = await _first_alive(synthetic)
    if picked is not None:
        return picked
    # Basename fallback only when no exact/synthetic candidates at all
    # (even dead ones), and only when basename match is unique.
    if exact or synthetic:
        return None
    if len(basename_only) != 1:
        return None
    return await _first_alive(basename_only)


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
        owns_session: bool = True,
    ) -> None:
        self.canonical = canonical
        self.server_relative = server_relative
        self.session_id = session_id
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name
        self.client = client
        self.exec_lock = asyncio.Lock()
        # When True, maybe_rejoin keeps its hands off — the user explicitly
        # pinned us to a kernel via rebind_kernel. Cleared by restart_kernel,
        # close_notebook, or an unpin tool.
        self.pinned: bool = False
        # Sessions we created (and can safely DELETE on close). Auto-rejoin
        # and rebind switch the active binding to sessions VS Code owns —
        # we must never delete those, because doing so would shut down the
        # user's kernel.
        self.owned_session_ids: set[str] = (
            {session_id} if owns_session else set()
        )
        # Every session_id the binding has ever pointed at. The iopub tap
        # uses this to attribute executions: anything whose parent_session
        # is in this set is "Claude's"; everything else is "the user's."
        # Updated by maybe_rejoin / rebind_to_kernel when the binding moves.
        self.historical_session_ids: set[str] = {session_id}
        # Persistent iopub subscriber (see iopub_tap.py). Started by
        # ``open`` once we have a running event loop; stopped by ``close``.
        # ``None`` if the tap couldn't start (e.g. test stubs that don't
        # implement kernel_channel) — callers must tolerate that.
        from .iopub_tap import IopubTap  # late import to avoid cycle

        self.iopub_tap: IopubTap | None = IopubTap(self)
        # Last-seen notebook snapshot for awareness diffing. Populated
        # lazily by callers (e.g. on each tool entry) — see awareness.py.
        self.last_seen_notebook: dict | None = None

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
        # ``start_session_for_notebook`` reuses an existing session if one
        # already lives for this path. In that case the kernel pre-existed —
        # treat it as non-owned so close_notebook doesn't tear down a kernel
        # the user might be sharing with VS Code.
        # Before spinning up our own kernel, look for one already running
        # for this notebook — typically VS Code's, possibly under its
        # synthetic ``<stem>-jvsc-…`` path. Attaching here means Claude
        # shares the user's kernel from the very first tool call, instead
        # of orphaning a parallel kernel until ``maybe_rejoin`` converges.
        existing = await find_existing_session_for_path(client, rel)
        if existing is not None:
            sess = existing
            owns = False
        else:
            pre_existing = {s["id"] for s in await client.list_sessions()}
            sess = await client.start_session_for_notebook(
                rel, kernel_name=kernel_name
            )
            owns = sess["id"] not in pre_existing
        nb_session = cls(
            canonical=canonical,
            server_relative=rel,
            session_id=sess["id"],
            kernel_id=sess["kernel"]["id"],
            kernel_name=sess["kernel"]["name"],
            client=client,
            owns_session=owns,
        )
        # Spin up the iopub tap so awareness is live from the first tool
        # call onward. Failures (e.g. test stubs without kernel_channel)
        # are non-fatal — drop the tap to None and continue.
        try:
            if nb_session.iopub_tap is not None:
                nb_session.iopub_tap.start()
        except Exception:
            nb_session.iopub_tap = None
        return nb_session

    async def close(self, *, shutdown_kernel: bool = True) -> None:
        """End this notebook binding.

        ``shutdown_kernel`` is best-effort: we only DELETE sessions we
        created. If auto-rejoin or rebind switched us to a session VS Code
        owns, we leave that session alone — closing a notebook should never
        kill someone else's kernel.
        """
        # Always stop the iopub tap; it holds an open WebSocket and a
        # background task that would otherwise leak after close.
        if self.iopub_tap is not None:
            try:
                await self.iopub_tap.stop()
            except Exception:
                pass
            self.iopub_tap = None
        if not shutdown_kernel:
            return
        for sid in list(self.owned_session_ids):
            try:
                await self.client.delete_session(sid)
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

        Candidate tiers (highest first):

        1. ``exact`` — server-side session path matches ``server_relative``.
        2. ``vscode_synthetic`` — path matches ``<stem>-jvsc-<uuid>-<uuid>.ipynb``,
           the synthetic encoding VS Code's Jupyter extension uses.
        3. ``basename`` — same basename only. Used **only when exactly one**
           live basename candidate exists, since multiple notebooks across
           the tree can share a basename (``project_a/analysis.ipynb`` vs.
           ``project_b/analysis.ipynb``) and hopping cross-directory would
           be a silent footgun.

        Our own session is excluded from the tier lists so we don't gate
        ourselves out of a synthetic/basename switch by also being at the
        exact path. Pin (``rebind_kernel``) suppresses everything.
        """
        if self.pinned:
            return False

        sessions = await self.client.list_sessions()

        # Stickiness: if we're already bound to a session we didn't create
        # ourselves (i.e. it's the user's), stay on it as long as it's still
        # alive on the server. Without this, the *next* maybe_rejoin call
        # after we attached to the user's kernel would see Claude's old
        # also-alive session at the exact path and switch us back, bouncing
        # us off the user's kernel on every other tool call.
        if self.session_id not in self.owned_session_ids:
            if any(s.get("id") == self.session_id for s in sessions):
                try:
                    await self.client.get_kernel(self.kernel_id)
                    return False
                except Exception:
                    # Our user session's kernel died — fall through and try
                    # to pick a new one.
                    pass

        target = _norm_server_path(self.server_relative)
        basename = Path(target).name
        stem = Path(basename).stem

        seen: set[str] = set()
        exact: list[dict] = []
        synthetic: list[dict] = []
        basename_only: list[dict] = []
        for s in sessions:
            sid = s.get("id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            # Skip our own session so its presence at the exact path
            # doesn't block fall-through to synthetic / basename tiers
            # when VS Code uses a different path encoding for its session.
            if sid == self.session_id:
                continue
            spath = _norm_server_path(s.get("path") or "")
            spath_name = Path(spath).name
            if spath == target:
                exact.append(s)
            elif is_vscode_synthetic_path(spath_name, stem):
                synthetic.append(s)
            elif spath_name == basename:
                basename_only.append(s)

        async def _filter_alive(lst: list[dict]) -> list[dict]:
            out: list[dict] = []
            for s in lst:
                try:
                    await self.client.get_kernel(s["kernel"]["id"])
                    out.append(s)
                except Exception:
                    pass
            return out

        # Try tiers in order of preference using *alive* candidates only.
        exact_alive = await _filter_alive(exact)
        if exact_alive:
            return self._switch_within_tier(exact_alive)

        synthetic_alive = await _filter_alive(synthetic)
        if synthetic_alive:
            return self._switch_within_tier(synthetic_alive)

        # Basename fallback is the cross-directory danger zone. Only allow
        # it when no exact/synthetic candidates exist for *other* sessions
        # (we already excluded our own above), and only when the basename
        # match is unique.
        if exact or synthetic:
            return False
        basename_alive = await _filter_alive(basename_only)
        if len(basename_alive) == 1:
            return self._switch_within_tier(basename_alive)
        return False

    def _switch_within_tier(self, alive: list[dict]) -> bool:
        if not alive:
            return False
        non_ours = [s for s in alive if s["kernel"]["id"] != self.kernel_id]
        if non_ours:
            pick = non_ours[0]
        elif any(s["kernel"]["id"] == self.kernel_id for s in alive):
            return False
        else:
            pick = alive[0]
        if pick["kernel"]["id"] == self.kernel_id:
            return False
        self.session_id = pick["id"]
        self.kernel_id = pick["kernel"]["id"]
        self.kernel_name = pick["kernel"]["name"]
        # Track the new session_id so the iopub tap's "Claude vs user"
        # attribution doesn't suddenly misclassify our own runs as the
        # user's after a rejoin.
        self.historical_session_ids.add(pick["id"])
        return True

    # Minimum length for a kernel_id prefix to be accepted by rebind_to_kernel.
    # Short prefixes can match many kernels; we want the user to be specific.
    REBIND_MIN_PREFIX_LEN = 8

    async def rebind_to_kernel(self, target: str) -> "RebindOutcome":
        """Rebind to the kernel matching ``target`` (exact session_id, exact
        kernel_id, or a kernel_id prefix of at least ``REBIND_MIN_PREFIX_LEN``
        characters).

        Returns a :class:`RebindOutcome`:

        - ``ok=True`` with new ids on success (also pins the binding).
        - ``ok=False, reason="empty"`` if ``target`` is empty/None.
        - ``ok=False, reason="too_short"`` if a non-exact prefix is too short.
        - ``ok=False, reason="ambiguous"`` with ``candidates`` listing the
          matches if more than one session matched.
        - ``ok=False, reason="not_found"`` if nothing matched.
        - ``ok=False, reason="dead"`` if the matched kernel is not alive.
        """
        if not isinstance(target, str) or not target.strip():
            return RebindOutcome(ok=False, reason="empty")
        t = target.strip()
        sessions = await self.client.list_sessions()

        exact_matches: list[dict] = []
        prefix_matches: list[dict] = []
        for s in sessions:
            kid = s["kernel"]["id"]
            if s["id"] == t or kid == t:
                exact_matches.append(s)
            elif (
                len(t) >= self.REBIND_MIN_PREFIX_LEN
                and kid.startswith(t)
            ):
                prefix_matches.append(s)

        if not exact_matches and not prefix_matches:
            # Distinguish "you typed a short non-matching prefix" from
            # "we just couldn't find your target," so the user knows why.
            if len(t) < self.REBIND_MIN_PREFIX_LEN and not _looks_like_uuid(t):
                return RebindOutcome(
                    ok=False,
                    reason="too_short",
                    detail=(
                        f"target {t!r} is shorter than the minimum prefix "
                        f"length ({self.REBIND_MIN_PREFIX_LEN}). Pass a full "
                        f"session_id, kernel_id, or a longer prefix."
                    ),
                )
            return RebindOutcome(ok=False, reason="not_found")

        candidates = exact_matches or prefix_matches
        if len(candidates) > 1:
            return RebindOutcome(
                ok=False,
                reason="ambiguous",
                candidates=[
                    {
                        "session_id": s["id"],
                        "kernel_id": s["kernel"]["id"],
                        "path": s.get("path") or "",
                    }
                    for s in candidates
                ],
            )

        pick = candidates[0]
        try:
            await self.client.get_kernel(pick["kernel"]["id"])
        except Exception:
            return RebindOutcome(ok=False, reason="dead")

        self.session_id = pick["id"]
        self.kernel_id = pick["kernel"]["id"]
        self.kernel_name = pick["kernel"]["name"]
        self.pinned = True
        self.historical_session_ids.add(pick["id"])
        return RebindOutcome(
            ok=True,
            new_session_id=pick["id"],
            new_kernel_id=pick["kernel"]["id"],
        )

    def unpin(self) -> None:
        """Allow maybe_rejoin to switch us again (e.g. after a fresh start)."""
        self.pinned = False

    async def read_notebook(self) -> dict:
        """Always fetch fresh from the server (source of truth)."""
        return await self.client.read_notebook(self.server_relative)

    async def read_notebook_with_meta(self) -> tuple[dict, str | None]:
        """Like ``read_notebook`` but also returns the server's ``last_modified``
        so the caller can pass it back as a write precondition.
        """
        return await self.client.read_notebook_with_meta(self.server_relative)

    async def write_notebook(
        self, nb: dict, *, if_unmodified_since: str | None = None
    ) -> None:
        await self.client.write_notebook(
            self.server_relative, nb, if_unmodified_since=if_unmodified_since
        )

    async def mutate_notebook_fresh(
        self,
        mutator: Callable[[dict], dict | None],
        *,
        expected_cell_id: str | None = None,
        expected_source: str | None = None,
        operation_name: str = "edit",
    ) -> dict:
        """Read the notebook fresh, optionally verify preconditions, apply
        the mutator, write back through the Contents API.

        This is the single sync-before-mutate primitive. Every MCP tool that
        changes notebook structure or metadata should go through it so a
        concurrent VS Code save can't be clobbered by a stale full-notebook
        PUT.

        Preconditions are checked on the **fresh** read (not on whatever
        snapshot the caller used earlier), so they catch the read→write race
        window that ``exec_lock`` alone can't close. The fresh read also
        captures ``last_modified``, which we re-check just before the PUT so
        the (small) window between our own read and PUT also raises
        ``ConcurrentWriteError`` instead of silently clobbering.

        - ``expected_cell_id``: the notebook must still contain a cell with
          this id, or ``NotebookConflictError`` is raised. Use it for
          edit/delete/move/per-cell-clear operations.
        - ``expected_source``: when combined with ``expected_cell_id``, the
          target cell's source must equal this string, or
          ``NotebookConflictError`` is raised. Use it when the caller wants
          "edit only if the cell still looks the way I saw it" semantics.
        - ``operation_name``: surfaces in the error message so callers can
          identify which tool produced the conflict.

        The mutator runs in place on the fresh notebook dict; returning
        ``None`` keeps that dict, or it may return a new dict to replace it
        entirely.
        """
        last_concurrent_err: ConcurrentWriteError | None = None
        for _ in range(_MUTATE_RETRY_ATTEMPTS):
            nb, last_modified = await self.read_notebook_with_meta()
            if expected_cell_id is not None:
                cells = nb.get("cells") or []
                found = None
                for c in cells:
                    if c.get("id") == expected_cell_id:
                        found = c
                        break
                if found is None:
                    raise NotebookConflictError(
                        f"{operation_name}: target cell {expected_cell_id!r} no "
                        f"longer exists on the server — was it deleted by "
                        f"another editor?"
                    )
                if expected_source is not None:
                    src = found.get("source", "")
                    if isinstance(src, list):
                        src = "".join(src)
                    if src != expected_source:
                        raise NotebookConflictError(
                            f"{operation_name}: target cell {expected_cell_id!r} "
                            f"source changed since the caller resolved the ref "
                            f"— another editor edited it. Re-read and retry."
                        )
            mutated = mutator(nb)
            out = mutated if mutated is not None else nb
            try:
                await self.write_notebook(out, if_unmodified_since=last_modified)
                return out
            except ConcurrentWriteError as e:
                # Someone else wrote between our read and our PUT. Re-read
                # fresh and re-apply the mutator — semantic preconditions
                # (cell_id / source) are re-checked on the next iteration's
                # fresh state, so a delete-during-race still raises cleanly.
                last_concurrent_err = e
                continue
        # Exhausted retries — surface the conflict so the caller knows.
        assert last_concurrent_err is not None  # for type-checkers
        raise last_concurrent_err

    async def patch_cell(
        self,
        *,
        cell_id: str | None,
        fallback_index: int | None,
        mutate,
    ) -> None:
        """Read fresh, locate one cell, apply ``mutate(cell)``, write back.

        - If ``cell_id`` is given, look it up by id. If the id has
          disappeared, raise ``CellNotFoundError`` (a
          ``NotebookConflictError`` subclass). Never silently fall back to
          ``fallback_index``.
        - If ``cell_id`` is ``None`` (legacy cells), use ``fallback_index``.

        Output-flushing during ``run_cell`` is the hot path here; structural
        edits should prefer ``mutate_notebook_fresh`` directly.
        """

        def _do(nb: dict) -> None:
            cells = nb.get("cells") or []
            if cell_id is not None:
                for c in cells:
                    if c.get("id") == cell_id:
                        mutate(c)
                        return None
                raise CellNotFoundError(cell_id)
            if fallback_index is None or not (0 <= fallback_index < len(cells)):
                raise CellNotFoundError(fallback_index)
            mutate(cells[fallback_index])
            return None

        await self.mutate_notebook_fresh(_do, operation_name="patch_cell")

    async def delete_cell_by_id(self, cell_id: str | None, fallback_index: int) -> None:
        """Read fresh, drop the target cell by id, write back."""

        def _do(nb: dict) -> None:
            cells = nb.get("cells") or []
            if cell_id is not None:
                for i, c in enumerate(cells):
                    if c.get("id") == cell_id:
                        del cells[i]
                        return None
                raise CellNotFoundError(cell_id)
            if not (0 <= fallback_index < len(cells)):
                raise CellNotFoundError(fallback_index)
            del cells[fallback_index]
            return None

        await self.mutate_notebook_fresh(_do, operation_name="delete_cell")

    async def move_cell_by_id(
        self, cell_id: str | None, fallback_index: int, to_index: int
    ) -> None:
        """Read fresh, pop the target cell by id, insert at ``to_index``."""

        def _do(nb: dict) -> None:
            cells = nb.get("cells") or []
            idx: int | None = None
            if cell_id is not None:
                for i, c in enumerate(cells):
                    if c.get("id") == cell_id:
                        idx = i
                        break
                if idx is None:
                    raise CellNotFoundError(cell_id)
            else:
                if not (0 <= fallback_index < len(cells)):
                    raise CellNotFoundError(fallback_index)
                idx = fallback_index
            cell = cells.pop(idx)
            to_idx = max(0, min(to_index, len(cells)))
            cells.insert(to_idx, cell)
            return None

        await self.mutate_notebook_fresh(_do, operation_name="move_cell")
