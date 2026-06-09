"""StateStore — production-grade kernel state persistence with versioning.

Provides a unified StateStore abstraction with two backends:
  - SqliteStore (default): single SQLite file with WAL mode, versioning, and migration
  - PickleStore: directory-per-run, one pickle file per checkpoint

Usage::

    from emergo.store import make_store, CheckpointKernelObserver

    store = make_store("sqlite", db_path="runs.db")
    run_id = "my_experiment_001"

    # Save a checkpoint
    ckpt_id = store.save(state, run_id=run_id, iteration=100)

    # Load latest checkpoint
    result = store.latest_checkpoint(run_id)
    if result:
        state, meta = result

    # Wire as observer (auto-checkpoints every 50 accepted CEs)
    obs = CheckpointKernelObserver(store, run_id=run_id, every_n_accepted=50)
    final_state, reason = emergo_kernel(initial_state, observers=[obs])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sqlite3
import threading
from typing import Any
import uuid

from emergo.observer import _NoOpMixin
from emergo.types import CoordinationEvent, Errors, State

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointMeta:
    """Immutable metadata record for a single checkpoint."""

    checkpoint_id: str
    run_id: str
    iteration: int
    reason: str
    created_at: str  # ISO-8601


@dataclass
class RunInfo:
    """Mutable run-level metadata."""

    run_id: str
    created_at: str
    updated_at: str
    n_checkpoints: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StateStore(ABC):
    """Unified interface for kernel state persistence."""

    @abstractmethod
    def save(self, state: State, *, run_id: str, iteration: int, reason: str = "") -> str:
        """Save state; return checkpoint_id."""

    @abstractmethod
    def load(self, checkpoint_id: str) -> tuple[State, CheckpointMeta]:
        """Load a checkpoint by its ID."""

    @abstractmethod
    def list_runs(self) -> list[RunInfo]:
        """Return all runs known to this store."""

    @abstractmethod
    def list_checkpoints(self, run_id: str) -> list[CheckpointMeta]:
        """Return all checkpoints for a run, ordered by iteration ascending."""

    @abstractmethod
    def latest_checkpoint(self, run_id: str) -> tuple[State, CheckpointMeta] | None:
        """Return (state, meta) for the most recent checkpoint, or None."""

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its checkpoints."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources (file handles, connections)."""


# ---------------------------------------------------------------------------
# SqliteStore
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    reason TEXT DEFAULT '',
    state BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_ckpt_run_iter ON checkpoints(run_id, iteration DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteStore(StateStore):
    """SQLite-backed StateStore with WAL mode and per-row pickle blobs.

    Args:
        db_path: Path to the SQLite database file (created if absent).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # StateStore interface
    # ------------------------------------------------------------------

    def save(self, state: State, *, run_id: str, iteration: int, reason: str = "") -> str:
        """Pickle-serialize *state* and store it; return the new checkpoint_id."""
        checkpoint_id = str(uuid.uuid4())
        now = _now_iso()
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            # Upsert run row (create on first save, update updated_at on subsequent)
            self._conn.execute(
                """
                INSERT INTO runs (run_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, '{}')
                ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (run_id, now, now),
            )
            self._conn.execute(
                """
                INSERT INTO checkpoints
                    (checkpoint_id, run_id, iteration, reason, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, run_id, iteration, reason, blob, now),
            )
            self._conn.commit()
        return checkpoint_id

    def load(self, checkpoint_id: str) -> tuple[State, CheckpointMeta]:
        """Load state and metadata for *checkpoint_id*.

        Raises:
            KeyError: If the checkpoint does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, iteration, reason, state, created_at "
                "FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")
        run_id, iteration, reason, blob, created_at = row
        state: State = pickle.loads(blob)
        meta = CheckpointMeta(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            iteration=iteration,
            reason=reason,
            created_at=created_at,
        )
        return state, meta

    def list_runs(self) -> list[RunInfo]:
        """Return all runs ordered by creation time ascending."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.run_id, r.created_at, r.updated_at, r.metadata, "
                "COUNT(c.checkpoint_id) AS n "
                "FROM runs r LEFT JOIN checkpoints c ON c.run_id = r.run_id "
                "GROUP BY r.run_id ORDER BY r.created_at ASC"
            ).fetchall()
        result: list[RunInfo] = []
        for run_id, created_at, updated_at, metadata_json, n in rows:
            try:
                metadata = json.loads(metadata_json or "{}")
            except json.JSONDecodeError:
                metadata = {}
            result.append(
                RunInfo(
                    run_id=run_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    n_checkpoints=int(n),
                    metadata=metadata,
                )
            )
        return result

    def list_checkpoints(self, run_id: str) -> list[CheckpointMeta]:
        """Return all checkpoints for *run_id*, ordered by iteration ascending."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT checkpoint_id, iteration, reason, created_at "
                "FROM checkpoints WHERE run_id = ? ORDER BY iteration ASC",
                (run_id,),
            ).fetchall()
        return [
            CheckpointMeta(
                checkpoint_id=ckpt_id,
                run_id=run_id,
                iteration=iteration,
                reason=reason,
                created_at=created_at,
            )
            for ckpt_id, iteration, reason, created_at in rows
        ]

    def latest_checkpoint(self, run_id: str) -> tuple[State, CheckpointMeta] | None:
        """Return the highest-iteration checkpoint for *run_id*, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_id FROM checkpoints "
                "WHERE run_id = ? ORDER BY iteration DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self.load(row[0])

    def delete_run(self, run_id: str) -> None:
        """Delete all checkpoints for *run_id* and then the run record itself."""
        with self._lock:
            self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# PickleStore
# ---------------------------------------------------------------------------


class PickleStore(StateStore):
    """File-system-backed StateStore: one directory per run, one pickle per checkpoint.

    Layout::

        <root_dir>/
            <run_id>/
                index.json          # list of CheckpointMeta dicts
                <checkpoint_id>.pkl # pickled State

    Args:
        root_dir: Root directory; created if absent.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    def _index_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "index.json"

    def _read_index(self, run_id: str) -> list[dict[str, Any]]:
        idx_path = self._index_path(run_id)
        if not idx_path.exists():
            return []
        with open(idx_path) as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _write_index(self, run_id: str, entries: list[dict[str, Any]]) -> None:
        idx_path = self._index_path(run_id)
        with open(idx_path, "w") as f:
            json.dump(entries, f, indent=2)

    # ------------------------------------------------------------------
    # StateStore interface
    # ------------------------------------------------------------------

    def save(self, state: State, *, run_id: str, iteration: int, reason: str = "") -> str:
        """Pickle-serialize *state* to disk; return the new checkpoint_id."""
        checkpoint_id = str(uuid.uuid4())
        now = _now_iso()
        run_dir = self._run_dir(run_id)
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Write state blob
            pkl_path = run_dir / f"{checkpoint_id}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Update index
            entries = self._read_index(run_id)
            entries.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "run_id": run_id,
                    "iteration": iteration,
                    "reason": reason,
                    "created_at": now,
                }
            )
            self._write_index(run_id, entries)
        return checkpoint_id

    def load(self, checkpoint_id: str) -> tuple[State, CheckpointMeta]:
        """Load state and metadata for *checkpoint_id*.

        Raises:
            KeyError: If the checkpoint is not found.
        """
        with self._lock:
            # Search all runs for the checkpoint
            for run_dir in sorted(self._root.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                entries = self._read_index(run_id)
                for entry in entries:
                    if entry["checkpoint_id"] == checkpoint_id:
                        pkl_path = run_dir / f"{checkpoint_id}.pkl"
                        with open(pkl_path, "rb") as f:
                            state: State = pickle.load(f)
                        meta = CheckpointMeta(
                            checkpoint_id=entry["checkpoint_id"],
                            run_id=entry["run_id"],
                            iteration=entry["iteration"],
                            reason=entry["reason"],
                            created_at=entry["created_at"],
                        )
                        return state, meta
        raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")

    def list_runs(self) -> list[RunInfo]:
        """Return all runs ordered by run_id (directory name) ascending."""
        with self._lock:
            result: list[RunInfo] = []
            for run_dir in sorted(self._root.iterdir()):
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                entries = self._read_index(run_id)
                if not entries:
                    # Empty run directory with no index — treat as created now
                    created_at = _now_iso()
                    updated_at = created_at
                else:
                    created_at = entries[0]["created_at"]
                    updated_at = entries[-1]["created_at"]
                result.append(
                    RunInfo(
                        run_id=run_id,
                        created_at=created_at,
                        updated_at=updated_at,
                        n_checkpoints=len(entries),
                        metadata={},
                    )
                )
            return result

    def list_checkpoints(self, run_id: str) -> list[CheckpointMeta]:
        """Return all checkpoints for *run_id*, ordered by iteration ascending."""
        with self._lock:
            entries = self._read_index(run_id)
        entries_sorted = sorted(entries, key=lambda e: e["iteration"])
        return [
            CheckpointMeta(
                checkpoint_id=e["checkpoint_id"],
                run_id=e["run_id"],
                iteration=e["iteration"],
                reason=e["reason"],
                created_at=e["created_at"],
            )
            for e in entries_sorted
        ]

    def latest_checkpoint(self, run_id: str) -> tuple[State, CheckpointMeta] | None:
        """Return the highest-iteration checkpoint for *run_id*, or None."""
        with self._lock:
            entries = self._read_index(run_id)
        if not entries:
            return None
        best = max(entries, key=lambda e: e["iteration"])
        return self.load(best["checkpoint_id"])

    def delete_run(self, run_id: str) -> None:
        """Delete the run directory and all its contents."""
        import shutil

        with self._lock:
            run_dir = self._run_dir(run_id)
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def close(self) -> None:
        """No-op: file handles are opened and closed per operation."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_store(backend: str = "sqlite", **kwargs: Any) -> StateStore:
    """Create a StateStore by name.

    Args:
        backend: ``"sqlite"`` or ``"pickle"``.
        **kwargs: Forwarded to the backend constructor.
                  Use ``db_path=`` for sqlite, ``root_dir=`` for pickle.

    Returns:
        A concrete :class:`StateStore` instance.

    Raises:
        ValueError: If *backend* is not recognised.
    """
    if backend == "sqlite":
        return SqliteStore(**kwargs)
    if backend == "pickle":
        return PickleStore(**kwargs)
    raise ValueError(f"Unknown backend {backend!r}. Choose 'sqlite' or 'pickle'.")


# ---------------------------------------------------------------------------
# CheckpointKernelObserver
# ---------------------------------------------------------------------------


class CheckpointKernelObserver(_NoOpMixin):
    """KernelObserver that auto-checkpoints to a StateStore.

    Checkpoints every ``every_n_accepted`` *accepted* CEs (not iterations —
    only counts accepted CEs to avoid checkpointing on no-progress stretches).
    Also checkpoints on kernel_done.

    Args:
        store:            The :class:`StateStore` to write checkpoints to.
        run_id:           Identifier for this kernel run (used as the store key).
        every_n_accepted: Checkpoint after this many accepted CEs.
    """

    def __init__(
        self,
        store: StateStore,
        *,
        run_id: str,
        every_n_accepted: int = 100,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._every_n = every_n_accepted
        self._n_accepted = 0
        # Keep a reference to the latest state seen (updated on each accepted CE)
        self._latest_state: State | None = None
        self._latest_iteration: int = 0

    def on_iteration_start(self, t: int, state: State) -> None:
        """Cache the current state so we can checkpoint it if needed."""
        # We always keep the freshest state available
        self._latest_state = state
        self._latest_iteration = t

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Errors | None,
    ) -> None:
        """Increment accepted counter; checkpoint every *every_n_accepted* accepts."""
        if not accepted:
            return
        self._n_accepted += 1
        if self._latest_state is not None and self._n_accepted % self._every_n == 0:
            self._store.save(
                self._latest_state,
                run_id=self._run_id,
                iteration=t,
                reason=f"auto_checkpoint_accepted_{self._n_accepted}",
            )

    def on_kernel_done(self, reason: str, state: State, n_iterations: int) -> None:
        """Save a final checkpoint when the kernel exits."""
        self._store.save(
            state,
            run_id=self._run_id,
            iteration=n_iterations,
            reason=reason,
        )
