"""Comprehensive tests for emergo.store — SqliteStore, PickleStore, make_store,
and CheckpointKernelObserver."""

from __future__ import annotations

import numpy as np
import pytest

from emergo import Graph, make_initial_authority, make_initial_phi
from emergo.store import (
    CheckpointKernelObserver,
    CheckpointMeta,
    PickleStore,
    RunInfo,
    SqliteStore,
    StateStore,
    make_store,
)
from emergo.types import CoordinationEvent, Errors

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_state(n: int = 3):
    adj = np.eye(n, k=1) + np.eye(n, k=-(n - 1))
    caps = np.ones((n, 2))
    G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=4, d_features=8, d_ce=4, seed=0)
    A = make_initial_authority(G.agent_ids)
    return G, phi, A, []


def _make_ce() -> CoordinationEvent:
    return CoordinationEvent(
        event_type="add_edge",
        participants=("a0", "a1"),
        params=frozenset(),
    )


# ---------------------------------------------------------------------------
# TestSqliteStore
# ---------------------------------------------------------------------------


class TestSqliteStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        G, phi, A, E = state

        ckpt_id = store.save(state, run_id="r1", iteration=10)
        assert isinstance(ckpt_id, str) and len(ckpt_id) > 0

        loaded_state, meta = store.load(ckpt_id)
        G2, phi2, A2, E2 = loaded_state

        assert G2.agent_ids == G.agent_ids
        np.testing.assert_array_equal(G2.adjacency, G.adjacency)
        np.testing.assert_array_almost_equal(phi2.W_phi, phi.W_phi)
        assert A2.scores == pytest.approx(A.scores)
        assert E2 == E

        assert meta.checkpoint_id == ckpt_id
        assert meta.run_id == "r1"
        assert meta.iteration == 10
        store.close()

    def test_list_runs_empty(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        assert store.list_runs() == []
        store.close()

    def test_list_runs_after_save(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        store.save(state, run_id="run_a", iteration=1)
        store.save(state, run_id="run_b", iteration=2)

        runs = store.list_runs()
        run_ids = [r.run_id for r in runs]
        assert "run_a" in run_ids
        assert "run_b" in run_ids
        assert len(runs) == 2
        for r in runs:
            assert isinstance(r, RunInfo)
            assert r.n_checkpoints == 1
        store.close()

    def test_list_checkpoints(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        store.save(state, run_id="r1", iteration=5)
        store.save(state, run_id="r1", iteration=10)
        store.save(state, run_id="r1", iteration=15)

        ckpts = store.list_checkpoints("r1")
        assert len(ckpts) == 3
        iterations = [c.iteration for c in ckpts]
        assert iterations == sorted(iterations), "Should be sorted ascending"
        assert all(isinstance(c, CheckpointMeta) for c in ckpts)
        assert all(c.run_id == "r1" for c in ckpts)
        store.close()

    def test_latest_checkpoint_none_when_empty(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        result = store.latest_checkpoint("nonexistent_run")
        assert result is None
        store.close()

    def test_latest_checkpoint_returns_last(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        store.save(state, run_id="r1", iteration=5)
        store.save(state, run_id="r1", iteration=20)
        store.save(state, run_id="r1", iteration=10)

        result = store.latest_checkpoint("r1")
        assert result is not None
        _, meta = result
        assert meta.iteration == 20
        store.close()

    def test_delete_run_removes_checkpoints(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        store.save(state, run_id="r1", iteration=1)
        store.save(state, run_id="r1", iteration=2)
        store.save(state, run_id="r2", iteration=1)

        store.delete_run("r1")

        runs = store.list_runs()
        run_ids = [r.run_id for r in runs]
        assert "r1" not in run_ids
        assert "r2" in run_ids
        assert store.list_checkpoints("r1") == []
        assert store.latest_checkpoint("r1") is None
        store.close()

    def test_save_multiple_iterations(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        ids = []
        for i in range(5):
            ckpt_id = store.save(state, run_id="r1", iteration=i * 10)
            ids.append(ckpt_id)

        # All IDs should be unique
        assert len(set(ids)) == 5

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].n_checkpoints == 5
        store.close()

    def test_state_roundtrip_preserves_graph(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        G, phi, A, _ = _make_state(n=5)
        state = (G, phi, A, [Errors(per_agent={"a0": 0.1, "a1": 0.2})])

        ckpt_id = store.save(state, run_id="graph_test", iteration=99)
        loaded_state, meta = store.load(ckpt_id)

        G2, _phi2, _A2, E2 = loaded_state
        assert G2.n_agents == 5
        np.testing.assert_array_equal(G2.adjacency, G.adjacency)
        np.testing.assert_array_equal(G2.capabilities, G.capabilities)
        assert len(E2) == 1
        assert E2[0].per_agent == pytest.approx({"a0": 0.1, "a1": 0.2})
        assert meta.iteration == 99
        store.close()

    def test_load_nonexistent_raises_key_error(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        with pytest.raises(KeyError, match="Checkpoint not found"):
            store.load("00000000-0000-0000-0000-000000000000")
        store.close()

    def test_reason_stored_and_retrieved(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        state = _make_state()
        ckpt_id = store.save(state, run_id="r1", iteration=1, reason="periodic")
        _, meta = store.load(ckpt_id)
        assert meta.reason == "periodic"
        store.close()


# ---------------------------------------------------------------------------
# TestPickleStore
# ---------------------------------------------------------------------------


class TestPickleStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        G, phi, A, _E = state

        ckpt_id = store.save(state, run_id="r1", iteration=42)
        assert isinstance(ckpt_id, str)

        loaded_state, meta = store.load(ckpt_id)
        G2, phi2, A2, _E2 = loaded_state

        assert G2.agent_ids == G.agent_ids
        np.testing.assert_array_equal(G2.adjacency, G.adjacency)
        np.testing.assert_array_almost_equal(phi2.W_phi, phi.W_phi)
        assert A2.scores == pytest.approx(A.scores)

        assert meta.checkpoint_id == ckpt_id
        assert meta.run_id == "r1"
        assert meta.iteration == 42
        store.close()

    def test_list_runs(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        store.save(state, run_id="run_alpha", iteration=1)
        store.save(state, run_id="run_beta", iteration=2)

        runs = store.list_runs()
        run_ids = [r.run_id for r in runs]
        assert "run_alpha" in run_ids
        assert "run_beta" in run_ids
        assert len(runs) == 2
        store.close()

    def test_list_checkpoints(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        store.save(state, run_id="r1", iteration=30)
        store.save(state, run_id="r1", iteration=10)
        store.save(state, run_id="r1", iteration=20)

        ckpts = store.list_checkpoints("r1")
        assert len(ckpts) == 3
        iterations = [c.iteration for c in ckpts]
        assert iterations == sorted(iterations), "Should be sorted ascending"
        store.close()

    def test_latest_checkpoint(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        store.save(state, run_id="r1", iteration=5)
        store.save(state, run_id="r1", iteration=50)
        store.save(state, run_id="r1", iteration=25)

        result = store.latest_checkpoint("r1")
        assert result is not None
        _, meta = result
        assert meta.iteration == 50
        store.close()

    def test_latest_checkpoint_empty(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        assert store.latest_checkpoint("no_such_run") is None
        store.close()

    def test_delete_run(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        store.save(state, run_id="r1", iteration=1)
        store.save(state, run_id="r2", iteration=1)

        store.delete_run("r1")

        runs = store.list_runs()
        run_ids = [r.run_id for r in runs]
        assert "r1" not in run_ids
        assert "r2" in run_ids

        assert store.list_checkpoints("r1") == []
        assert store.latest_checkpoint("r1") is None

        # Run directory must not exist
        assert not (tmp_path / "store" / "r1").exists()
        store.close()

    def test_load_nonexistent_raises_key_error(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        with pytest.raises(KeyError, match="Checkpoint not found"):
            store.load("00000000-0000-0000-0000-000000000000")
        store.close()

    def test_n_checkpoints_in_list_runs(self, tmp_path):
        store = PickleStore(tmp_path / "store")
        state = _make_state()
        for i in range(3):
            store.save(state, run_id="r1", iteration=i)

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].n_checkpoints == 3
        store.close()


# ---------------------------------------------------------------------------
# TestMakeStore
# ---------------------------------------------------------------------------


class TestMakeStore:
    def test_make_sqlite(self, tmp_path):
        store = make_store("sqlite", db_path=tmp_path / "runs.db")
        assert isinstance(store, SqliteStore)
        assert isinstance(store, StateStore)
        store.close()

    def test_make_pickle(self, tmp_path):
        store = make_store("pickle", root_dir=tmp_path / "store")
        assert isinstance(store, PickleStore)
        assert isinstance(store, StateStore)
        store.close()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            make_store("hdf5")


# ---------------------------------------------------------------------------
# TestCheckpointKernelObserver
# ---------------------------------------------------------------------------


class TestCheckpointKernelObserver:
    def _make_obs(self, store: StateStore, every_n: int = 3) -> CheckpointKernelObserver:
        return CheckpointKernelObserver(store, run_id="test_run", every_n_accepted=every_n)

    def _fire_accepted(self, obs: CheckpointKernelObserver, t: int, state) -> None:
        ce = _make_ce()
        obs.on_ce_result(t, ce, accepted=True, errors=None)

    def _fire_rejected(self, obs: CheckpointKernelObserver, t: int) -> None:
        ce = _make_ce()
        obs.on_ce_result(t, ce, accepted=False, errors=None)

    def test_checkpoints_every_n_accepted(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        obs = self._make_obs(store, every_n=3)
        state = _make_state()

        # Seed initial state via on_iteration_start
        obs.on_iteration_start(0, state)

        # Fire 6 accepted CEs; expect checkpoints at accepted counts 3 and 6
        for t in range(6):
            obs.on_iteration_start(t, state)
            self._fire_accepted(obs, t, state)

        ckpts = store.list_checkpoints("test_run")
        assert len(ckpts) == 2, f"Expected 2 periodic checkpoints, got {len(ckpts)}"
        store.close()

    def test_checkpoints_on_kernel_done(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        obs = self._make_obs(store, every_n=100)  # high threshold → no periodic ckpts
        state = _make_state()

        obs.on_kernel_done(reason="Converged", state=state, n_iterations=50)

        ckpts = store.list_checkpoints("test_run")
        assert len(ckpts) == 1
        assert ckpts[0].reason == "Converged"
        assert ckpts[0].iteration == 50
        store.close()

    def test_does_not_checkpoint_on_rejected(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        obs = self._make_obs(store, every_n=3)
        state = _make_state()

        obs.on_iteration_start(0, state)
        # Fire only rejected CEs
        for t in range(10):
            self._fire_rejected(obs, t)

        # No checkpoints should have been created
        ckpts = store.list_checkpoints("test_run")
        assert len(ckpts) == 0
        store.close()

    def test_no_checkpoint_before_first_n_accepted(self, tmp_path):
        store = SqliteStore(tmp_path / "runs.db")
        obs = self._make_obs(store, every_n=5)
        state = _make_state()

        obs.on_iteration_start(0, state)
        # Only 4 accepted — should not trigger checkpoint
        for t in range(4):
            obs.on_iteration_start(t, state)
            self._fire_accepted(obs, t, state)

        assert store.list_checkpoints("test_run") == []
        store.close()

    def test_checkpoint_uses_latest_state(self, tmp_path):
        """on_kernel_done checkpoints the final state, not an earlier one."""
        store = SqliteStore(tmp_path / "runs.db")
        obs = CheckpointKernelObserver(store, run_id="test_run", every_n_accepted=100)

        state_v1 = _make_state(n=3)
        state_v2 = _make_state(n=4)  # different graph size

        obs.on_iteration_start(0, state_v1)
        obs.on_kernel_done(reason="Converged", state=state_v2, n_iterations=10)

        result = store.latest_checkpoint("test_run")
        assert result is not None
        (G, _phi, _A, _E), _meta = result
        assert G.n_agents == 4, "Should have checkpointed the final state (n=4)"
        store.close()

    def test_pickle_store_observer(self, tmp_path):
        """Observer works with PickleStore backend too."""
        store = PickleStore(tmp_path / "store")
        obs = CheckpointKernelObserver(store, run_id="r1", every_n_accepted=2)
        state = _make_state()

        for t in range(4):
            obs.on_iteration_start(t, state)
            obs.on_ce_result(t, _make_ce(), accepted=True, errors=None)

        ckpts = store.list_checkpoints("r1")
        assert len(ckpts) == 2  # at accepted=2 and accepted=4
        store.close()
