"""Tests for emergo.serde — kernel state round-trip serialization."""

import numpy as np
import pytest

from emergo import make_initial_authority, make_initial_phi
from emergo.serde import load_checkpoint, save_checkpoint
from emergo.types import Errors, Graph


def _make_state(n: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = rng.uniform(0, 1, (n, n))
    caps = rng.uniform(0, 1, (n, 4))
    G = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=4, d_features=8, d_ce=4, seed=seed)
    A = make_initial_authority(ids, baseline=0.5)
    errors = [
        Errors(
            per_agent={a: float(i * 0.1) for i, a in enumerate(ids)},
            proposer_id=ids[0],
        )
    ]
    return (G, phi, A, errors)


class TestSerdeRoundTrip:
    def test_round_trip_graph(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        G_orig, _, _, _ = state
        G_loaded, _, _, _ = loaded
        assert G_orig.agent_ids == G_loaded.agent_ids
        assert np.allclose(G_orig.adjacency, G_loaded.adjacency)
        assert np.allclose(G_orig.capabilities, G_loaded.capabilities)

    def test_round_trip_phi(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        _, phi_orig, _, _ = state
        _, phi_loaded, _, _ = loaded
        assert np.allclose(phi_orig.W_phi, phi_loaded.W_phi)
        assert np.allclose(phi_orig.W_F, phi_loaded.W_F)
        assert phi_orig.d_latent == phi_loaded.d_latent
        assert phi_orig.d_ce == phi_loaded.d_ce

    def test_round_trip_authority(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        _, _, A_orig, _ = state
        _, _, A_loaded, _ = loaded
        assert A_orig.scores == A_loaded.scores
        assert A_orig.baseline == A_loaded.baseline

    def test_round_trip_error_history(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        _, _, _, errs_orig = state
        _, _, _, errs_loaded = loaded
        assert len(errs_orig) == len(errs_loaded)
        assert errs_orig[0].proposer_id == errs_loaded[0].proposer_id
        assert errs_orig[0].per_agent == errs_loaded[0].per_agent

    def test_empty_error_history(self, tmp_path):
        """State with no error history round-trips cleanly."""
        ids = ("a", "b")
        G = Graph(ids, np.eye(2), np.ones((2, 4)))
        phi = make_initial_phi(4, 8, 4, seed=0)
        A = make_initial_authority(ids)
        state = (G, phi, A, [])
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        _, _, _, errs = loaded
        assert errs == []

    def test_caller_metadata_preserved(self, tmp_path):
        state = _make_state()
        meta = {"iteration": 500, "reason": "converged"}
        save_checkpoint(state, tmp_path, metadata=meta)
        _, loaded_meta = load_checkpoint(tmp_path)
        assert loaded_meta["iteration"] == 500
        assert loaded_meta["reason"] == "converged"

    def test_no_metadata_returns_empty_dict(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        _, loaded_meta = load_checkpoint(tmp_path)
        assert loaded_meta == {}

    def test_missing_npz_raises(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        (tmp_path / "emergo_state.npz").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_missing_json_raises(self, tmp_path):
        state = _make_state()
        save_checkpoint(state, tmp_path)
        (tmp_path / "emergo_meta.json").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_creates_directory_if_absent(self, tmp_path):
        state = _make_state()
        target = tmp_path / "deep" / "nested" / "checkpoint"
        save_checkpoint(state, target)
        assert (target / "emergo_state.npz").exists()
        assert (target / "emergo_meta.json").exists()

    def test_returns_checkpoint_directory_path(self, tmp_path):
        state = _make_state()
        result = save_checkpoint(state, tmp_path)
        assert result == tmp_path

    def test_wrong_version_raises(self, tmp_path):
        """Loading a checkpoint with a mismatched serde_version raises ValueError."""
        import json

        state = _make_state()
        save_checkpoint(state, tmp_path)
        json_path = tmp_path / "emergo_meta.json"
        meta = json.loads(json_path.read_text())
        meta["serde_version"] = "0.0"
        json_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="serde_version"):
            load_checkpoint(tmp_path)

    def test_proposer_id_none_preserved(self, tmp_path):
        """Errors with proposer_id=None round-trips to None, not a string."""
        ids = ("x", "y")
        G = Graph(ids, np.eye(2), np.ones((2, 4)))
        phi = make_initial_phi(4, 8, 4, seed=7)
        A = make_initial_authority(ids)
        errors = [Errors(per_agent={"x": 0.3}, proposer_id=None)]
        state = (G, phi, A, errors)
        save_checkpoint(state, tmp_path)
        loaded, _ = load_checkpoint(tmp_path)
        _, _, _, errs = loaded
        assert errs[0].proposer_id is None
