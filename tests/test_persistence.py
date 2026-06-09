"""Tests for emergo.persistence — save/load round-trip."""

import numpy as np
import pytest

from emergo import Graph, emergo_kernel, make_initial_authority, make_initial_phi
from emergo.persistence import load_state, save_state
from emergo.types import Errors


def _make_state(n=5, seed=0):
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = rng.uniform(0.3, 0.7)
    caps = rng.uniform(0.2, 0.8, (n, 4))
    G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids)
    E = [Errors(per_agent={"a0": 0.1, "a1": 0.2})]
    return G, phi, A, E


class TestPickleRoundTrip:
    def test_save_and_load_pickle(self, tmp_path):
        state = _make_state()
        p = save_state(state, tmp_path / "state", format="pickle")
        assert p.suffix == ".pkl"
        assert p.exists()
        loaded = load_state(p)
        G, phi, A, E = state
        G2, phi2, A2, E2 = loaded
        assert G2.agent_ids == G.agent_ids
        np.testing.assert_array_equal(G2.adjacency, G.adjacency)
        np.testing.assert_array_almost_equal(phi2.W_phi, phi.W_phi)
        assert A2.scores == pytest.approx(A.scores)
        assert len(E2) == len(E)

    def test_pickle_preserves_empty_errors(self, tmp_path):
        G, phi, A, _ = _make_state()
        state = G, phi, A, []
        p = save_state(state, tmp_path / "empty", format="pickle")
        _, _, _, E2 = load_state(p)
        assert E2 == []

    def test_pickle_preserves_authority_baseline(self, tmp_path):
        G, phi, A, E = _make_state()
        A.baseline = 0.7
        p = save_state((G, phi, A, E), tmp_path / "baseline")
        _, _, A2, _ = load_state(p)
        assert A2.baseline == pytest.approx(0.7)


class TestJsonRoundTrip:
    def test_save_and_load_json(self, tmp_path):
        state = _make_state()
        p = save_state(state, tmp_path / "state", format="json")
        assert p.suffix == ".json"
        assert p.exists()
        G, phi, A, _E = state
        G2, phi2, A2, _E2 = load_state(p)
        assert G2.agent_ids == G.agent_ids
        np.testing.assert_array_almost_equal(G2.adjacency, G.adjacency, decimal=10)
        np.testing.assert_array_almost_equal(phi2.W_phi, phi.W_phi, decimal=10)
        assert list(A2.scores.keys()) == list(A.scores.keys())
        assert list(A2.scores.values()) == pytest.approx(list(A.scores.values()))

    def test_json_is_human_readable(self, tmp_path):
        state = _make_state()
        p = save_state(state, tmp_path / "readable", format="json")
        text = p.read_text()
        assert "agent_ids" in text
        assert "W_phi" in text
        assert "scores" in text

    def test_json_errors_preserved(self, tmp_path):
        G, phi, A, E = _make_state()
        p = save_state((G, phi, A, E), tmp_path / "errors", format="json")
        _, _, _, E2 = load_state(p)
        assert len(E2) == len(E)
        assert E2[0].per_agent == pytest.approx(E[0].per_agent)


class TestRoundTripAfterKernelRun:
    def test_save_load_resume_produces_stable_state(self, tmp_path):
        """Save state after 50 iters, reload, run 50 more — should not crash."""
        G, phi, A, _ = _make_state(n=5, seed=7)
        state_in = (G, phi, A, [])

        state_mid, _ = emergo_kernel(state_in, max_iterations=50)

        p = save_state(state_mid, tmp_path / "mid")
        state_restored = load_state(p)

        state_final, reason = emergo_kernel(state_restored, max_iterations=50)
        assert reason in {"Converged", "Max iterations reached"}
        _, phi_f, _A_f, _ = state_final
        assert np.all(np.isfinite(phi_f.W_phi))


class TestEdgeCases:
    def test_unknown_format_raises(self, tmp_path):
        state = _make_state()
        with pytest.raises(ValueError, match="Unknown format"):
            save_state(state, tmp_path / "f", format="csv")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_state(tmp_path / "nonexistent.pkl")

    def test_auto_extension_pickle(self, tmp_path):
        state = _make_state()
        p = save_state(state, tmp_path / "noext")
        assert p.suffix == ".pkl"

    def test_auto_extension_json(self, tmp_path):
        state = _make_state()
        p = save_state(state, tmp_path / "noext", format="json")
        assert p.suffix == ".json"
