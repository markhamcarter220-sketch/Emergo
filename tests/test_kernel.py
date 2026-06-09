"""Tests for the full fixed-point loop: state progression, convergence detection."""

import numpy as np

from emergo import (
    Graph,
    Lux,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)


def _small_initial_state():
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    caps = np.ones((3, 2), dtype=float) * 0.5
    G0 = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
    A0 = make_initial_authority(G0.agent_ids)
    return G0, phi0, A0, []


class TestKernel:
    def test_kernel_returns_state_tuple(self):
        state, reason = emergo_kernel(_small_initial_state(), max_iterations=20, lux=Lux())
        G, _phi, _A, _E_history = state
        assert isinstance(G, Graph)
        assert reason in {"Converged", "Max iterations reached"}

    def test_state_n_agents_nonnegative(self):
        state, _ = emergo_kernel(_small_initial_state(), max_iterations=50)
        G, _, _, _ = state
        assert G.n_agents >= 0

    def test_authority_scores_in_unit_interval(self):
        state, _ = emergo_kernel(_small_initial_state(), max_iterations=50)
        _, _, A, _ = state
        for v in A.scores.values():
            assert 0.0 <= v <= 1.0

    def test_error_history_accumulates(self):
        state, _ = emergo_kernel(_small_initial_state(), max_iterations=30)
        _, _, _, E_history = state
        assert len(E_history) > 0

    def test_kernel_deterministic_with_same_seed(self):
        s1, r1 = emergo_kernel(
            _small_initial_state(), max_iterations=30, rng=np.random.default_rng(42)
        )
        s2, r2 = emergo_kernel(
            _small_initial_state(), max_iterations=30, rng=np.random.default_rng(42)
        )
        assert r1 == r2
        np.testing.assert_array_equal(s1[0].adjacency, s2[0].adjacency)

    def test_all_authority_bounded_throughout(self):
        """Run kernel and verify final authority is bounded."""
        state, _ = emergo_kernel(
            _small_initial_state(), max_iterations=100, rng=np.random.default_rng(7)
        )
        _, _, A, _ = state
        for aid, v in A.scores.items():
            assert 0.0 <= v <= 1.0, f"Authority for {aid} = {v} out of [0,1]"

    def test_single_agent_graph_runs_without_crash(self):
        adj = np.zeros((1, 1))
        caps = np.array([[1.0, 0.0]])
        G0 = Graph(agent_ids=("solo",), adjacency=adj, capabilities=caps)
        phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        A0 = make_initial_authority(G0.agent_ids)
        # Single-agent graph: no CEs can be sampled; should exit cleanly
        _state, reason = emergo_kernel((G0, phi0, A0, []), max_iterations=10)
        assert reason == "Max iterations reached"
