"""Tests for PhiUpdate: loss reduction, entanglement guard, gradient correctness."""
import numpy as np
import pytest

from emergo.phi_update import phi_update, _loss_and_grads
from emergo.kernel import make_initial_phi
from emergo.types import CoordinationEvent, Errors, Graph
from emergo.features import extract_graph_features, encode_ce
from tests.conftest import make_ce


def _make_graph(n: int = 3, d_cap: int = 2, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    adj = rng.uniform(0, 1, (n, n)) * (rng.random((n, n)) > 0.5)
    np.fill_diagonal(adj, 0)
    caps = rng.uniform(0, 1, (n, d_cap))
    ids = tuple(f"A{i}" for i in range(n))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _make_history(length: int = 5):
    g_history = [_make_graph(seed=i) for i in range(length + 1)]
    ce_history = [
        make_ce("add_edge", (f"A0", f"A1"), weight=float(i) / length)
        for i in range(length)
    ]
    return g_history, ce_history


class TestPhiUpdate:
    def test_loss_decreases_with_gradient_steps(self):
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        g_hist, ce_hist = _make_history(10)

        # Compute initial loss
        from emergo.features import extract_graph_features, encode_ce
        T = len(g_hist) - 1
        features = np.array([extract_graph_features(G, phi.d_features) for G in g_hist])
        ce_encs = np.array([encode_ce(ce, phi.d_ce) for ce in ce_hist[:T]])
        initial_loss, _ = _loss_and_grads(phi, features, ce_encs, T)

        phi_next, final_loss = phi_update(phi, g_hist, ce_hist, [], n_steps=50, lr=1e-2)

        assert final_loss <= initial_loss + 1e-6, "loss must not increase"

    def test_entanglement_guard_rejects_degenerate_phi(self):
        """If W_phi collapses to rank 0, phi_update must revert to phi_t."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        # Force W_phi to be all zeros (rank 0)
        phi.W_phi[:] = 0.0

        g_hist, ce_hist = _make_history(5)
        phi_next, _ = phi_update(phi, g_hist, ce_hist, [], n_steps=1)

        # Should have reverted to the degenerate phi_t itself
        np.testing.assert_array_equal(phi_next.W_phi, phi.W_phi)

    def test_returns_phi_t_when_insufficient_history(self):
        phi = make_initial_phi()
        g_hist = [_make_graph()]  # only 1 graph → T=0 transitions
        phi_next, loss = phi_update(phi, g_hist, [], [])
        assert loss == float("inf")
        np.testing.assert_array_equal(phi_next.W_phi, phi.W_phi)

    def test_phi_dimensions_preserved(self):
        phi = make_initial_phi(d_latent=6, d_features=16, d_ce=4)
        g_hist, ce_hist = _make_history(8)
        phi_next, _ = phi_update(phi, g_hist, ce_hist, [], n_steps=5)
        assert phi_next.W_phi.shape == phi.W_phi.shape
        assert phi_next.W_F.shape == phi.W_F.shape
        assert phi_next.d_latent == phi.d_latent

    def test_original_phi_not_mutated(self):
        phi = make_initial_phi()
        W_phi_before = phi.W_phi.copy()
        g_hist, ce_hist = _make_history(5)
        phi_update(phi, g_hist, ce_hist, [], n_steps=10)
        np.testing.assert_array_equal(phi.W_phi, W_phi_before)

    def test_gradient_finite(self):
        """Gradients should be finite for well-formed inputs."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        g_hist, ce_hist = _make_history(5)
        T = len(g_hist) - 1
        features = np.array([extract_graph_features(G, phi.d_features) for G in g_hist])
        ce_encs = np.array([encode_ce(ce, phi.d_ce) for ce in ce_hist[:T]])
        _, grads = _loss_and_grads(phi, features, ce_encs, T)
        for name, g in grads.items():
            assert np.all(np.isfinite(g)), f"gradient {name} contains non-finite values"
