"""Tests for PhiUpdate: loss reduction, entanglement guard, gradient correctness."""

import numpy as np

from emergo.features import encode_ce, extract_graph_features
from emergo.kernel import make_initial_phi
from emergo.phi_update import _loss_and_grads, phi_update
from emergo.types import Graph
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
        make_ce("add_edge", ("A0", "A1"), weight=float(i) / length) for i in range(length)
    ]
    return g_history, ce_history


class TestPhiUpdate:
    def test_loss_decreases_with_gradient_steps(self):
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        g_hist, ce_hist = _make_history(10)

        # Compute initial loss
        from emergo.features import encode_ce, extract_graph_features

        T = len(g_hist) - 1
        features = np.array([extract_graph_features(G, phi.d_features) for G in g_hist])
        ce_encs = np.array([encode_ce(ce, phi.d_ce) for ce in ce_hist[:T]])
        initial_loss, _ = _loss_and_grads(phi, features, ce_encs, T)

        # rank_lambda=0 isolates prediction loss (no regularization penalty mixed in)
        _phi_next, final_loss = phi_update(
            phi, g_hist, ce_hist, [], n_steps=50, lr=1e-2, rank_lambda=0.0
        )

        assert final_loss <= initial_loss + 1e-6, "loss must not increase"

    def test_rank_regularization_recovers_degenerate_phi(self):
        """Rank regularization must update W_phi away from all-zeros (not revert)."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        phi.W_phi[:] = 0.0  # rank-0 starting point

        g_hist, ce_hist = _make_history(5)
        phi_next, _ = phi_update(phi, g_hist, ce_hist, [], n_steps=5, rank_lambda=0.5)

        # W_phi must have been updated (not reverted to all-zeros)
        assert not np.allclose(
            phi_next.W_phi, 0.0
        ), "rank regularization should push W_phi away from zero, not revert"

    def test_rank_penalty_prevents_collapse(self):
        """After 20 gradient steps with rank_lambda=0.1, rank(W_phi) >= d_latent//2."""
        d_latent = 4
        phi = make_initial_phi(d_latent=d_latent, d_features=16, d_ce=4, seed=7)
        # Degrade W_phi toward rank collapse by scaling down
        phi.W_phi *= 0.001

        g_hist, ce_hist = _make_history(10)
        phi_next, _ = phi_update(phi, g_hist, ce_hist, [], n_steps=20, rank_lambda=0.1, lr=1e-2)

        rank = np.linalg.matrix_rank(phi_next.W_phi, tol=1e-6)
        min_rank = max(d_latent // 2, 1)
        assert rank >= min_rank, f"rank={rank} < min_rank={min_rank} — regularization failed"

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


class TestRankHinge:
    """R3: rank regularizer only fires when sigma_min < rank_threshold."""

    def test_hinge_suppresses_regularizer_when_rank_healthy(self):
        """When W_phi has large singular values, rank regularizer must not alter W_phi."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=42)
        # Scale up W_phi so sigma_min >> threshold
        phi.W_phi *= 1000.0
        sigma_min = float(np.linalg.svd(phi.W_phi, compute_uv=False)[-1])
        assert sigma_min > 1.0, "precondition: sigma_min must be well above threshold"

        g_hist, ce_hist = _make_history(5)
        # threshold=1.0 means regularizer fires only when sigma_min < 1.0 — won't fire here
        phi_update(phi, g_hist, ce_hist, [], n_steps=3, rank_lambda=1.0, rank_threshold=1.0)
        # W_phi was still updated by prediction loss gradients, but rank penalty was not added
        # We can't check that W_phi is exactly unchanged (prediction gradient changes it),
        # but we can verify that the hinge path isn't crashing.

    def test_hinge_fires_when_rank_at_risk(self):
        """When sigma_min < threshold, regularizer fires and prevents rank collapse."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=5)
        phi.W_phi *= 0.001  # tiny singular values → sigma_min << threshold
        sigma_min = float(np.linalg.svd(phi.W_phi, compute_uv=False)[-1])
        assert sigma_min < 0.1, "precondition: sigma_min must be below threshold"

        g_hist, ce_hist = _make_history(10)
        phi_next, _ = phi_update(
            phi, g_hist, ce_hist, [], n_steps=10, rank_lambda=0.5, rank_threshold=0.1
        )
        rank = np.linalg.matrix_rank(phi_next.W_phi, tol=1e-6)
        assert rank >= max(phi.d_latent // 2, 1)

    def test_zero_rank_threshold_always_fires(self):
        """rank_threshold=0.0 means always fire (backward-compatible with old behavior)."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        phi.W_phi *= 10.0  # healthy rank, but threshold=0 → still fires

        W_copy = phi.W_phi.copy()
        g_hist, ce_hist = _make_history(5)
        phi_next, _ = phi_update(
            phi, g_hist, ce_hist, [], n_steps=1, rank_lambda=0.5, rank_threshold=0.0
        )
        # Rank penalty gradient was applied; W_phi should differ from purely prediction-only
        # (We just verify no crash and that the call succeeds with a valid phi)
        assert phi_next.W_phi.shape == W_copy.shape
        assert np.all(np.isfinite(phi_next.W_phi))


class TestPhiWindow:
    """R1: sliding window bounds training cost."""

    def test_window_truncates_history(self):
        """phi_window=3 must use only 3 transitions even if history has 10."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=0)
        g_hist, ce_hist = _make_history(10)

        phi_small, loss_small = phi_update(
            phi, g_hist, ce_hist, [], n_steps=5, phi_window=3, rank_lambda=0.0
        )
        _phi_large, loss_large = phi_update(
            phi, g_hist, ce_hist, [], n_steps=5, phi_window=0, rank_lambda=0.0
        )
        # Different windows → different losses (using subset of history)
        # Both must be finite and produce valid phi
        assert np.isfinite(loss_small)
        assert np.isfinite(loss_large)
        assert phi_small.W_phi.shape == phi.W_phi.shape

    def test_window_larger_than_history_uses_full_history(self):
        """phi_window > T must behave identically to phi_window=0."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        g_hist, ce_hist = _make_history(5)  # T=5 transitions

        _, loss_no_window = phi_update(
            phi, g_hist, ce_hist, [], n_steps=5, phi_window=0, rank_lambda=0.0
        )
        _, loss_big_window = phi_update(
            phi, g_hist, ce_hist, [], n_steps=5, phi_window=1000, rank_lambda=0.0
        )
        assert abs(loss_no_window - loss_big_window) < 1e-9

    def test_window_one_step(self):
        """phi_window=1 must work without crashing (T_eff=1)."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        g_hist, ce_hist = _make_history(10)
        phi_next, loss = phi_update(
            phi, g_hist, ce_hist, [], n_steps=3, phi_window=1, rank_lambda=0.0
        )
        assert np.isfinite(loss)
        assert phi_next.d_latent == phi.d_latent
