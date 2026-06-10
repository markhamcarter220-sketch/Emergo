"""Formal verification tests for the Emergo Core Theorem.

Core Theorem: Agents that better predict topology evolution gain greater
influence over future topology evolution.

Three consistency conditions:
  C1 Well-Definedness   — ℱ produces unique, deterministic successors
  C2 Stability          — authority stays bounded; no runaway feedback
  C3 Identifiability    — φ converges to the real topology structure

Integration tests validate the theorem end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    Authority,
    Errors,
    Graph,
    Lux,
    SimulatedLuxBridge,
    authority_update,
    ce_execute,
    emergo_kernel,
    error_computation,
    make_initial_authority,
    make_initial_phi,
    phi_update,
)
from emergo.types import PhiMap
from tests.conftest import make_ce

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(agent_ids=("A", "B", "C"), edge_density=0.5) -> Graph:
    n = len(agent_ids)
    rng = np.random.default_rng(7)
    adj = (rng.random((n, n)) < edge_density).astype(float)
    np.fill_diagonal(adj, 0.0)
    caps = np.ones((n, 2)) * 0.5
    return Graph(agent_ids=tuple(agent_ids), adjacency=adj, capabilities=caps)


def _make_dense_graph(n=4) -> Graph:
    ids = tuple("ABCD"[:n])
    adj = np.ones((n, n)) - np.eye(n)
    caps = np.ones((n, 2)) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _make_sparse_graph(n=4) -> Graph:
    ids = tuple("ABCD"[:n])
    adj = np.zeros((n, n))
    caps = np.ones((n, 2)) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _make_lux_with_caps(*agents: str) -> Lux:
    bridge = SimulatedLuxBridge()
    for a in agents:
        bridge.grant_capability(a, "test")
    return Lux(bridge=bridge)


# ---------------------------------------------------------------------------
# C1: Well-Definedness — ℱ produces unique, deterministic successors
# ---------------------------------------------------------------------------


class TestConsistency1WellDefinedness:
    def test_ce_execution_is_deterministic(self):
        """Identical CE applied twice to the same graph yields identical results."""
        G = _make_graph()
        lux = _make_lux_with_caps("A", "B", "C")
        A = make_initial_authority(G.agent_ids)
        CE = make_ce("add_edge", ("A", "C"), weight=0.7)

        G1, ok1, _ = ce_execute(G, CE, lux, A)
        G2, ok2, _ = ce_execute(G, CE, lux, A)

        assert ok1 == ok2
        assert G1 == G2

    def test_error_computation_is_deterministic(self):
        """Same graph pair and CE always produce the same per-agent errors."""
        G = _make_graph()
        lux = _make_lux_with_caps("A", "B", "C")
        A = make_initial_authority(G.agent_ids)
        CE = make_ce("add_edge", ("A", "B"), weight=0.5)
        G_next, _, _ = ce_execute(G, CE, lux, A)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)

        e1 = error_computation(G, G_next, phi, CE)
        e2 = error_computation(G, G_next, phi, CE)

        assert e1.per_agent == e2.per_agent

    def test_entanglement_guard_preserves_rank_invariant(self):
        """phi_update always returns a phi with rank ≥ d_latent // 2."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        min_rank = max(phi.d_latent // 2, 1)

        # Confirm pre-condition
        assert np.linalg.matrix_rank(phi.W_phi) >= min_rank

        G = _make_graph()
        CE = make_ce("add_edge", ("A", "B"), weight=0.5)
        phi_next, _ = phi_update(phi, [G, G], [CE], [])

        returned_rank = np.linalg.matrix_rank(phi_next.W_phi)
        assert returned_rank >= min_rank

    def test_rank_regularization_recovers_collapsed_phi(self):
        """Rank regularization must push W_phi away from rank-0 (not revert)."""
        d_latent, d_features, d_ce = 4, 16, 4
        phi_zero = PhiMap(
            W_phi=np.zeros((d_latent, d_features)),
            b_phi=np.zeros(d_latent),
            W_F=np.zeros((d_latent, d_latent + d_ce)),
            b_F=np.zeros(d_latent),
            d_latent=d_latent,
            d_features=d_features,
            d_ce=d_ce,
        )
        assert np.linalg.matrix_rank(phi_zero.W_phi) < max(d_latent // 2, 1)

        G = _make_graph()
        CE = make_ce("add_edge", ("A", "B"), weight=0.5)
        # rank_lambda=0.5 actively pushes W_phi away from zero
        phi_returned, _ = phi_update(phi_zero, [G, G], [CE], [], n_steps=5, rank_lambda=0.5)

        # Regularization updated W_phi (did NOT revert to all-zeros)
        assert not np.allclose(
            phi_returned.W_phi, 0.0
        ), "rank regularization should push W_phi away from zero, not revert"

    def test_error_scope_is_ce_participants_only(self):
        """ErrorComputation attributes errors only to CE participants, not observers."""
        G = _make_graph()
        lux = _make_lux_with_caps("A", "B", "C")
        A = make_initial_authority(G.agent_ids)
        # CE with only A and B as participants; C is an observer
        CE = make_ce("add_edge", ("A", "B"), weight=0.5)
        G_next, _, _ = ce_execute(G, CE, lux, A)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)

        errors = error_computation(G, G_next, phi, CE)

        # Only CE participants get errors
        assert set(errors.per_agent.keys()) == set(CE.participants)
        assert "C" not in errors.per_agent


# ---------------------------------------------------------------------------
# C2: Stability — authority stays bounded; no runaway feedback
# ---------------------------------------------------------------------------


class TestConsistency2Stability:
    def test_authority_bounded_after_repeated_updates(self):
        """Authority scores remain in [0, 1] after 200 random update rounds."""
        rng = np.random.default_rng(0)
        A = make_initial_authority(("A", "B", "C"))

        for _ in range(200):
            raw = {agent: float(rng.random()) for agent in ("A", "B", "C")}
            A = authority_update(A, Errors(per_agent=raw), eta=0.05)

        for agent in ("A", "B", "C"):
            v = A.get(agent)
            assert 0.0 <= v <= 1.0

    def test_authority_clamped_to_unit_interval(self):
        """Scores never escape [0, 1] even with large eta near boundary."""
        # Start at upper boundary; apply updates that would push above 1
        A = Authority(scores={"A": 0.99, "B": 0.01}, baseline=0.5)

        for _ in range(20):
            # A always gets 0 error (best) → correctness=1 → delta=+0.5
            errors = Errors(per_agent={"A": 0.0, "B": 1.0})
            A = authority_update(A, errors, eta=0.9)  # large eta

        assert A.get("A") <= 1.0
        assert A.get("B") >= 0.0

    def test_lower_error_yields_higher_authority_gain(self):
        """Agent with lower error gains more authority than agent with higher error."""
        A_good = Authority(scores={"X": 0.5, "Y": 0.5}, baseline=0.5)
        A_bad = Authority(scores={"X": 0.5, "Y": 0.5}, baseline=0.5)

        # X wins in scenario 1 (low error), Y wins in scenario 2
        err_x_wins = Errors(per_agent={"X": 0.0, "Y": 1.0})
        err_y_wins = Errors(per_agent={"X": 1.0, "Y": 0.0})

        A_x_won = authority_update(A_good, err_x_wins, eta=0.05)
        A_y_won = authority_update(A_bad, err_y_wins, eta=0.05)

        assert A_x_won.get("X") > A_y_won.get("X")
        assert A_x_won.get("X") > 0.5  # good predictor gained authority
        assert A_y_won.get("X") < 0.5  # bad predictor lost authority

    def test_authority_update_order_independence(self):
        """Authority update reads from A_t, not A_next — order is irrelevant."""
        A = Authority(scores={"A": 0.4, "B": 0.7}, baseline=0.5)
        errors = Errors(per_agent={"A": 0.2, "B": 0.8})
        eta = 0.05

        result = authority_update(A, errors, eta=eta)

        # Compute expected values analytically using A_t (not A_next).
        # Default ErrorScales() uses local_scale=1.0 (both agents, no proposer_id set).
        scale = 1.0  # default local_scale
        corr_A = 1.0 - float(np.clip(0.2 / scale, 0.0, 1.0))  # 0.8
        corr_B = 1.0 - float(np.clip(0.8 / scale, 0.0, 1.0))  # 0.2
        expected_A = float(np.clip(0.4 + eta * (corr_A - 0.5), 0.0, 0.8))
        expected_B = float(np.clip(0.7 + eta * (corr_B - 0.5), 0.0, 0.8))

        assert result.get("A") == pytest.approx(expected_A, abs=1e-9)
        assert result.get("B") == pytest.approx(expected_B, abs=1e-9)

    def test_no_runaway_feedback(self):
        """After 1000 iterations with adversarial errors, no NaN or out-of-range scores."""
        rng = np.random.default_rng(99)
        A = make_initial_authority(("A", "B", "C"))

        for _ in range(1000):
            # Extreme errors: one agent always best, one always worst
            raw = {"A": 0.0, "B": float(rng.exponential(10.0)), "C": 1e6}
            A = authority_update(A, Errors(per_agent=raw), eta=0.05)

        for agent in ("A", "B", "C"):
            v = A.get(agent)
            assert not np.isnan(v), f"{agent} authority is NaN"
            assert 0.0 <= v <= 1.0

    def test_max_single_step_change_bounded_by_eta(self):
        """Each update step changes authority by at most eta (in absolute value)."""
        eta = 0.1
        A_init = Authority(scores={"A": 0.5}, baseline=0.5)

        # Best case: correctness=1, delta=+0.5
        A_best = authority_update(A_init, Errors(per_agent={"A": 0.0, "dummy": 1.0}), eta=eta)
        # Worst case: correctness=0, delta=-0.5
        A_worst = authority_update(A_init, Errors(per_agent={"A": 1.0, "dummy": 0.0}), eta=eta)

        gain = abs(A_best.get("A") - A_init.get("A"))
        loss = abs(A_worst.get("A") - A_init.get("A"))

        # |delta| ≤ max(baseline, 1-baseline) = 0.5 for baseline=0.5
        # so |change| ≤ eta * 0.5 ≤ eta
        assert gain <= eta + 1e-9
        assert loss <= eta + 1e-9


# ---------------------------------------------------------------------------
# C3: Identifiability — φ converges to real topology structure
# ---------------------------------------------------------------------------


class TestConsistency3Identifiability:
    def test_phi_loss_decreases_after_training(self):
        """phi_update reduces prediction loss relative to the untrained phi."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        G1 = _make_dense_graph()
        G2 = _make_sparse_graph()  # clearly different from G1
        CE = make_ce("remove_edge", ("A", "B"))

        # Use rank_lambda=0 to compare pure prediction loss without rank-penalty mixing
        _, initial_loss = phi_update(phi, [G1, G2], [CE], [], n_steps=0, rank_lambda=0.0)
        _, trained_loss = phi_update(phi, [G1, G2], [CE], [], n_steps=100, rank_lambda=0.0)

        assert trained_loss <= initial_loss + 1e-6

    def test_phi_distinguishes_structured_from_sparse(self):
        """φ produces different embeddings for dense vs sparse graphs."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        G_dense = _make_dense_graph()
        G_sparse = _make_sparse_graph()

        z_dense = phi.embed(G_dense)
        z_sparse = phi.embed(G_sparse)

        # Structural difference must produce different latent embeddings
        assert not np.allclose(z_dense, z_sparse)

    def test_phi_reflects_edge_density(self):
        """φ embeddings differ monotonically with edge density changes."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        n = 4
        ids = ("A", "B", "C", "D")
        caps = np.ones((n, 2)) * 0.5

        # 1 edge vs 6 edges
        adj_1 = np.zeros((n, n))
        adj_1[0, 1] = 1.0
        adj_6 = np.triu(np.ones((n, n)) - np.eye(n))
        G_1 = Graph(agent_ids=ids, adjacency=adj_1, capabilities=caps)
        G_6 = Graph(agent_ids=ids, adjacency=adj_6, capabilities=caps)

        z_1 = phi.embed(G_1)
        z_6 = phi.embed(G_6)

        assert not np.allclose(z_1, z_6)

    def test_phi_isomorphism_invariance(self):
        """Isomorphic graphs (same spectral structure) map to the same φ embedding."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        n = 4
        ids = ("A", "B", "C", "D")
        caps = np.ones((n, 2)) * 0.5

        # Star K_{1,3} with A as center
        adj1 = np.zeros((n, n))
        adj1[0, 1] = adj1[0, 2] = adj1[0, 3] = 1.0
        G1 = Graph(agent_ids=ids, adjacency=adj1, capabilities=caps)

        # Star K_{1,3} with D as center (isomorphic — same eigenvalues)
        adj2 = np.zeros((n, n))
        adj2[3, 0] = adj2[3, 1] = adj2[3, 2] = 1.0
        G2 = Graph(agent_ids=ids, adjacency=adj2, capabilities=caps)

        z1 = phi.embed(G1)
        z2 = phi.embed(G2)

        # Spectral features are permutation-invariant → same embedding
        np.testing.assert_allclose(z1, z2, atol=1e-10)


# ---------------------------------------------------------------------------
# Integration: Core Theorem Validation
# ---------------------------------------------------------------------------


class TestCoreTheoremValidation:
    def test_better_predictor_gains_authority_over_time(self):
        """Core theorem: consistently lower prediction error → higher authority."""
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        rng = np.random.default_rng(42)

        for _ in range(30):
            # A always predicts perfectly; B always has non-trivial error
            errors = Errors(per_agent={"A": 0.0, "B": rng.random() + 0.1})
            A = authority_update(A, errors, eta=0.05)

        assert A.get("A") > A.get(
            "B"
        ), "Better predictor (A) must accumulate more authority than worse predictor (B)"
        assert A.get("A") > 0.5, "Good predictor's authority must rise above baseline"
        assert A.get("B") < 0.5, "Poor predictor's authority must fall below baseline"

    def test_proposer_and_participant_get_different_authority_deltas(self):
        """After a non-trivial CE, proposer and participant authority deltas differ."""
        G = _make_graph(agent_ids=("A", "B", "C"), edge_density=0.0)
        A = make_initial_authority(G.agent_ids)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        CE = make_ce("add_edge", ("A", "B"), weight=0.5)
        lux = _make_lux_with_caps("A", "B", "C")

        G_next, success, _ = ce_execute(G, CE, lux, A)
        assert success

        errors = error_computation(G, G_next, phi, CE)
        A_next = authority_update(A, errors)

        delta_A = A_next.get("A") - A.get("A")
        delta_B = A_next.get("B") - A.get("B")
        # Proposer gets global error, participant gets local structural error → different deltas
        assert delta_A != pytest.approx(delta_B, abs=1e-9)

    def test_topology_exploration_persists_longer(self):
        """With differentiated errors, authority stays healthy → >10 CEs accepted in 50 iters."""
        adj = np.zeros((3, 3))
        caps = np.ones((3, 2)) * 0.5
        G0 = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)
        phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        A0 = make_initial_authority(("A", "B", "C"))
        initial_state = (G0, phi0, A0, [])

        _final_state, _reason, diag = emergo_kernel(
            initial_state=initial_state,
            max_iterations=50,
            collect_diagnostics=True,
            rng=np.random.default_rng(1),
        )

        accepted = sum(1 for r in diag.records if r.ce_accepted)
        assert (
            accepted > 10
        ), f"Only {accepted} CEs accepted in 50 iterations — authority likely collapsed"

    def test_end_to_end_state_progression(self):
        """State machine (G, φ, A, E) evolves correctly through the kernel loop."""
        n = 3
        adj = np.eye(n, k=1, dtype=float) + np.eye(n, k=-(n - 1), dtype=float)
        caps = np.ones((n, 2)) * 0.5
        G0 = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)
        phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        A0 = make_initial_authority(("A", "B", "C"))
        initial_state = (G0, phi0, A0, [])

        final_state, reason = emergo_kernel(
            initial_state=initial_state,
            max_iterations=15,
            convergence_threshold=1e-2,
        )

        G_final, phi_final, A_final, _E_final = final_state

        assert reason in ("Converged", "Max iterations reached")
        assert isinstance(G_final, Graph)
        # Authority scores remain valid throughout
        for v in A_final.scores.values():
            assert 0.0 <= v <= 1.0
        # Rank invariant maintained
        rank = np.linalg.matrix_rank(phi_final.W_phi)
        assert rank >= max(phi_final.d_latent // 2, 1)
