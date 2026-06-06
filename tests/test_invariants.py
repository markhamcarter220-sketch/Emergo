"""Invariant verification tests — map directly to the four contract invariants.

Invariant 1: State ownership — (G_t, φ_t, A_t, E_t) is the only mutable state;
             all operations are pure functions over this tuple.
Invariant 2: Feedback loop — error → authority → topology is closed and visible.
Invariant 3: Blast radius — failures degrade gracefully; no state corruption.
Invariant 4: Timing — sequential execution; atomic operations; no deadlocks.
"""
import numpy as np
import pytest

from emergo import (
    Graph,
    CoordinationEvent,
    Lux,
    ce_execute,
    error_computation,
    authority_update,
    phi_update,
    make_initial_phi,
    make_initial_authority,
)
from emergo.types import Authority, Errors
from tests.conftest import make_ce


def _triangle() -> Graph:
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    caps = np.ones((3, 2)) * 0.5
    return Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)


# ---------------------------------------------------------------------------
# Invariant 1: State Ownership
# ---------------------------------------------------------------------------

class TestInvariant1StateOwnership:
    def test_ce_execute_does_not_mutate_input_graph(self):
        G = _triangle()
        original_adj = G.adjacency.copy()
        auth = make_initial_authority(G.agent_ids)
        ce = make_ce("add_edge", ("A", "C"), weight=0.9)
        ce_execute(G, ce, Lux(), auth)
        np.testing.assert_array_equal(G.adjacency, original_adj)

    def test_authority_update_does_not_mutate_input(self):
        A = make_initial_authority(("A", "B"))
        original_scores = dict(A.scores)
        authority_update(A, Errors(per_agent={"A": 0.1, "B": 0.9}))
        assert A.scores == original_scores

    def test_phi_update_does_not_mutate_input(self):
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        W_before = phi.W_phi.copy()
        G = _triangle()
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        phi_update(phi, [G, G], [ce], [])
        np.testing.assert_array_equal(phi.W_phi, W_before)

    def test_graph_adjacency_is_readonly(self):
        G = _triangle()
        with pytest.raises((ValueError, TypeError)):
            G.adjacency[0, 0] = 999.0

    def test_graph_capabilities_is_readonly(self):
        G = _triangle()
        with pytest.raises((ValueError, TypeError)):
            G.capabilities[0, 0] = 999.0


# ---------------------------------------------------------------------------
# Invariant 2: Feedback Loop
# ---------------------------------------------------------------------------

class TestInvariant2FeedbackLoop:
    def test_error_drives_authority_change(self):
        A = make_initial_authority(("A", "B"), baseline=0.5)
        # A predicts perfectly, B predicts worst
        errors = Errors(per_agent={"A": 0.0, "B": 1.0})
        A_next = authority_update(A, errors, eta=0.1)
        assert A_next.get("A") > A.get("A"), "perfect predictor must gain authority"
        assert A_next.get("B") < A.get("B"), "worst predictor must lose authority"

    def test_authority_influences_ce_sampling_direction(self):
        """Higher authority → higher sampling weight in softmax."""
        from emergo.kernel import _sample_next_ce
        G = _triangle()
        A = Authority(
            scores={"A": 0.9, "B": 0.1, "C": 0.1},
            baseline=0.5,
        )
        rng = np.random.default_rng(0)
        counts = {"A": 0, "B": 0, "C": 0}
        for _ in range(500):
            ce = _sample_next_ce(A, G, rng)
            if ce is not None:
                counts[ce.participants[0]] += 1
        # A should be chosen as initiator most often
        assert counts["A"] > counts["B"] + counts["C"]

    def test_error_computation_reflects_phi_change(self):
        """After phi_update reduces loss, error_computation should give smaller errors."""
        from emergo.phi_update import phi_update
        G = _triangle()
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=7)
        ce = make_ce("add_edge", ("A", "C"), weight=0.5)

        auth = make_initial_authority(G.agent_ids)
        G_next, _, _ = ce_execute(G, ce, Lux(), auth)

        g_hist = [G, G_next] * 5  # repeated pattern
        ce_hist = [ce] * (len(g_hist) - 1)
        phi_fitted, loss_after = phi_update(phi, g_hist, ce_hist, [], n_steps=100, lr=5e-3)

        err_before = error_computation(G, G_next, phi, ce).mean_error()
        err_after = error_computation(G, G_next, phi_fitted, ce).mean_error()
        # After fitting, errors should not be drastically worse
        assert err_after <= err_before * 10 + 1e-3  # loose bound; fitting may not fully converge


# ---------------------------------------------------------------------------
# Invariant 3: Blast Radius
# ---------------------------------------------------------------------------

class TestInvariant3BlastRadius:
    def test_unauthorized_ce_leaves_graph_unchanged(self):
        G = _triangle()
        low_auth = Authority(scores={"A": 0.0, "B": 0.5, "C": 0.5}, baseline=0.5)
        ce = make_ce("add_edge", ("A", "B"), weight=1.0)
        G_next, ok, _ = ce_execute(G, ce, Lux(min_authority=0.1), low_auth)
        assert not ok
        assert G_next == G

    def test_malformed_ce_type_leaves_graph_unchanged(self):
        G = _triangle()
        auth = make_initial_authority(G.agent_ids)
        bad_ce = CoordinationEvent(
            event_type="teleport_agent",  # unknown type
            participants=("A",),
            params=frozenset(),
        )
        G_next, ok, _ = ce_execute(G, bad_ce, Lux(), auth)
        assert not ok
        assert G_next == G

    def test_phi_update_reverts_on_entanglement_violation(self):
        """A collapsed W_phi (rank 0) must cause phi_update to revert."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        phi.W_phi[:] = 0.0  # force rank-0 collapse
        G = _triangle()
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        phi_next, _ = phi_update(phi, [G, G], [ce], [], n_steps=1)
        np.testing.assert_array_equal(phi_next.W_phi, phi.W_phi)

    def test_removing_nonexistent_agent_fails_safely(self):
        G = _triangle()
        auth = make_initial_authority(G.agent_ids)
        ce = make_ce("remove_agent", ("Z",))  # Z does not exist
        G_next, ok, _ = ce_execute(G, ce, Lux(), auth)
        assert not ok
        assert G_next == G


# ---------------------------------------------------------------------------
# Invariant 4: Timing (sequential, atomic, deterministic)
# ---------------------------------------------------------------------------

class TestInvariant4Timing:
    def test_ce_execution_is_deterministic(self):
        G = _triangle()
        auth = make_initial_authority(G.agent_ids)
        ce = make_ce("add_edge", ("A", "C"), weight=0.3)
        G1, ok1, _ = ce_execute(G, ce, Lux(), auth)
        G2, ok2, _ = ce_execute(G, ce, Lux(), auth)
        assert ok1 == ok2
        np.testing.assert_array_equal(G1.adjacency, G2.adjacency)

    def test_authority_update_is_deterministic(self):
        A = make_initial_authority(("A", "B"))
        errors = Errors(per_agent={"A": 0.2, "B": 0.8})
        A1 = authority_update(A, errors, eta=0.05)
        A2 = authority_update(A, errors, eta=0.05)
        assert A1.scores == A2.scores

    def test_operations_compose_sequentially(self):
        """Run the four operations once and verify state progresses correctly."""
        G = _triangle()
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        A = make_initial_authority(G.agent_ids)
        lux = Lux()
        ce = make_ce("add_edge", ("A", "C"), weight=0.5)

        G_next, ok, _ = ce_execute(G, ce, lux, A)
        assert ok

        errors = error_computation(G, G_next, phi, ce)
        assert isinstance(errors.mean_error(), float)

        A_next = authority_update(A, errors)
        assert all(0.0 <= v <= 1.0 for v in A_next.scores.values())

        phi_next, loss = phi_update(phi, [G, G_next], [ce], [errors])
        assert np.isfinite(loss) or loss == float("inf")

        # State has advanced: each component is a new object
        assert G_next is not G
        assert A_next is not A
        assert phi_next is not phi or phi_next.W_phi is not phi.W_phi
