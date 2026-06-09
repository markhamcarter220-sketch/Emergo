"""Tests for ErrorComputation: determinism, correct error signal, per-agent tracking."""

import numpy as np
import pytest

from emergo.error_computation import error_computation
from emergo.types import CoordinationEvent
from tests.conftest import make_ce


class TestErrorComputation:
    def test_returns_errors_for_all_participants(self, three_agent_graph, phi, lux):
        ce = make_ce("add_edge", ("A", "B"), weight=1.0)
        from emergo.ce_execution import ce_execute
        from emergo.types import Authority

        auth = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        G_next, _, _ = ce_execute(three_agent_graph, ce, lux, auth)
        errors = error_computation(three_agent_graph, G_next, phi, ce)
        assert set(errors.per_agent.keys()) == {"A", "B"}

    def test_error_is_nonnegative(self, three_agent_graph, phi, lux):
        ce = make_ce("remove_edge", ("A", "B"))
        from emergo.ce_execution import ce_execute
        from emergo.types import Authority

        auth = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        G_next, _, _ = ce_execute(three_agent_graph, ce, lux, auth)
        errors = error_computation(three_agent_graph, G_next, phi, ce)
        for err in errors.per_agent.values():
            assert err >= 0.0

    def test_identical_graphs_give_zero_error_on_trivial_phi(self, three_agent_graph, phi):
        """When G_t == G_{t+1}, predicted and actual embed identically → zero residual."""
        # Only true when F(z, c) ≈ z, which holds when W_F ≈ [I | 0] and b_F ≈ 0.
        # With random near-zero init, z_pred ≈ b_F ≈ 0 and z_actual ≈ 0 too,
        # so error should be small (not exactly zero, but close).
        ce = make_ce("add_edge", ("A", "B"), weight=0.0)  # no-op weight
        errors = error_computation(three_agent_graph, three_agent_graph, phi, ce)
        # With near-zero W_F, z_pred ≈ 0 and z_actual ≈ 0, so error is small
        assert errors.mean_error() < 1.0  # sanity bound

    def test_deterministic_for_same_inputs(self, three_agent_graph, phi):
        ce = make_ce("add_edge", ("B", "C"), weight=0.5)
        e1 = error_computation(three_agent_graph, three_agent_graph, phi, ce)
        e2 = error_computation(three_agent_graph, three_agent_graph, phi, ce)
        assert e1.per_agent == e2.per_agent

    def test_phi_not_mutated(self, three_agent_graph, phi):
        W_phi_before = phi.W_phi.copy()
        ce = make_ce("add_edge", ("A", "C"), weight=0.3)
        error_computation(three_agent_graph, three_agent_graph, phi, ce)
        np.testing.assert_array_equal(phi.W_phi, W_phi_before)

    def test_empty_participants_gives_empty_errors(self, three_agent_graph, phi):
        ce = CoordinationEvent(event_type="add_edge", participants=(), params=frozenset())
        errors = error_computation(three_agent_graph, three_agent_graph, phi, ce)
        assert errors.per_agent == {}

    def test_proposer_gets_global_phi_error(self, three_agent_graph, phi, lux):
        """Proposer error equals ||z_predicted - z_actual||."""
        from emergo.features import encode_ce

        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        from emergo.ce_execution import ce_execute
        from emergo.types import Authority

        auth = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        G_next, _, _ = ce_execute(three_agent_graph, ce, lux, auth)

        ce_enc = encode_ce(ce, phi.d_ce)
        z_t = phi.embed(three_agent_graph)
        z_pred = phi.transition(z_t, ce_enc)
        z_actual = phi.embed(G_next)
        expected_global = float(np.linalg.norm(z_pred - z_actual))

        errors = error_computation(three_agent_graph, G_next, phi, ce)
        proposer = ce.participants[0]
        assert errors.per_agent[proposer] == pytest.approx(expected_global, abs=1e-9)

    def test_participant_gets_local_structural_error(self, three_agent_graph, phi, lux):
        """Non-proposing participant error equals adjacency delta norm."""
        from emergo.error_computation import _agent_local_error

        ce = make_ce("add_edge", ("A", "B"), weight=0.7)
        from emergo.ce_execution import ce_execute
        from emergo.types import Authority

        auth = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        G_next, _, _ = ce_execute(three_agent_graph, ce, lux, auth)

        errors = error_computation(three_agent_graph, G_next, phi, ce)
        participant = ce.participants[1]  # "B"
        expected_local = _agent_local_error(three_agent_graph, G_next, participant)
        assert errors.per_agent[participant] == pytest.approx(expected_local, abs=1e-9)

    def test_errors_differ_for_proposer_and_participant(self, three_agent_graph, phi, lux):
        """For a non-trivial CE, proposer and participant have different errors."""
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        from emergo.ce_execution import ce_execute
        from emergo.types import Authority

        auth = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        G_next, success, _ = ce_execute(three_agent_graph, ce, lux, auth)
        assert success, "CE must be accepted for this test to be meaningful"

        errors = error_computation(three_agent_graph, G_next, phi, ce)
        proposer_err = errors.per_agent["A"]
        participant_err = errors.per_agent["B"]
        # Global φ-error differs from local structural delta
        assert proposer_err != pytest.approx(participant_err, abs=1e-9)
