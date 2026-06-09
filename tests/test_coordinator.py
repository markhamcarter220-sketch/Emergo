"""Tests for MultiAgentCoordinator — INV-9 and coordination semantics.

Verifies:
  - Authority-ordered serialization (INV-9)
  - Conflict detection and rejection
  - Per-proposal audit trail (INV-7)
  - No resource charge on conflict rejection (INV-6)
  - State consistency after accepted CEs
  - max_agents cap enforcement
  - Empty and single-proposal edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    CoordinationRound,
    Graph,
    Lux,
    MultiAgentCoordinator,
    ProposedCE,
    make_initial_authority,
    make_initial_phi,
)
from emergo.lux_bridge import SimulatedLuxBridge
from emergo.types import Authority, CoordinationEvent
from tests.conftest import make_ce

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_graph(agent_ids=("A", "B", "C", "D")) -> Graph:
    n = len(agent_ids)
    adj = np.zeros((n, n), dtype=float)
    caps = np.ones((n, 2), dtype=float) * 0.5
    return Graph(agent_ids=tuple(agent_ids), adjacency=adj, capabilities=caps)


def _state(agent_ids=("A", "B", "C", "D")):
    G = _empty_graph(agent_ids)
    phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
    A = make_initial_authority(agent_ids, baseline=0.5)
    return G, phi, A, []


def _lux_with_all(*agent_ids: str) -> Lux:
    bridge = SimulatedLuxBridge(initial_budget=100.0)
    return Lux(bridge=bridge)


# ---------------------------------------------------------------------------
# 1. Empty proposals
# ---------------------------------------------------------------------------


class TestEmptyProposals:
    def test_empty_proposals_returns_empty_round(self):
        coord = MultiAgentCoordinator(lux=_lux_with_all())
        state = _state()
        round_ = coord.coordinate([], state)
        assert isinstance(round_, CoordinationRound)
        assert round_.accepted == []
        assert round_.rejected == []
        assert round_.n_conflicts_detected == 0
        assert round_.audit_ids == []

    def test_empty_proposals_state_unchanged(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=_lux_with_all())
        round_ = coord.coordinate([], state)
        G_in, _phi_in, A_in, E_in = state
        G_out, _phi_out, A_out, E_out = round_.final_state
        assert G_out == G_in
        assert A_out.scores == A_in.scores
        assert E_out == E_in


# ---------------------------------------------------------------------------
# 2. Single proposal
# ---------------------------------------------------------------------------


class TestSingleProposal:
    def test_single_accepted(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=_lux_with_all("A", "B", "C", "D"))
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        round_ = coord.coordinate([ProposedCE("A", ce)], state)
        assert len(round_.accepted) == 1
        assert len(round_.rejected) == 0
        assert len(round_.audit_ids) == 1

    def test_single_accepted_graph_mutated(self):
        state = _state()
        G_in = state[0]
        coord = MultiAgentCoordinator(lux=_lux_with_all())
        ce = make_ce("add_edge", ("A", "B"), weight=0.7)
        round_ = coord.coordinate([ProposedCE("A", ce)], state)
        G_out = round_.final_state[0]
        assert G_out is not G_in
        a_idx = G_out.agent_ids.index("A")
        b_idx = G_out.agent_ids.index("B")
        assert G_out.adjacency[a_idx, b_idx] == pytest.approx(0.7)

    def test_single_rejected_low_authority(self):
        state = _state()
        G, phi, A, E = state
        # Force authority below min_authority threshold
        A = Authority(scores={aid: 0.0 for aid in G.agent_ids}, baseline=0.5)
        state = (G, phi, A, E)
        coord = MultiAgentCoordinator(lux=Lux(min_authority=0.1))
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        round_ = coord.coordinate([ProposedCE("A", ce)], state)
        assert len(round_.accepted) == 0
        assert len(round_.rejected) == 1
        assert round_.rejection_reasons["A"] == "ce_execute_rejected"


# ---------------------------------------------------------------------------
# 3. INV-9: Authority-ordered serialization
# ---------------------------------------------------------------------------


class TestAuthorityOrdering:
    def test_higher_authority_proposal_processed_first(self):
        """With A-priority=0.9 and B-priority=0.1, A's proposal on a shared
        edge wins; B's is detected as a conflict and rejected."""
        G = _empty_graph(("A", "B", "C"))
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        A = Authority(scores={"A": 0.9, "B": 0.1, "C": 0.5}, baseline=0.5)
        state = (G, phi, A, [])

        coord = MultiAgentCoordinator(lux=Lux())
        # Both propose the same edge A→C
        ce_a = make_ce("add_edge", ("A", "C"), weight=0.8)
        ce_b = make_ce("add_edge", ("A", "C"), weight=0.3)
        round_ = coord.coordinate(
            [
                ProposedCE("B", ce_b),  # submitted first but lower priority
                ProposedCE("A", ce_a),  # submitted second but higher priority
            ],
            state,
        )

        # A should win (higher authority), B should be conflict-rejected
        accepted_agents = [p.agent_id for p in round_.accepted]
        rejected_agents = [p.agent_id for p in round_.rejected]
        assert "A" in accepted_agents
        assert "B" in rejected_agents
        assert round_.n_conflicts_detected == 1

    def test_equal_authority_stable_sort(self):
        """When priorities are equal, the sort is stable (input order preserved)."""
        G = _empty_graph(("A", "B", "C"))
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        A = Authority(scores={"A": 0.5, "B": 0.5, "C": 0.5}, baseline=0.5)
        state = (G, phi, A, [])

        coord = MultiAgentCoordinator(lux=Lux())
        # Non-conflicting edges
        ce_a = make_ce("add_edge", ("A", "B"), weight=0.5)
        ce_b = make_ce("add_edge", ("B", "C"), weight=0.5)
        round_ = coord.coordinate(
            [
                ProposedCE("A", ce_a),
                ProposedCE("B", ce_b),
            ],
            state,
        )
        # Both should be accepted (non-conflicting edges)
        assert len(round_.accepted) == 2
        assert len(round_.rejected) == 0

    def test_priority_override_respected(self):
        """Explicit priority overrides authority score for ordering."""
        G = _empty_graph(("A", "B", "C"))
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        A = Authority(scores={"A": 0.1, "B": 0.9, "C": 0.5}, baseline=0.5)
        state = (G, phi, A, [])

        # A has low authority but a high explicit priority
        # Both target the same edge: A→C wins because of override
        coord = MultiAgentCoordinator(lux=Lux())
        ce_a = make_ce("add_edge", ("A", "C"), weight=0.5)
        ce_b = make_ce("add_edge", ("A", "C"), weight=0.5)
        round_ = coord.coordinate(
            [
                ProposedCE("A", ce_a, priority=0.99),  # explicit override
                ProposedCE("B", ce_b),  # falls back to authority=0.9
            ],
            state,
        )

        accepted_agents = [p.agent_id for p in round_.accepted]
        assert "A" in accepted_agents
        assert "B" not in accepted_agents


# ---------------------------------------------------------------------------
# 4. Conflict detection
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_same_edge_two_agents_conflict(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=Lux())
        # Both propose the same directed edge
        ce_a = make_ce("add_edge", ("A", "B"), weight=0.5)
        ce_b = make_ce("add_edge", ("A", "B"), weight=0.3)
        round_ = coord.coordinate(
            [
                ProposedCE("A", ce_a, priority=0.9),
                ProposedCE("B", ce_b, priority=0.1),
            ],
            state,
        )
        assert round_.n_conflicts_detected == 1
        assert len(round_.accepted) == 1
        assert len(round_.rejected) == 1

    def test_different_edges_no_conflict(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=Lux())
        round_ = coord.coordinate(
            [
                ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
                ProposedCE("B", make_ce("add_edge", ("C", "D"), weight=0.5)),
            ],
            state,
        )
        assert round_.n_conflicts_detected == 0
        assert len(round_.accepted) == 2

    def test_non_edge_ces_do_not_conflict(self):
        """update_capabilities CEs have no edge key — multiple can coexist."""
        G = _empty_graph(("A", "B"))
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=1)
        A = make_initial_authority(("A", "B"))
        state = (G, phi, A, [])
        coord = MultiAgentCoordinator(lux=Lux())
        ce_a = CoordinationEvent(
            event_type="update_capabilities",
            participants=("A",),
            params=frozenset([("capabilities", (0.7, 0.3))]),
        )
        ce_b = CoordinationEvent(
            event_type="update_capabilities",
            participants=("B",),
            params=frozenset([("capabilities", (0.6, 0.4))]),
        )
        round_ = coord.coordinate(
            [
                ProposedCE("A", ce_a),
                ProposedCE("B", ce_b),
            ],
            state,
        )
        assert round_.n_conflicts_detected == 0

    def test_conflict_rejection_reason_recorded(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=Lux())
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        round_ = coord.coordinate(
            [
                ProposedCE("A", ce, priority=0.9),
                ProposedCE("B", ce, priority=0.1),
            ],
            state,
        )
        assert "B" in round_.rejection_reasons
        assert (
            "conflict" in round_.rejection_reasons["B"].lower()
            or "A" in round_.rejection_reasons["B"]
        )


# ---------------------------------------------------------------------------
# 5. INV-7: Audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_every_proposal_produces_audit_record(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)
        state = _state()
        coord = MultiAgentCoordinator(lux=lux)
        round_ = coord.coordinate(
            [
                ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
                ProposedCE("B", make_ce("add_edge", ("C", "D"), weight=0.5)),
            ],
            state,
        )
        total_proposals = len(round_.accepted) + len(round_.rejected)
        assert len(round_.audit_ids) == total_proposals
        assert len(bridge.get_audit_log()) >= total_proposals

    def test_conflict_rejected_proposal_is_audited(self):
        bridge = SimulatedLuxBridge()
        lux = Lux(bridge=bridge)
        state = _state()
        coord = MultiAgentCoordinator(lux=lux)
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        coord.coordinate(
            [
                ProposedCE("A", ce, priority=0.9),
                ProposedCE("B", ce, priority=0.1),
            ],
            state,
        )
        log = bridge.get_audit_log()
        conflict_records = [r for r in log if r.get("details", {}).get("reason") == "conflict"]
        assert len(conflict_records) >= 1

    def test_audit_ids_are_unique(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=Lux())
        round_ = coord.coordinate(
            [
                ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
                ProposedCE("B", make_ce("add_edge", ("C", "D"), weight=0.5)),
            ],
            state,
        )
        assert len(set(round_.audit_ids)) == len(round_.audit_ids)


# ---------------------------------------------------------------------------
# 6. INV-6: No resource charge on conflict rejection
# ---------------------------------------------------------------------------


class TestResourceConservation:
    def test_conflict_rejection_charges_no_resources(self):
        """Conflict-rejected proposals must not deduct any ledger balance."""
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)
        state = _state()
        coord = MultiAgentCoordinator(lux=lux)
        # B's proposal conflicts with A's and will be conflict-rejected
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        coord.coordinate(
            [
                ProposedCE("A", ce, priority=0.9),
                ProposedCE("B", ce, priority=0.1),
            ],
            state,
        )
        # coordinator uses check-only Lux auth (no resource deduction)
        # so balance stays at 100.0
        assert bridge.get_balance("A") == pytest.approx(100.0)
        assert bridge.get_balance("B") == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 7. INV-8: max_agents cap
# ---------------------------------------------------------------------------


class TestMaxAgentsCap:
    def test_excess_proposals_truncated(self):
        state = _state(("A", "B", "C", "D"))
        coord = MultiAgentCoordinator(lux=Lux(), max_agents=2)
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
            ProposedCE("B", make_ce("add_edge", ("C", "D"), weight=0.5)),
            ProposedCE("C", make_ce("add_edge", ("B", "C"), weight=0.5)),
            ProposedCE("D", make_ce("add_edge", ("A", "C"), weight=0.5)),
        ]
        round_ = coord.coordinate(proposals, state)
        # Only first 2 were considered
        total = len(round_.accepted) + len(round_.rejected)
        assert total <= 2

    def test_default_max_agents_is_four(self):
        coord = MultiAgentCoordinator(lux=Lux())
        assert coord._max_agents == 4


# ---------------------------------------------------------------------------
# 8. State consistency
# ---------------------------------------------------------------------------


class TestStateConsistency:
    def test_authority_updates_after_accepted_ce(self):
        """After an accepted CE, authority scores must change for participants."""
        state = _state(("A", "B", "C"))
        A_before = state[2]
        coord = MultiAgentCoordinator(lux=Lux())
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        round_ = coord.coordinate([ProposedCE("A", ce)], state)
        if round_.accepted:
            A_after = round_.final_state[2]
            # At least one score changed (proposer gets global error)
            changed = any(A_after.get(aid) != A_before.get(aid) for aid in ("A", "B"))
            assert changed

    def test_error_history_grows_with_accepted_ces(self):
        """E_history must grow by one entry per accepted CE."""
        state = _state(("A", "B", "C"))
        E_before = state[3]
        coord = MultiAgentCoordinator(lux=Lux())
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
            ProposedCE("B", make_ce("add_edge", ("B", "C"), weight=0.3)),
        ]
        round_ = coord.coordinate(proposals, state)
        E_after = round_.final_state[3]
        assert len(E_after) == len(E_before) + len(round_.accepted)

    def test_graph_immutability_preserved(self):
        """Input graph must not be mutated after coordination."""
        state = _state()
        G_in = state[0]
        adj_before = G_in.adjacency.copy()
        coord = MultiAgentCoordinator(lux=Lux())
        coord.coordinate(
            [
                ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
            ],
            state,
        )
        np.testing.assert_array_equal(G_in.adjacency, adj_before)

    def test_round_id_is_unique_uuid(self):
        state = _state()
        coord = MultiAgentCoordinator(lux=Lux())
        r1 = coord.coordinate([], state)
        r2 = coord.coordinate([], state)
        assert r1.round_id != r2.round_id
        # Both should be valid UUIDs (no exception)
        import uuid

        uuid.UUID(r1.round_id)
        uuid.UUID(r2.round_id)
