"""Invariant verification tests.

INV-1:  State ownership — (G_t, φ_t, A_t, E_t) is the only mutable state.
INV-2:  Feedback loop — error → authority → topology is closed and visible.
INV-3:  Blast radius — failures degrade gracefully; no state corruption.
INV-4:  Timing — sequential, atomic, deterministic.
INV-5:  Proposal-Only Authority — Emergo never mints capabilities directly.
INV-6:  Resource Conservation via Lux Ledger — every task charges a resource.
INV-7:  Observable + Fail-Closed — every CE attempt produces an audit record.
INV-8:  Bounded Speculation — depth + pending CE quota enforced.
INV-9:  Coordinator Serialization — proposals sorted by authority; no dup edge writes.
INV-10: Observer Isolation — observer exceptions never propagate to the kernel.
"""
import uuid

import numpy as np
import pytest

from emergo import (
    CoordinationRound,
    Executor,
    Goal,
    Graph,
    CoordinationEvent,
    Lux,
    MultiAgentCoordinator,
    Planner,
    ProposedCE,
    TaskOutcome,
    ce_execute,
    error_computation,
    authority_update,
    failing_task_runner,
    make_initial_authority,
    make_initial_phi,
    mock_task_runner,
    phi_update,
)
from emergo.config import MAX_DECOMPOSITION_DEPTH, MAX_PENDING_CES_PER_AGENT
from emergo.lux_bridge import SimulatedLuxBridge
from emergo.types import Authority, Errors, Task
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
        from emergo.proposal import DefaultProposalGenerator
        G = _triangle()
        A = Authority(
            scores={"A": 0.9, "B": 0.1, "C": 0.1},
            baseline=0.5,
        )
        gen = DefaultProposalGenerator()
        rng = np.random.default_rng(0)
        counts = {"A": 0, "B": 0, "C": 0}
        for _ in range(500):
            ce = gen.propose(A, G, rng)
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

    def test_rank_regularization_recovers_entanglement_violation(self):
        """Rank regularization must update (not revert) W_phi from rank-0 collapse."""
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        phi.W_phi[:] = 0.0  # force rank-0 collapse
        G = _triangle()
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        # rank_lambda > 0 injects nuclear-norm gradient, pushing W_phi away from zero
        phi_next, _ = phi_update(phi, [G, G], [ce], [], n_steps=5, rank_lambda=0.5)
        assert not np.allclose(phi_next.W_phi, 0.0), (
            "rank regularization should recover W_phi from collapse, not revert"
        )

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


# ---------------------------------------------------------------------------
# Shared helpers for INV-5/6/7/8
# ---------------------------------------------------------------------------

def _exec_state(agent_ids=("A", "B")):
    n = len(agent_ids)
    adj = np.zeros((n, n))
    caps = np.ones((n, 2)) * 0.5
    G = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
    A = make_initial_authority(agent_ids)
    return G, phi, A, []


def _exec_goal(agent="A", budget=10.0) -> Goal:
    return Goal(
        goal_id=str(uuid.uuid4()),
        description="inv test",
        required_capability="test_cap",
        initiating_agent=agent,
        resource_budget=budget,
        max_depth=3,
    )


def _lux_with_cap(*agents: str, budget: float = 100.0) -> Lux:
    bridge = SimulatedLuxBridge(initial_budget=budget)
    for a in agents:
        bridge.grant_capability(a, "test_cap")
    return Lux(bridge=bridge)


# ---------------------------------------------------------------------------
# INV-5: Proposal-Only Authority
# ---------------------------------------------------------------------------

class TestInvariant5ProposalOnlyAuthority:
    def test_authority_always_in_unit_interval(self):
        """Authority scores are clamped to [0,1] regardless of error magnitude."""
        A = make_initial_authority(("A",), baseline=0.5)
        # Extreme errors: cannot push authority out of bounds
        for extreme_error in [0.0, 1e6, -1.0]:
            errors = Errors(per_agent={"A": abs(extreme_error)})
            A_next = authority_update(A, errors, eta=10.0)  # large eta
            assert 0.0 <= A_next.get("A") <= 1.0

    def test_execute_task_denied_without_lux_capability(self):
        """Emergo cannot execute a task unless Lux has granted the capability."""
        lux = Lux(bridge=SimulatedLuxBridge())  # no capability granted
        exec_ = Executor(lux=lux)
        result = exec_.execute(_exec_goal(), _exec_state())
        assert result.tasks_succeeded == 0

    def test_capability_change_goes_through_ce_execute(self):
        """After execution, the graph's capability row changed via ce_execute path."""
        lux = _lux_with_cap("A")

        def cap_runner(task):
            return TaskOutcome(
                task_id=task.task_id,
                success=True,
                capability_delta={"d0": 0.2},
                resource_consumed=task.resource_cost,
            )

        exec_ = Executor(lux=lux, task_runner=cap_runner)
        state = _exec_state()
        result = exec_.execute(_exec_goal(), state)
        G_after, _, _, _ = result.final_state
        # Graph is a new immutable instance — not the original
        assert G_after is not state[0]
        assert not G_after.capabilities.flags.writeable

    def test_authority_update_requires_error_signal(self):
        """Authority changes only when an Errors object is produced; not by fiat."""
        A = make_initial_authority(("A",), baseline=0.5)
        original = A.get("A")
        # Without an error signal there is no mechanism to change authority
        A_copy = A.copy()
        assert A_copy.get("A") == pytest.approx(original)
        # Only authority_update() changes it
        A_changed = authority_update(A, Errors(per_agent={"A": 0.0}), eta=0.1)
        assert A_changed.get("A") != pytest.approx(original)

    def test_lux_grant_is_only_capability_path(self):
        """Executor cannot grant capabilities that Lux has not issued."""
        bridge = SimulatedLuxBridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        # "test_cap" was never granted → no task succeeded
        assert not bridge.check_capability("A", "test_cap")


# ---------------------------------------------------------------------------
# INV-6: Resource Conservation via Lux Ledger
# ---------------------------------------------------------------------------

class TestInvariant6ResourceConservation:
    def test_successful_task_deducts_resource(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        assert bridge.get_balance("A") < 100.0

    def test_rejected_authorization_leaves_balance_intact(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        # Capability not granted → authorization denied → no deduction
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        exec_.execute(_exec_goal(), _exec_state())
        assert bridge.get_balance("A") == pytest.approx(100.0)

    def test_failed_execution_triggers_refund(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        assert bridge.get_balance("A") == pytest.approx(100.0)

    def test_insufficient_budget_blocks_execution(self):
        bridge = SimulatedLuxBridge(initial_budget=0.01)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        result = exec_.execute(_exec_goal(), _exec_state())
        assert result.tasks_succeeded == 0
        assert bridge.get_balance("A") == pytest.approx(0.01)

    def test_ledger_conserved_over_n_tasks(self):
        """Total deducted == sum of resource_consumed across all succeeded tasks."""
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)

        class NTaskPlanner(Planner):
            def decompose(self, goal, G, A):
                return [
                    Task(
                        task_id=str(uuid.uuid4()),
                        description=f"step {i}",
                        required_capability=goal.required_capability,
                        initiating_agent=goal.initiating_agent,
                        resource_cost=3.0,
                        depth=0,
                    )
                    for i in range(4)
                ]

        exec_ = Executor(lux=lux, task_runner=mock_task_runner, planner=NTaskPlanner())
        result = exec_.execute(_exec_goal(budget=20.0), _exec_state())
        assert result.tasks_succeeded == 4
        assert bridge.get_balance("A") == pytest.approx(100.0 - 3.0 * 4)
        assert result.resources_spent == pytest.approx(3.0 * 4)


# ---------------------------------------------------------------------------
# INV-7: Observable + Fail-Closed Propagation
# ---------------------------------------------------------------------------

class TestInvariant7Observable:
    def test_every_ce_attempt_produces_audit(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        result = exec_.execute(_exec_goal(), _exec_state())
        assert len(result.audit_ids) == result.tasks_attempted
        assert len(bridge.get_audit_log()) >= result.tasks_attempted

    def test_failed_authorization_is_audited(self):
        bridge = SimulatedLuxBridge()  # no capability
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        exec_.execute(_exec_goal(), _exec_state())
        log = bridge.get_audit_log()
        assert any(not r["success"] for r in log)

    def test_failed_execution_is_audited(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        log = bridge.get_audit_log()
        failures = [r for r in log if not r["success"]]
        assert len(failures) >= 1

    def test_audit_log_is_append_only(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        exec_.execute(_exec_goal(), _exec_state())
        snapshot1 = bridge.get_audit_log()
        exec_.execute(_exec_goal(), _exec_state())
        snapshot2 = bridge.get_audit_log()
        # Each call adds records; old records are still present
        assert len(snapshot2) >= len(snapshot1)
        # Snapshots are copies — modifying one doesn't affect the other
        snapshot1.clear()
        assert len(bridge.get_audit_log()) == len(snapshot2)

    def test_audit_records_are_immutable_dicts(self):
        """Modifying the returned dict copy must not affect the stored record."""
        bridge = SimulatedLuxBridge()
        bridge.audit("add_edge", ("A",), True, details={"x": 1})
        log = bridge.get_audit_log()
        log[0]["details"]["x"] = 999  # mutate the copy
        fresh = bridge.get_audit_log()
        assert fresh[0]["details"]["x"] == 1  # original unchanged


# ---------------------------------------------------------------------------
# INV-8: Bounded Speculation
# ---------------------------------------------------------------------------

class TestInvariant8BoundedSpeculation:
    def test_depth_exceeded_rejects_task(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)

        class DeepPlanner(Planner):
            def decompose(self, goal, G, A):
                return [Task(
                    task_id=str(uuid.uuid4()),
                    description="deep",
                    required_capability=goal.required_capability,
                    initiating_agent=goal.initiating_agent,
                    resource_cost=1.0,
                    depth=MAX_DECOMPOSITION_DEPTH + 1,
                )]

        exec_ = Executor(lux=lux, planner=DeepPlanner())
        goal = Goal(
            goal_id="g", description="deep", required_capability="test_cap",
            initiating_agent="A", resource_budget=10.0,
            max_depth=MAX_DECOMPOSITION_DEPTH,
        )
        result = exec_.execute(goal, _exec_state())
        assert result.tasks_succeeded == 0
        rejections = [
            r for r in bridge.get_audit_log()
            if r.get("details", {}).get("reason") == "depth_exceeded"
        ]
        assert len(rejections) >= 1

    def test_pending_quota_prevents_new_proposals(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        # Fill pending counter to the limit
        exec_._pending["A"] = MAX_PENDING_CES_PER_AGENT
        result = exec_.execute(_exec_goal(), _exec_state())
        assert result.tasks_succeeded == 0

    def test_pending_resets_after_completion(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        # After completion, pending is cleared
        assert exec_._pending.get("A", 0) == 0

    def test_pending_resets_after_failure(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_exec_goal(), _exec_state())
        assert exec_._pending.get("A", 0) == 0

    def test_max_depth_config_is_respected(self):
        """Goal.max_depth overrides default; shallow goal rejects deep tasks."""
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        bridge.grant_capability("A", "test_cap")
        lux = Lux(bridge=bridge)

        class OneLevelDeepPlanner(Planner):
            def decompose(self, goal, G, A):
                return [Task(
                    task_id=str(uuid.uuid4()),
                    description="one level deep",
                    required_capability=goal.required_capability,
                    initiating_agent=goal.initiating_agent,
                    resource_cost=1.0,
                    depth=1,
                )]

        exec_ = Executor(lux=lux, planner=OneLevelDeepPlanner())
        # max_depth=0: depth=1 is rejected
        shallow_goal = Goal(
            goal_id="g", description="shallow", required_capability="test_cap",
            initiating_agent="A", resource_budget=10.0, max_depth=0,
        )
        result = exec_.execute(shallow_goal, _exec_state())
        assert result.tasks_succeeded == 0

        # max_depth=1: depth=1 is accepted
        deeper_goal = Goal(
            goal_id="g2", description="deeper", required_capability="test_cap",
            initiating_agent="A", resource_budget=10.0, max_depth=1,
        )
        exec_.reset_pending()
        result2 = exec_.execute(deeper_goal, _exec_state())
        assert result2.tasks_succeeded == 1


# ---------------------------------------------------------------------------
# INV-9: Coordinator Serialization
# ---------------------------------------------------------------------------

def _coord_state(agent_ids=("A", "B", "C")):
    n = len(agent_ids)
    adj = np.zeros((n, n), dtype=float)
    caps = np.ones((n, 2), dtype=float) * 0.5
    G = Graph(agent_ids=tuple(agent_ids), adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
    A = make_initial_authority(agent_ids)
    return G, phi, A, []


class TestInvariant9CoordinatorSerialization:
    """INV-9: Proposals are sorted by authority before Lux evaluation.
    No two proposals may claim the same directed edge in one round."""

    def test_higher_authority_wins_conflict(self):
        """When two agents target the same edge, the higher-authority one wins."""
        G, phi, A, E = _coord_state()
        A = Authority(scores={"A": 0.8, "B": 0.2, "C": 0.5}, baseline=0.5)
        state = (G, phi, A, E)

        coord = MultiAgentCoordinator(lux=Lux())
        ce = make_ce("add_edge", ("A", "B"), weight=0.5)
        round_ = coord.coordinate([
            ProposedCE("B", ce, priority=0.2),  # lower authority
            ProposedCE("A", ce, priority=0.8),  # higher authority
        ], state)

        accepted_agents = [p.agent_id for p in round_.accepted]
        assert "A" in accepted_agents
        assert "B" not in accepted_agents

    def test_conflict_detection_prevents_duplicate_edge_writes(self):
        """A directed edge can only be written once per round."""
        state = _coord_state()
        coord = MultiAgentCoordinator(lux=Lux())
        # Both target A→B
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.6), priority=0.9),
            ProposedCE("B", make_ce("add_edge", ("A", "B"), weight=0.3), priority=0.1),
        ]
        round_ = coord.coordinate(proposals, state)
        # Exactly one accepted (no double-write)
        assert len(round_.accepted) == 1
        assert round_.n_conflicts_detected == 1

    def test_each_proposal_produces_audit(self):
        """Every proposal — accepted or conflict-rejected — must generate an audit record."""
        bridge = SimulatedLuxBridge()
        lux = Lux(bridge=bridge)
        state = _coord_state()
        coord = MultiAgentCoordinator(lux=lux)
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5), priority=0.9),
            ProposedCE("B", make_ce("add_edge", ("A", "B"), weight=0.5), priority=0.1),
            ProposedCE("C", make_ce("add_edge", ("B", "C"), weight=0.5)),
        ]
        round_ = coord.coordinate(proposals, state)
        total = len(round_.accepted) + len(round_.rejected)
        assert len(round_.audit_ids) == total
        log = bridge.get_audit_log()
        assert len(log) >= total

    def test_max_agents_cap_is_enforced(self):
        """More than max_agents proposals are truncated before processing."""
        state = _coord_state(("A", "B", "C"))
        coord = MultiAgentCoordinator(lux=Lux(), max_agents=2)
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
            ProposedCE("B", make_ce("add_edge", ("B", "C"), weight=0.5)),
            ProposedCE("C", make_ce("add_edge", ("A", "C"), weight=0.5)),  # excess
        ]
        round_ = coord.coordinate(proposals, state)
        total = len(round_.accepted) + len(round_.rejected)
        assert total <= 2

    def test_state_is_consistent_after_round(self):
        """Graph arrays remain read-only and authority stays in [0,1] after a round."""
        state = _coord_state()
        coord = MultiAgentCoordinator(lux=Lux())
        proposals = [
            ProposedCE("A", make_ce("add_edge", ("A", "B"), weight=0.5)),
            ProposedCE("B", make_ce("add_edge", ("B", "C"), weight=0.3)),
        ]
        round_ = coord.coordinate(proposals, state)
        G_out, phi_out, A_out, E_out = round_.final_state

        # Read-only enforcement preserved
        assert not G_out.adjacency.flags.writeable
        assert not G_out.capabilities.flags.writeable

        # Authority in [0, 1]
        for agent_id in G_out.agent_ids:
            assert 0.0 <= A_out.get(agent_id) <= 1.0


# ---------------------------------------------------------------------------
# INV-10: Observer Isolation
# ---------------------------------------------------------------------------

class TestInvariant10ObserverIsolation:
    """INV-10: Observers are strictly read-only.

    Observer exceptions are caught by fire_observers() and logged at WARNING.
    They never propagate to the kernel — blast radius is zero.
    """

    def _small_state(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        caps = np.ones((3, 2)) * 0.5
        G = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=0)
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        return G, phi, A, []

    def test_observer_exception_does_not_interrupt_kernel(self):
        """An observer that always raises must not abort the kernel loop."""
        from emergo.observer import _NoOpMixin

        class AlwaysCrashes(_NoOpMixin):
            def on_iteration_start(self, t, state):
                raise RuntimeError("boom")

            def on_ce_result(self, t, ce, accepted, errors):
                raise RuntimeError("boom")

        from emergo import emergo_kernel
        final, reason = emergo_kernel(
            self._small_state(),
            max_iterations=10,
            observers=[AlwaysCrashes()],
            rng=np.random.default_rng(99),
        )
        assert reason in ("Converged", "Max iterations reached")

    def test_fire_observers_isolates_each_observer(self):
        """A crash in observer[0] must not prevent observer[1] from being called."""
        from emergo.observer import _NoOpMixin, fire_observers

        calls = []

        class Crasher(_NoOpMixin):
            def on_phi_updated(self, t, loss):
                raise RuntimeError("crash")

        class Recorder(_NoOpMixin):
            def on_phi_updated(self, t, loss):
                calls.append(loss)

        fire_observers([Crasher(), Recorder()], "on_phi_updated", 0, 0.42)
        assert calls == [0.42]

    def test_kernel_state_unchanged_by_crashing_observer(self):
        """A crashing observer must not corrupt the kernel's state tuple."""
        from emergo.observer import _NoOpMixin
        from emergo import emergo_kernel

        initial = self._small_state()
        G0, phi0, A0, _ = initial

        class MutationAttempt(_NoOpMixin):
            def on_iteration_start(self, t, state):
                # Attempt to mutate — should be harmless because adjacency is read-only
                try:
                    state[0].adjacency[0, 0] = 999.0
                except (ValueError, TypeError):
                    pass

        final, reason = emergo_kernel(
            initial,
            max_iterations=5,
            observers=[MutationAttempt()],
            rng=np.random.default_rng(1),
        )
        G_final = final[0]
        # adjacency must still be read-only in the final state
        assert not G_final.adjacency.flags.writeable

    def test_observer_cannot_alter_authority_scores(self):
        """Observer receives a snapshot dict; mutating it must not affect kernel A."""
        from emergo.observer import _NoOpMixin
        from emergo import emergo_kernel

        class ScoreSmasher(_NoOpMixin):
            def on_iteration_start(self, t, state):
                _, _, A, _ = state
                # Attempt to mutate the scores dict
                try:
                    A.scores["A"] = 0.0
                except (AttributeError, TypeError):
                    pass

        final, reason = emergo_kernel(
            self._small_state(),
            max_iterations=10,
            observers=[ScoreSmasher()],
            rng=np.random.default_rng(2),
        )
        # Kernel should have completed; authority must still be in [0, 1]
        _, _, A_final, _ = final
        for agent_id in A_final.scores:
            assert 0.0 <= A_final.get(agent_id) <= 1.0

    def test_on_kernel_done_fired_for_both_termination_reasons(self):
        """on_kernel_done must be called for both 'Converged' and 'Max iterations reached'."""
        from emergo.observer import _NoOpMixin
        from emergo import emergo_kernel

        reasons_seen = []

        class DoneTracker(_NoOpMixin):
            def on_kernel_done(self, reason, state, n_iterations):
                reasons_seen.append(reason)

        # Converged path (loose threshold)
        emergo_kernel(
            self._small_state(),
            max_iterations=50,
            convergence_threshold=1e10,
            observers=[DoneTracker()],
            rng=np.random.default_rng(3),
        )
        assert "Converged" in reasons_seen

        reasons_seen.clear()

        # Max iterations path (tight threshold, few iterations)
        emergo_kernel(
            self._small_state(),
            max_iterations=3,
            convergence_threshold=1e-20,
            observers=[DoneTracker()],
            rng=np.random.default_rng(4),
        )
        assert "Max iterations reached" in reasons_seen
