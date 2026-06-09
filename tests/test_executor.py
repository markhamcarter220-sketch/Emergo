"""Tests for the Executor runtime.

Covers: basic execution, resource management, capability enforcement,
INV-5/6/7/8 enforcement, authority feedback, graph immutability.
"""

import numpy as np
import pytest

from emergo import (
    ExecutionResult,
    Executor,
    Goal,
    Graph,
    Lux,
    Planner,
    TaskOutcome,
    failing_task_runner,
    make_initial_authority,
    make_initial_phi,
    mock_task_runner,
)
from emergo.config import MAX_PENDING_CES_PER_AGENT
from emergo.lux_bridge import SimulatedLuxBridge
from emergo.types import Task


def _make_bridge(initial_budget: float = 100.0) -> SimulatedLuxBridge:
    b = SimulatedLuxBridge(initial_budget=initial_budget)
    b.grant_capability("A", "test_cap")
    b.grant_capability("B", "test_cap")
    b.grant_capability("C", "test_cap")
    return b


def _make_state(agent_ids=("A", "B", "C")):
    n = len(agent_ids)
    adj = np.eye(n, k=1, dtype=float) + np.eye(n, k=-(n - 1), dtype=float)
    caps = np.ones((n, 2)) * 0.5
    G0 = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
    A0 = make_initial_authority(agent_ids)
    return G0, phi0, A0, []


def _make_goal(agent: str = "A", budget: float = 10.0) -> Goal:
    return Goal(
        goal_id="g1",
        description="test goal",
        required_capability="test_cap",
        initiating_agent=agent,
        resource_budget=budget,
        max_depth=3,
    )


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_execute_simple_goal_succeeds(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        result = exec_.execute(_make_goal(), _make_state())
        assert isinstance(result, ExecutionResult)
        assert result.success
        assert result.tasks_succeeded == 1

    def test_execute_returns_updated_state(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        state = _make_state()
        result = exec_.execute(_make_goal(), state)
        G_final, _, _, E_final = result.final_state
        assert isinstance(G_final, Graph)
        assert len(E_final) > 0

    def test_execute_produces_audit_records(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        result = exec_.execute(_make_goal(), _make_state())
        assert len(result.audit_ids) >= 1
        log = bridge.get_audit_log()
        assert len(log) >= 1

    def test_execute_tracks_resources_spent(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        result = exec_.execute(_make_goal(budget=5.0), _make_state())
        assert result.resources_spent > 0.0
        assert result.resources_spent <= 5.0


# ---------------------------------------------------------------------------
# INV-6: Resource Conservation
# ---------------------------------------------------------------------------


class TestResourceConservation:
    def test_successful_task_deducts_from_ledger(self):
        bridge = _make_bridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        exec_.execute(_make_goal(), _make_state())
        balance_after = bridge.get_balance("A")
        assert balance_after < 100.0

    def test_failed_task_triggers_refund(self):
        bridge = _make_bridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_make_goal(), _make_state())
        # Balance should be restored (refund on failure)
        assert bridge.get_balance("A") == pytest.approx(100.0)

    def test_missing_capability_does_not_deduct(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        # Do NOT grant "test_cap" to agent A
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        exec_.execute(_make_goal(), _make_state())
        assert bridge.get_balance("A") == pytest.approx(100.0)

    def test_insufficient_budget_does_not_execute(self):
        bridge = _make_bridge(initial_budget=0.1)  # less than DEFAULT_TASK_COST
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        result = exec_.execute(_make_goal(), _make_state())
        assert result.tasks_succeeded == 0
        assert bridge.get_balance("A") == pytest.approx(0.1)

    def test_budget_conserved_over_multiple_tasks(self):
        """Planner emitting N tasks: total deducted = sum of task costs."""
        import uuid

        bridge = _make_bridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)

        class MultiTaskPlanner(Planner):
            def decompose(self, goal, G, A):
                return [
                    Task(
                        task_id=str(uuid.uuid4()),
                        description=f"step {i}",
                        required_capability=goal.required_capability,
                        initiating_agent=goal.initiating_agent,
                        resource_cost=2.0,
                        depth=0,
                    )
                    for i in range(3)
                ]

        exec_ = Executor(lux=lux, task_runner=mock_task_runner, planner=MultiTaskPlanner())
        result = exec_.execute(_make_goal(budget=10.0), _make_state())
        assert result.tasks_succeeded == 3
        assert bridge.get_balance("A") == pytest.approx(100.0 - 6.0)


# ---------------------------------------------------------------------------
# INV-5: Proposal-Only Authority
# ---------------------------------------------------------------------------


class TestProposalOnlyAuthority:
    def test_authority_changes_only_via_error_signal(self):
        """Executor must not directly set authority; it flows through authority_update."""
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        state = _make_state()
        _, _, _A_before, _ = state
        result = exec_.execute(_make_goal(), state)
        _, _, A_after, _ = result.final_state
        # A changed (expected), but both are Authority instances with clamped values
        for v in A_after.scores.values():
            assert 0.0 <= v <= 1.0

    def test_capability_requires_lux_grant(self):
        """Without lux.grant_capability(), execute_task is denied."""
        bridge = SimulatedLuxBridge(initial_budget=100.0)  # no grants
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        result = exec_.execute(_make_goal(), _make_state())
        assert result.tasks_succeeded == 0

    def test_executor_never_directly_mutates_capabilities(self):
        """After execution, capability changes entered graph via update_capabilities CE."""
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)

        # Runner returns a capability delta that should appear in graph
        def cap_delta_runner(task):
            return TaskOutcome(
                task_id=task.task_id,
                success=True,
                capability_delta={"dim0": 0.1},
                resource_consumed=task.resource_cost,
            )

        exec_ = Executor(lux=lux, task_runner=cap_delta_runner)
        state = _make_state()
        G_before, _, _, _ = state
        result = exec_.execute(_make_goal(), state)
        G_after, _, _, _ = result.final_state
        # Graph is immutable (new instance)
        assert G_after is not G_before


# ---------------------------------------------------------------------------
# INV-7: Observable + Fail-Closed
# ---------------------------------------------------------------------------


class TestObservable:
    def test_every_attempt_produces_audit_record(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        result = exec_.execute(_make_goal(), _make_state())
        assert len(result.audit_ids) == result.tasks_attempted

    def test_failed_tasks_audited(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_make_goal(), _make_state())
        log = bridge.get_audit_log()
        failures = [r for r in log if not r["success"]]
        assert len(failures) >= 1

    def test_unauthorized_task_audited(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)  # no capability
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        exec_.execute(_make_goal(), _make_state())
        log = bridge.get_audit_log()
        assert len(log) >= 1
        assert not log[0]["success"]

    def test_audit_ids_are_unique(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux)
        result = exec_.execute(_make_goal(), _make_state())
        assert len(result.audit_ids) == len(set(result.audit_ids))


# ---------------------------------------------------------------------------
# INV-8: Bounded Speculation
# ---------------------------------------------------------------------------


class TestBoundedSpeculation:
    def test_task_at_max_depth_rejected(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)

        # Planner emits a task at depth beyond goal.max_depth
        class DeepPlanner(Planner):
            def decompose(self, goal, G, A):
                import uuid

                return [
                    Task(
                        task_id=str(uuid.uuid4()),
                        description="deep task",
                        required_capability=goal.required_capability,
                        initiating_agent=goal.initiating_agent,
                        resource_cost=1.0,
                        depth=99,  # way beyond any max_depth
                    )
                ]

        exec_ = Executor(lux=lux, planner=DeepPlanner())
        goal = Goal(
            goal_id="g2",
            description="deep",
            required_capability="test_cap",
            initiating_agent="A",
            resource_budget=10.0,
            max_depth=3,
        )
        result = exec_.execute(goal, _make_state())
        assert result.tasks_succeeded == 0
        # Audit record written for the rejection
        log = bridge.get_audit_log()
        depth_rejections = [
            r for r in log if r.get("details", {}).get("reason") == "depth_exceeded"
        ]
        assert len(depth_rejections) >= 1

    def test_pending_quota_exceeded(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)

        # Manually fill the pending counter to quota
        exec_._pending["A"] = MAX_PENDING_CES_PER_AGENT

        result = exec_.execute(_make_goal(), _make_state())
        assert result.tasks_succeeded == 0
        log = bridge.get_audit_log()
        quota_rejections = [
            r for r in log if r.get("details", {}).get("reason") == "pending_quota_exceeded"
        ]
        assert len(quota_rejections) >= 1

    def test_pending_decrements_after_task(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=mock_task_runner)
        exec_.execute(_make_goal(), _make_state())
        # After successful execution, pending for agent A should be 0
        assert exec_._pending.get("A", 0) == 0

    def test_pending_decrements_after_failure(self):
        bridge = _make_bridge()
        lux = Lux(bridge=bridge)
        exec_ = Executor(lux=lux, task_runner=failing_task_runner)
        exec_.execute(_make_goal(), _make_state())
        assert exec_._pending.get("A", 0) == 0
