"""End-to-end integration tests for Emergo.

Tests the full pipeline: kernel + coordinator + executor + observer + planner.
These are longer-running tests that exercise multiple components together.
"""
from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    CoordinationRound,
    DependencyPlanner,
    Executor,
    Goal,
    Graph,
    HistoryObserver,
    LoggingObserver,
    Lux,
    MultiAgentCoordinator,
    ProposedCE,
    SequentialPlanner,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
    run_health_check,
)
from emergo.lux_bridge import SimulatedLuxBridge, validate_bridge
from emergo.types import Authority, CoordinationEvent
from tests.conftest import make_ce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangle() -> Graph:
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    caps = np.ones((3, 2)) * 0.5
    return Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)


def _empty_graph(n: int = 3) -> Graph:
    ids = tuple(f"agent{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    caps = np.ones((n, 2), dtype=float) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _initial_state(G=None):
    if G is None:
        G = _triangle()
    phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=99)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    return G, phi, A, []


# ---------------------------------------------------------------------------
# Kernel + Observer end-to-end
# ---------------------------------------------------------------------------

class TestKernelWithObservers:
    def test_kernel_runs_with_history_observer(self):
        obs = HistoryObserver()
        state = _initial_state()
        final, reason = emergo_kernel(
            state,
            max_iterations=30,
            observers=[obs],
            rng=np.random.default_rng(10),
        )
        assert reason in ("Converged", "Max iterations reached")

    def test_multiple_observers_all_receive_events(self):
        obs1 = HistoryObserver()
        obs2 = HistoryObserver()
        state = _initial_state()
        emergo_kernel(
            state,
            max_iterations=20,
            observers=[obs1, obs2],
            rng=np.random.default_rng(11),
        )
        # Both accumulate the same number of events
        assert len(obs1.mean_errors) == len(obs2.mean_errors)

    def test_kernel_diagnostics_with_observer(self):
        obs = HistoryObserver()
        state = _initial_state()
        result = emergo_kernel(
            state,
            max_iterations=30,
            collect_diagnostics=True,
            observers=[obs],
            rng=np.random.default_rng(12),
        )
        assert len(result) == 3
        final_state, reason, diag = result
        assert diag is not None

    def test_logging_observer_does_not_raise(self):
        obs = LoggingObserver(log_every=5)
        state = _initial_state()
        emergo_kernel(
            state,
            max_iterations=15,
            observers=[obs],
            rng=np.random.default_rng(13),
        )


# ---------------------------------------------------------------------------
# Kernel + Diagnostics health check
# ---------------------------------------------------------------------------

class TestKernelDiagnosticsIntegration:
    def test_health_check_runs_after_kernel(self):
        state = _initial_state()
        final_state, reason, diag = emergo_kernel(
            state,
            max_iterations=50,
            collect_diagnostics=True,
            rng=np.random.default_rng(20),
        )
        results = run_health_check(diag, final_state)
        assert len(results) == 8
        for r in results:
            assert 0.0 <= r.severity <= 1.0
            assert isinstance(r.failure_detected, bool)

    def test_diag_records_present(self):
        state = _initial_state()
        _, _, diag = emergo_kernel(
            state,
            max_iterations=20,
            collect_diagnostics=True,
            rng=np.random.default_rng(21),
        )
        assert isinstance(diag.records, list)


# ---------------------------------------------------------------------------
# Executor + DependencyPlanner end-to-end
# ---------------------------------------------------------------------------

class TestExecutorWithDependencyPlanner:
    def test_three_step_dependency_plan_executes(self):
        bridge = SimulatedLuxBridge(initial_budget=200.0)
        lux = Lux(bridge=bridge)
        lux.grant_capability("A", "summarize")

        G = _triangle()
        state = _initial_state(G)

        planner = DependencyPlanner([
            ("fetch",   "Retrieve docs", []),
            ("extract", "Extract facts", ["fetch"]),
            ("write",   "Write summary", ["extract"]),
        ])
        executor = Executor(lux=lux, planner=planner)

        goal = Goal(
            goal_id="g1",
            description="Research task",
            required_capability="summarize",
            initiating_agent="A",
            resource_budget=9.0,
        )
        result = executor.execute(goal, state)
        assert result.tasks_attempted == 3
        assert result.tasks_succeeded == 3

    def test_sequential_planner_with_executor(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)
        lux.grant_capability("A", "analyze")

        G = _triangle()
        state = _initial_state(G)

        planner = SequentialPlanner(["Step 1", "Step 2"])
        executor = Executor(lux=lux, planner=planner)

        goal = Goal(
            goal_id="g2",
            description="Two-step task",
            required_capability="analyze",
            initiating_agent="A",
            resource_budget=4.0,
        )
        result = executor.execute(goal, state)
        assert result.tasks_attempted == 2


# ---------------------------------------------------------------------------
# Coordinator + Kernel state continuity
# ---------------------------------------------------------------------------

class TestCoordinatorIntegration:
    def test_coordinator_round_then_kernel_continues(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)

        G = _empty_graph(3)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=30)
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        state = (G, phi, A, [])

        coord = MultiAgentCoordinator(lux=lux)
        round_ = coord.coordinate([
            ProposedCE("agent0", make_ce("add_edge", ("agent0", "agent1"), weight=0.5)),
            ProposedCE("agent1", make_ce("add_edge", ("agent1", "agent2"), weight=0.5)),
        ], state)

        assert isinstance(round_, CoordinationRound)
        # Continue kernel from the coordinator's final state
        final_state, reason = emergo_kernel(
            round_.final_state,
            max_iterations=10,
            rng=np.random.default_rng(31),
        )
        assert reason in ("Converged", "Max iterations reached")

    def test_coordinator_authority_ordering_persists_to_kernel(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        lux = Lux(bridge=bridge)

        G = _empty_graph(2)
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=32)
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        state = (G, phi, A, [])

        coord = MultiAgentCoordinator(lux=lux)
        round_ = coord.coordinate([
            ProposedCE("agent0", make_ce("add_edge", ("agent0", "agent1"), weight=0.6),
                       priority=0.9),
            ProposedCE("agent1", make_ce("add_edge", ("agent0", "agent1"), weight=0.4),
                       priority=0.3),
        ], state)

        # Only one can win (same edge conflict)
        assert len(round_.accepted) == 1
        assert round_.accepted[0].agent_id == "agent0"


# ---------------------------------------------------------------------------
# validate_bridge smoke test
# ---------------------------------------------------------------------------

class TestValidateBridge:
    def test_simulated_bridge_passes_validation(self):
        bridge = SimulatedLuxBridge(initial_budget=100.0)
        assert validate_bridge(bridge) is True

    def test_validate_bridge_returns_true(self):
        bridge = SimulatedLuxBridge()
        result = validate_bridge(bridge)
        assert result is True
