"""Tests for Planner, SequentialPlanner, and DependencyPlanner.

Verifies:
  - Base Planner emits a single task with depth=0 (INV-8)
  - SequentialPlanner emits ordered tasks, one per step
  - DependencyPlanner emits tasks in topological order
  - DependencyPlanner rejects cycles, unknown deps, and duplicate names
  - All planners emit tasks with depth=0 (INV-8 invariant)
  - Resource budget is correctly divided across tasks
"""
from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    DependencyPlanner,
    Graph,
    Planner,
    SequentialPlanner,
    make_initial_authority,
)
from emergo.types import Goal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph() -> Graph:
    adj = np.zeros((2, 2), dtype=float)
    caps = np.ones((2, 2), dtype=float) * 0.5
    return Graph(agent_ids=("A", "B"), adjacency=adj, capabilities=caps)


def _goal(budget: float = 10.0, capability: str = "summarize") -> Goal:
    return Goal(
        goal_id="g-test",
        description="Test goal",
        required_capability=capability,
        initiating_agent="A",
        resource_budget=budget,
    )


def _auth():
    return make_initial_authority(("A", "B"), baseline=0.5)


# ---------------------------------------------------------------------------
# Base Planner
# ---------------------------------------------------------------------------

class TestBasePlanner:
    def test_single_task_emitted(self):
        tasks = Planner().decompose(_goal(), _graph(), _auth())
        assert len(tasks) == 1

    def test_task_depth_is_zero(self):
        tasks = Planner().decompose(_goal(), _graph(), _auth())
        assert tasks[0].depth == 0

    def test_task_description_matches_goal(self):
        tasks = Planner().decompose(_goal(), _graph(), _auth())
        assert tasks[0].description == "Test goal"

    def test_task_capability_matches_goal(self):
        tasks = Planner().decompose(_goal(capability="analyze"), _graph(), _auth())
        assert tasks[0].required_capability == "analyze"

    def test_task_id_is_unique(self):
        p = Planner()
        G, A = _graph(), _auth()
        t1 = p.decompose(_goal(), G, A)
        t2 = p.decompose(_goal(), G, A)
        assert t1[0].task_id != t2[0].task_id


# ---------------------------------------------------------------------------
# SequentialPlanner
# ---------------------------------------------------------------------------

class TestSequentialPlanner:
    def test_emits_one_task_per_step(self):
        steps = ["Fetch", "Extract", "Summarize"]
        tasks = SequentialPlanner(steps).decompose(_goal(), _graph(), _auth())
        assert len(tasks) == 3

    def test_task_descriptions_match_steps_in_order(self):
        steps = ["Fetch", "Extract", "Summarize"]
        tasks = SequentialPlanner(steps).decompose(_goal(), _graph(), _auth())
        assert [t.description for t in tasks] == steps

    def test_all_tasks_have_depth_zero(self):
        tasks = SequentialPlanner(["A", "B"]).decompose(_goal(), _graph(), _auth())
        assert all(t.depth == 0 for t in tasks)

    def test_per_step_cost_respects_default_task_cost_cap(self):
        from emergo.config import DEFAULT_TASK_COST
        budget = 9.0
        steps = ["A", "B", "C"]
        tasks = SequentialPlanner(steps).decompose(
            _goal(budget=budget), _graph(), _auth()
        )
        per_step = budget / len(steps)
        expected_per = min(per_step, DEFAULT_TASK_COST)
        for t in tasks:
            assert t.resource_cost == pytest.approx(expected_per)

    def test_empty_steps_returns_no_tasks(self):
        tasks = SequentialPlanner([]).decompose(_goal(), _graph(), _auth())
        assert tasks == []


# ---------------------------------------------------------------------------
# DependencyPlanner
# ---------------------------------------------------------------------------

class TestDependencyPlanner:
    def _chain_steps(self):
        return [
            ("fetch",     "Retrieve source",  []),
            ("extract",   "Extract facts",    ["fetch"]),
            ("summarize", "Write summary",    ["extract"]),
        ]

    def test_emits_tasks_in_topological_order_chain(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(_goal(), _graph(), _auth())
        descs = [t.description for t in tasks]
        assert descs.index("Retrieve source") < descs.index("Extract facts")
        assert descs.index("Extract facts") < descs.index("Write summary")

    def test_diamond_dependency_resolved(self):
        steps = [
            ("A", "Step A", []),
            ("B", "Step B", ["A"]),
            ("C", "Step C", ["A"]),
            ("D", "Step D", ["B", "C"]),
        ]
        tasks = DependencyPlanner(steps).decompose(_goal(), _graph(), _auth())
        descs = [t.description for t in tasks]
        assert descs.index("Step A") < descs.index("Step B")
        assert descs.index("Step A") < descs.index("Step C")
        assert descs.index("Step B") < descs.index("Step D")
        assert descs.index("Step C") < descs.index("Step D")

    def test_all_tasks_have_depth_zero(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(_goal(), _graph(), _auth())
        assert all(t.depth == 0 for t in tasks)

    def test_task_count_matches_step_count(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(_goal(), _graph(), _auth())
        assert len(tasks) == 3

    def test_no_deps_all_independent(self):
        steps = [("x", "X", []), ("y", "Y", []), ("z", "Z", [])]
        tasks = DependencyPlanner(steps).decompose(_goal(), _graph(), _auth())
        assert len(tasks) == 3
        assert all(t.depth == 0 for t in tasks)

    def test_cycle_raises_value_error(self):
        steps = [
            ("A", "A", ["B"]),
            ("B", "B", ["A"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            DependencyPlanner(steps)

    def test_self_loop_raises_value_error(self):
        steps = [("A", "A", ["A"])]
        with pytest.raises(ValueError):
            DependencyPlanner(steps)

    def test_unknown_dep_raises_value_error(self):
        steps = [("A", "A", ["nonexistent"])]
        with pytest.raises(ValueError, match="unknown"):
            DependencyPlanner(steps)

    def test_duplicate_name_raises_value_error(self):
        steps = [("A", "First A", []), ("A", "Second A", [])]
        with pytest.raises(ValueError, match="Duplicate"):
            DependencyPlanner(steps)

    def test_per_step_cost_respects_default_task_cost_cap(self):
        from emergo.config import DEFAULT_TASK_COST
        steps = self._chain_steps()
        budget = 6.0
        tasks = DependencyPlanner(steps).decompose(
            _goal(budget=budget), _graph(), _auth()
        )
        per_step = budget / len(steps)
        expected_per = min(per_step, DEFAULT_TASK_COST)
        for t in tasks:
            assert t.resource_cost == pytest.approx(expected_per)

    def test_required_capability_inherited_from_goal(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(
            _goal(capability="analyze"), _graph(), _auth()
        )
        assert all(t.required_capability == "analyze" for t in tasks)

    def test_initiating_agent_inherited_from_goal(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(_goal(), _graph(), _auth())
        assert all(t.initiating_agent == "A" for t in tasks)

    def test_task_ids_are_unique(self):
        tasks = DependencyPlanner(self._chain_steps()).decompose(_goal(), _graph(), _auth())
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_three_way_cycle_raises(self):
        steps = [
            ("A", "A", ["C"]),
            ("B", "B", ["A"]),
            ("C", "C", ["B"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            DependencyPlanner(steps)
