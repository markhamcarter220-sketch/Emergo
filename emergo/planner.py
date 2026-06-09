"""Planner — decomposes Goals into ordered Task sequences.

The base Planner is flat (one task per goal) and intentionally simple.
It is the right extension point for learned decomposition strategies:
as PhiUpdate accumulates history, a future planner can use φ-embeddings
to choose decompositions that are likely to succeed given current topology.

INV-8: all tasks emitted by decompose() start at depth=0.
       The Executor enforces the depth ceiling from Goal.max_depth.

Three planners are provided:
  Planner           — single flat task (base case)
  SequentialPlanner — ordered list of flat steps (flat, no dependencies)
  DependencyPlanner — explicit DAG of tasks; emits in topological order
"""

from __future__ import annotations

import uuid

from emergo.config import DEFAULT_TASK_COST
from emergo.types import Authority, Goal, Graph, Task

# ---------------------------------------------------------------------------
# Base planner
# ---------------------------------------------------------------------------


class Planner:
    """Decomposes a Goal into an ordered list of Tasks.

    Subclass or compose to implement multi-step or recursive decomposition.
    The only invariant this class enforces: all emitted Tasks have depth=0
    (depth increments happen in the Executor during recursive decomposition).
    """

    def decompose(self, goal: Goal, G: Graph, A: Authority) -> list[Task]:
        """Return tasks to execute for goal.  Base: single flat task."""
        return [
            Task(
                task_id=str(uuid.uuid4()),
                description=goal.description,
                required_capability=goal.required_capability,
                initiating_agent=goal.initiating_agent,
                resource_cost=min(goal.resource_budget, DEFAULT_TASK_COST),
                resource_type="compute",
                depth=0,
            )
        ]


# ---------------------------------------------------------------------------
# Sequential planner
# ---------------------------------------------------------------------------


class SequentialPlanner(Planner):
    """Emits one task per step_description in an ordered list.

    Useful for multi-step goals where steps are known at planning time.
    Each task requires the same capability and shares the same initiating agent.
    """

    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    def decompose(self, goal: Goal, G: Graph, A: Authority) -> list[Task]:
        per_step_cost = goal.resource_budget / max(len(self._steps), 1)
        return [
            Task(
                task_id=str(uuid.uuid4()),
                description=step,
                required_capability=goal.required_capability,
                initiating_agent=goal.initiating_agent,
                resource_cost=min(per_step_cost, DEFAULT_TASK_COST),
                resource_type="compute",
                depth=0,
            )
            for step in self._steps
        ]


# ---------------------------------------------------------------------------
# Dependency planner
# ---------------------------------------------------------------------------


class DependencyPlanner(Planner):
    """Decomposes a goal into explicitly dependency-ordered tasks.

    Accepts a DAG specification as a list of (name, description, [dep_names]).
    Tasks are emitted in topological order so the Executor sees them in the
    correct sequence.  Raises ValueError if the DAG has cycles or unknown deps.

    INV-8: all emitted tasks have depth=0.

    Example::

        planner = DependencyPlanner(steps=[
            ("fetch",     "Retrieve source documents",  []),
            ("extract",   "Extract key facts",          ["fetch"]),
            ("summarize", "Write summary draft",        ["extract"]),
            ("validate",  "Validate the draft",         ["summarize", "extract"]),
        ])
        executor = Executor(lux=lux, planner=planner)
        result = executor.execute(goal, state)
    """

    def __init__(self, steps: list[tuple[str, str, list[str]]]) -> None:
        """
        Args:
            steps: List of (name, description, [dep_names]).
                   Names must be unique; dep_names must all appear in names.
        """
        _validate_dag(steps)
        self._steps = steps

    def decompose(self, goal: Goal, G: Graph, A: Authority) -> list[Task]:
        per_step_cost = goal.resource_budget / max(len(self._steps), 1)
        ordered_names = _topological_sort(self._steps)

        desc_map = {name: desc for name, desc, _ in self._steps}
        return [
            Task(
                task_id=str(uuid.uuid4()),
                description=desc_map[name],
                required_capability=goal.required_capability,
                initiating_agent=goal.initiating_agent,
                resource_cost=min(per_step_cost, DEFAULT_TASK_COST),
                resource_type="compute",
                depth=0,
            )
            for name in ordered_names
        ]


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------


def _validate_dag(steps: list[tuple[str, str, list[str]]]) -> None:
    """Raise ValueError if steps has duplicate names, unknown deps, or cycles."""
    names: set[str] = set()
    for name, _desc, _deps in steps:
        if name in names:
            raise ValueError(f"Duplicate task name: {name!r}")
        names.add(name)

    for name, _desc, deps in steps:
        for dep in deps:
            if dep not in names:
                raise ValueError(f"Task {name!r} depends on unknown task {dep!r}")

    if _has_cycle(steps):
        raise ValueError("DependencyPlanner: task dependency graph contains a cycle")


def _has_cycle(steps: list[tuple[str, str, list[str]]]) -> bool:
    """Return True if the dependency graph contains a directed cycle."""
    graph: dict[str, list[str]] = {name: list(deps) for name, _, deps in steps}
    # Standard DFS cycle detection: white/gray/black coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in graph}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbour in graph[node]:
            if color[neighbour] == GRAY:
                return True
            if color[neighbour] == WHITE and dfs(neighbour):
                return True
        color[node] = BLACK
        return False

    return any(color[name] == WHITE and dfs(name) for name in graph)


def _topological_sort(steps: list[tuple[str, str, list[str]]]) -> list[str]:
    """Return names in topological order (Kahn's algorithm)."""
    in_degree: dict[str, int] = {name: 0 for name, _, _ in steps}
    children: dict[str, list[str]] = {name: [] for name, _, _ in steps}

    for name, _, deps in steps:
        for dep in deps:
            children[dep].append(name)
            in_degree[name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order
