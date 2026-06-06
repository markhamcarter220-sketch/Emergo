"""Planner — decomposes Goals into ordered Task sequences.

The base Planner is flat (one task per goal) and intentionally simple.
It is the right extension point for learned decomposition strategies:
as PhiUpdate accumulates history, a future planner can use φ-embeddings
to choose decompositions that are likely to succeed given current topology.

INV-8: all tasks emitted by decompose() start at depth=0.
       The Executor enforces the depth ceiling from Goal.max_depth.
"""
from __future__ import annotations

import uuid
from typing import List

from emergo.config import DEFAULT_TASK_COST
from emergo.types import Authority, Goal, Graph, Task


class Planner:
    """Decomposes a Goal into an ordered list of Tasks.

    Subclass or compose to implement multi-step or recursive decomposition.
    The only invariant this class enforces: all emitted Tasks have depth=0
    (depth increments happen in the Executor during recursive decomposition).
    """

    def decompose(self, goal: Goal, G: Graph, A: Authority) -> List[Task]:
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


class SequentialPlanner(Planner):
    """Emits one task per step_description in an ordered list.

    Useful for multi-step goals where steps are known at planning time.
    Each task requires the same capability and shares the same initiating agent.
    """

    def __init__(self, steps: List[str]) -> None:
        self._steps = steps

    def decompose(self, goal: Goal, G: Graph, A: Authority) -> List[Task]:
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
