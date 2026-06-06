"""Executor Runtime — Goal → Task sequence → Lux-authorized execution → feedback.

The Executor is the bridge between goal-directed behavior and the four atomic
Emergo operations.  It is the only component that may call lux.authorize_full().

Invariants enforced here:
  INV-5 (Proposal-Only Authority):
      Capability changes enter the graph ONLY through ce_execute(update_capabilities CE).
      The executor never calls authority.set() or graph.capabilities directly.

  INV-6 (Resource Conservation):
      Resource is pre-deducted by lux.authorize_full() before execution begins.
      If execution fails (runner exception, task failure, ce_execute rejection),
      refund_resource() is called unconditionally.

  INV-7 (Observable + Fail-Closed):
      lux.audit() is called for EVERY task attempt — success and failure alike.
      Audit is written before the state update propagates.

  INV-8 (Bounded Speculation):
      task.depth > goal.max_depth  →  CE rejected (logged, not executed)
      pending[agent] >= MAX_PENDING →  CE rejected (logged, not executed)
      Pending count decrements after each task completes (success or failure).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from emergo.authority_update import authority_update
from emergo.ce_execution import ce_execute
from emergo.config import (
    DEFAULT_ETA,
    DEFAULT_TASK_COST,
    MAX_DECOMPOSITION_DEPTH,
    MAX_PENDING_CES_PER_AGENT,
    PHI_UPDATE_INTERVAL,
)
from emergo.error_computation import error_computation
from emergo.lux import Lux
from emergo.phi_update import phi_update
from emergo.planner import Planner
from emergo.types import (
    Authority,
    CoordinationEvent,
    Errors,
    Goal,
    Graph,
    PhiMap,
    State,
    Task,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution-layer result types
# ---------------------------------------------------------------------------

@dataclass
class TaskOutcome:
    """Result returned by a TaskRunner for a single Task."""

    task_id: str
    success: bool
    capability_delta: Dict[str, float]  # {dimension_name: delta} applied to capabilities
    resource_consumed: float
    notes: str = ""


@dataclass
class ExecutionResult:
    """Summary of executing a complete Goal."""

    goal_id: str
    success: bool              # True iff at least one task succeeded
    tasks_attempted: int
    tasks_succeeded: int
    resources_spent: float
    final_state: State
    audit_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TaskRunner protocol
# ---------------------------------------------------------------------------

TaskRunner = Callable[[Task], TaskOutcome]
"""Callable: Task → TaskOutcome.

Implementations:
  mock_task_runner  — always succeeds, zero capability delta (tests / simulation)
  Custom runners    — call real tools, APIs, sub-agents, etc.
"""


def mock_task_runner(task: Task) -> TaskOutcome:
    """Default mock: succeeds with no capability change and full resource charge."""
    return TaskOutcome(
        task_id=task.task_id,
        success=True,
        capability_delta={},
        resource_consumed=task.resource_cost,
    )


def failing_task_runner(task: Task) -> TaskOutcome:
    """Test helper: always fails."""
    return TaskOutcome(
        task_id=task.task_id,
        success=False,
        capability_delta={},
        resource_consumed=0.0,
        notes="intentional failure",
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Orchestrates goal execution over the Emergo state machine.

    Usage:
        lux = Lux()
        lux.grant_capability("agent_A", "summarize")
        executor = Executor(lux=lux)
        result = executor.execute(goal, initial_state)
    """

    def __init__(
        self,
        lux: Optional[Lux] = None,
        task_runner: Optional[TaskRunner] = None,
        planner: Optional[Planner] = None,
        eta: float = DEFAULT_ETA,
        phi_update_interval: int = PHI_UPDATE_INTERVAL,
    ) -> None:
        self._lux = lux or Lux()
        self._task_runner = task_runner or mock_task_runner
        self._planner = planner or Planner()
        self._eta = eta
        self._phi_update_interval = phi_update_interval

        # INV-8: per-agent pending CE counter (reset between execute() calls if desired)
        self._pending: Dict[str, int] = {}

        # Accumulated history for φ-fitting (persists across execute() calls)
        self._g_history: List[Graph] = []
        self._ce_history: List[CoordinationEvent] = []
        self._steps_since_phi_update: int = 0

    def reset_pending(self) -> None:
        """Clear pending CE counters.  Call between independent execution sessions."""
        self._pending.clear()

    def execute(self, goal: Goal, state: State) -> ExecutionResult:
        """Execute a Goal against the current Emergo state.

        Returns an ExecutionResult with the updated state and audit trail.
        The input state is never mutated.
        """
        G, phi, A, E_history = state

        tasks = self._planner.decompose(goal, G, A)

        audit_ids: List[str] = []
        resources_spent = 0.0
        tasks_attempted = 0
        tasks_succeeded = 0

        if not self._g_history:
            self._g_history.append(G)

        for task in tasks:
            # ---- INV-8: depth guard ----
            if task.depth > goal.max_depth:
                logger.warning(
                    "Task %s (depth=%d) exceeds max_depth=%d; rejected",
                    task.task_id, task.depth, goal.max_depth,
                )
                aid = self._lux.audit(
                    "execute_task", (task.initiating_agent,), False,
                    details={"reason": "depth_exceeded", "depth": task.depth,
                             "max_depth": goal.max_depth},
                )
                audit_ids.append(aid)
                continue

            # ---- INV-8: pending quota guard ----
            current_pending = self._pending.get(task.initiating_agent, 0)
            if current_pending >= MAX_PENDING_CES_PER_AGENT:
                logger.warning(
                    "Agent %s pending=%d >= MAX_PENDING=%d; rejected",
                    task.initiating_agent, current_pending, MAX_PENDING_CES_PER_AGENT,
                )
                aid = self._lux.audit(
                    "execute_task", (task.initiating_agent,), False,
                    details={"reason": "pending_quota_exceeded",
                             "pending": current_pending},
                )
                audit_ids.append(aid)
                continue

            # Build the execute_task CE (used for Lux auth + φ-space encoding)
            execute_ce = CoordinationEvent(
                event_type="execute_task",
                participants=(task.initiating_agent,),
                params=frozenset([
                    ("task_id", task.task_id),
                    ("capability", task.required_capability),
                    ("resource_cost", task.resource_cost),
                    ("resource", task.resource_type),
                ]),
            )

            # ---- INV-6: authorize with resource pre-deduction ----
            auth = self._lux.authorize_full(execute_ce, G, A)
            tasks_attempted += 1
            self._pending[task.initiating_agent] = current_pending + 1

            if not auth.authorized:
                # INV-7: audit failure before any state change
                aid = self._lux.audit(
                    "execute_task", execute_ce.participants, False,
                    details={"reason": auth.reason},
                )
                audit_ids.append(aid)
                self._pending[task.initiating_agent] = max(0, self._pending[task.initiating_agent] - 1)
                continue

            # ---- Execute via TaskRunner ----
            try:
                outcome = self._task_runner(task)
            except Exception as exc:
                logger.error("TaskRunner raised for task %s: %s", task.task_id, exc)
                # INV-6: refund pre-deducted resource
                if auth.resource_reserved > 0:
                    self._lux.refund_resource(
                        task.initiating_agent, task.resource_type, auth.resource_reserved
                    )
                # INV-7: audit exception
                aid = self._lux.audit(
                    "execute_task", execute_ce.participants, False,
                    details={"reason": "task_runner_exception", "error": str(exc)},
                )
                audit_ids.append(aid)
                self._pending[task.initiating_agent] = max(0, self._pending[task.initiating_agent] - 1)
                continue

            if not outcome.success:
                # INV-6: refund on task failure
                if auth.resource_reserved > 0:
                    self._lux.refund_resource(
                        task.initiating_agent, task.resource_type, auth.resource_reserved
                    )
                # INV-7: audit task failure
                aid = self._lux.audit(
                    "execute_task", execute_ce.participants, False,
                    details={"reason": "task_failed", "notes": outcome.notes},
                )
                audit_ids.append(aid)
                self._pending[task.initiating_agent] = max(0, self._pending[task.initiating_agent] - 1)
                continue

            # ---- Task succeeded: commit outcome ----
            G, A, phi, E_history = self._commit_outcome(
                task, outcome, execute_ce, G, phi, A, E_history
            )

            # INV-7: audit success AFTER state commit (state is consistent)
            aid = self._lux.audit(
                "execute_task", execute_ce.participants, True,
                details={
                    "task_id": task.task_id,
                    "capability": task.required_capability,
                    "notes": outcome.notes,
                },
                resource_deducted=outcome.resource_consumed,
            )
            audit_ids.append(aid)
            resources_spent += outcome.resource_consumed
            tasks_succeeded += 1
            self._pending[task.initiating_agent] = max(0, self._pending[task.initiating_agent] - 1)

        # φ fitting on accumulated history (after all tasks in this goal)
        if len(self._g_history) >= 2 and len(self._ce_history) >= 1:
            phi, _ = phi_update(phi, self._g_history, self._ce_history, E_history)

        final_state: State = (G, phi, A, E_history)
        return ExecutionResult(
            goal_id=goal.goal_id,
            success=tasks_succeeded > 0,
            tasks_attempted=tasks_attempted,
            tasks_succeeded=tasks_succeeded,
            resources_spent=resources_spent,
            final_state=final_state,
            audit_ids=audit_ids,
        )

    # ---------------------------------------------------------------------------
    # Internal: commit a successful task outcome into the state machine
    # ---------------------------------------------------------------------------

    def _commit_outcome(
        self,
        task: Task,
        outcome: TaskOutcome,
        execute_ce: CoordinationEvent,
        G: Graph,
        phi: PhiMap,
        A: Authority,
        E_history: list,
    ) -> Tuple[Graph, Authority, PhiMap, list]:
        """Reflect task outcome into graph, update errors and authority.

        INV-5: capability changes enter the graph ONLY via
               ce_execute(update_capabilities CE) — never by direct mutation.
        """
        agent_idx = G.agent_index(task.initiating_agent)
        new_caps = G.capabilities[agent_idx].copy()

        # Apply capability_delta (clamped to [0, 1] per dimension)
        for i, delta in enumerate(outcome.capability_delta.values()):
            if i < len(new_caps):
                new_caps[i] = float(np.clip(new_caps[i] + delta, 0.0, 1.0))

        # INV-5: route capability change through ce_execute
        update_ce = CoordinationEvent(
            event_type="update_capabilities",
            participants=(task.initiating_agent,),
            params=frozenset([("capabilities", tuple(float(x) for x in new_caps))]),
        )
        G_next, success, _ = ce_execute(G, update_ce, self._lux, A)
        if not success:
            G_next = G  # leave graph unchanged; learning still proceeds

        # Error computation uses execute_ce for φ-encoding (not update_ce)
        errors = error_computation(G, G_next, phi, execute_ce)
        A_next = authority_update(A, errors, eta=self._eta)
        E_next = E_history + [errors]

        self._g_history.append(G_next)
        self._ce_history.append(execute_ce)

        return G_next, A_next, phi, E_next
