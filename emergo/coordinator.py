"""Multi-Agent Coordinator — parallel proposal serialization via Lux.

MultiAgentCoordinator accepts up to MAX_COORDINATOR_AGENTS simultaneous
CoordinationEvent proposals and applies them sequentially in authority order.

INV-9 (Coordinator Serialization): proposals are sorted by authority score
(descending) before Lux evaluation so the ordering is deterministic and
reproducible.  Conflicting proposals (same directed edge) are detected before
evaluation; the lower-priority one is rejected without touching the ledger.

Each accepted proposal passes through the full atomic sequence:
  ce_execute → error_computation → authority_update

Each rejected proposal writes an audit record (INV-7) but charges no
resources (INV-6 conservation).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import uuid

from emergo.authority_update import authority_update
from emergo.ce_execution import ce_execute
from emergo.config import COORDINATOR_CONFLICT_STRATEGY, MAX_COORDINATOR_AGENTS
from emergo.error_computation import error_computation
from emergo.lux import Lux
from emergo.types import CoordinationEvent, State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class ProposedCE:
    """One agent's CE proposal, with an optional explicit priority override.

    If priority == 0.0 (default), the coordinator fills it from A.get(agent_id).
    """

    agent_id: str
    ce: CoordinationEvent
    priority: float = 0.0


@dataclass
class CoordinationRound:
    """Complete record of one coordination round."""

    round_id: str
    accepted: list[ProposedCE]
    rejected: list[ProposedCE]
    rejection_reasons: dict[str, str]  # agent_id → human-readable reason
    final_state: State
    audit_ids: list[str]
    n_conflicts_detected: int


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _edge_key(ce: CoordinationEvent) -> tuple | None:
    """Return a canonical directed-edge key for edge-touching CEs, else None."""
    if ce.event_type in ("add_edge", "remove_edge") and len(ce.participants) >= 2:
        return (ce.participants[0], ce.participants[1])
    return None


def _detect_conflicts(sorted_proposals: list[ProposedCE]) -> dict[str, str]:
    """Return {agent_id: reason} for lower-priority conflicting proposals.

    Proposals are assumed already sorted highest-priority-first.
    The first proposal that claims an edge wins; subsequent ones are conflicts.
    """
    claimed_edges: dict[tuple, str] = {}  # edge_key → winning agent_id
    conflicts: dict[str, str] = {}
    for p in sorted_proposals:
        key = _edge_key(p.ce)
        if key is None:
            continue
        if key in claimed_edges:
            conflicts[p.agent_id] = f"Edge {key} already claimed by {claimed_edges[key]!r}"
        else:
            claimed_edges[key] = p.agent_id
    return conflicts


# ---------------------------------------------------------------------------
# MultiAgentCoordinator
# ---------------------------------------------------------------------------


class MultiAgentCoordinator:
    """Serializes parallel CE proposals from multiple agents through Lux.

    Usage::

        coord = MultiAgentCoordinator(lux=lux)
        round_ = coord.coordinate([
            ProposedCE("A", ce_A),
            ProposedCE("B", ce_B),
        ], state)
        final_state = round_.final_state

    Design constraints:

    - At most max_agents proposals per round (INV-8 analogue for coordination).
    - Proposals sorted by authority score (INV-9: deterministic serialization).
    - Each accepted CE executes through ce_execute → error_computation →
      authority_update (full atomic sequence).
    - Rejected proposals are audited (INV-7) but no resources are charged (INV-6).
    - phi is carried forward unchanged; phi_update is the kernel's responsibility.
    """

    def __init__(
        self,
        lux: Lux | None = None,
        max_agents: int = MAX_COORDINATOR_AGENTS,
        conflict_strategy: str = COORDINATOR_CONFLICT_STRATEGY,
    ) -> None:
        self._lux = lux or Lux()
        self._max_agents = max_agents
        self._conflict_strategy = conflict_strategy

    def coordinate(
        self,
        proposals: list[ProposedCE],
        state: State,
    ) -> CoordinationRound:
        """Process proposals: fill priorities, sort, detect conflicts, then apply.

        State evolves incrementally as each CE is accepted:
        later proposals see the updated graph and authority from earlier ones.
        """
        round_id = str(uuid.uuid4())
        G, phi, A, E_history = state

        # INV-8 analogue: cap proposal count before any processing
        if len(proposals) > self._max_agents:
            logger.warning(
                "Coordinator %s: %d proposals exceed max_agents=%d; truncating",
                round_id[:8],
                len(proposals),
                self._max_agents,
            )
            proposals = proposals[: self._max_agents]

        # INV-9: fill priorities from current authority, then sort descending
        enriched: list[ProposedCE] = []
        for p in proposals:
            pri = p.priority if p.priority != 0.0 else A.get(p.agent_id)
            enriched.append(ProposedCE(agent_id=p.agent_id, ce=p.ce, priority=pri))
        enriched.sort(key=lambda p: p.priority, reverse=True)

        # Detect conflicts among the sorted proposals
        conflicts = _detect_conflicts(enriched)

        accepted: list[ProposedCE] = []
        rejected: list[ProposedCE] = []
        rejection_reasons: dict[str, str] = {}
        audit_ids: list[str] = []

        for proposal in enriched:
            agent_id = proposal.agent_id

            # Conflict rejection — before touching Lux (INV-6: no ledger charge)
            if agent_id in conflicts:
                reason = conflicts[agent_id]
                rejection_reasons[agent_id] = reason
                rejected.append(proposal)
                aid = self._lux.audit(
                    proposal.ce.event_type,
                    proposal.ce.participants,
                    False,
                    details={
                        "reason": "conflict",
                        "detail": reason,
                        "round_id": round_id,
                    },
                )
                audit_ids.append(aid)
                logger.debug(
                    "Coordinator %s: conflict-rejected %s — %s",
                    round_id[:8],
                    agent_id,
                    reason,
                )
                continue

            # Apply through atomic kernel operations (ce_execute is check-only Lux auth)
            G_next, success, _ = ce_execute(G, proposal.ce, self._lux, A)
            if not success:
                reason = "ce_execute_rejected"
                rejection_reasons[agent_id] = reason
                rejected.append(proposal)
                aid = self._lux.audit(
                    proposal.ce.event_type,
                    proposal.ce.participants,
                    False,
                    details={"reason": reason, "round_id": round_id},
                )
                audit_ids.append(aid)
                logger.debug(
                    "Coordinator %s: Lux-rejected %s",
                    round_id[:8],
                    agent_id,
                )
                continue

            # CE accepted: update errors, authority, and state
            errors = error_computation(G, G_next, phi, proposal.ce)
            A = authority_update(A, errors)
            E_history = [*E_history, errors]
            G = G_next

            accepted.append(proposal)
            aid = self._lux.audit(
                proposal.ce.event_type,
                proposal.ce.participants,
                True,
                details={"round_id": round_id},
            )
            audit_ids.append(aid)
            logger.debug("Coordinator %s: accepted %s", round_id[:8], agent_id)

        final_state: State = (G, phi, A, E_history)
        return CoordinationRound(
            round_id=round_id,
            accepted=accepted,
            rejected=rejected,
            rejection_reasons=rejection_reasons,
            final_state=final_state,
            audit_ids=audit_ids,
            n_conflicts_detected=len(conflicts),
        )
