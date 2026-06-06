"""Lux: authorization oracle for Coordination Events.

Lux.authorize(CE, G, A) → bool

Rules enforced:
  - Participants must exist in G (except add_agent, which introduces new nodes)
  - Participants must have authority ≥ min_authority
  - Structural preconditions per CE type
"""
from __future__ import annotations

from emergo.types import Authority, CoordinationEvent, Graph


class Lux:
    """Authorization oracle.  Stateless: all decisions derive from (CE, G, A)."""

    def __init__(self, min_authority: float = 0.1) -> None:
        self.min_authority = min_authority

    def authorize(self, CE: CoordinationEvent, G: Graph, A: Authority) -> bool:
        """Return True iff CE is authorized to execute against (G, A)."""
        params = dict(CE.params)

        if CE.event_type == "add_agent":
            new_id = params.get("agent_id")
            if not new_id:
                return False
            return new_id not in G.agent_ids

        # All remaining types require participants to already exist and be authorized
        for agent_id in CE.participants:
            if agent_id not in G.agent_ids:
                return False
            if A.get(agent_id) < self.min_authority:
                return False

        if CE.event_type in ("add_edge", "remove_edge"):
            if len(CE.participants) < 2:
                return False

        if CE.event_type == "remove_agent":
            if len(CE.participants) < 1:
                return False

        if CE.event_type == "update_capabilities":
            if len(CE.participants) < 1:
                return False
            if params.get("capabilities") is None:
                return False

        return True
