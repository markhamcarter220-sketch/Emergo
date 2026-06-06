"""Lux: the single authorization gate for all CE execution in Emergo.

This module provides the Lux class used throughout ce_execution, executor,
and kernel.  All decisions delegate to an underlying LuxBridge.

Interface contract:
  - lux.authorize(CE, G, A)      → bool     [used by ce_execute; no resource deduction]
  - lux.authorize_full(CE, G, A) → AuthResult [used by Executor; includes pre-deduction]
  - lux.audit(...)               → str      [writes immutable record]
  - lux.grant_capability(...)               [system init only; never called by CEs]

Backward compatibility: Lux() with no args still works exactly as before,
using SimulatedLuxBridge under the hood.
"""
from __future__ import annotations

from typing import Optional

from emergo.lux_bridge import AuthResult, LuxBridge, make_lux_bridge
from emergo.types import Authority, CoordinationEvent, Graph


class Lux:
    """Single authorization gate.  Stateless wrapper around a LuxBridge.

    Constructed with a bridge=None, it creates a SimulatedLuxBridge via
    make_lux_bridge() (which respects EMERGO_LUX_MODE env var).
    """

    def __init__(
        self,
        min_authority: float = 0.1,
        bridge: Optional[LuxBridge] = None,
    ) -> None:
        self.min_authority = min_authority
        self._bridge: LuxBridge = bridge if bridge is not None else make_lux_bridge()

    @property
    def bridge(self) -> LuxBridge:
        return self._bridge

    def authorize(self, CE: CoordinationEvent, G: Graph, A: Authority) -> bool:
        """Check-only authorization (no resource deduction).

        Used by ce_execute() — the graph mutation path.
        Resource deduction is intentionally excluded here: it is the Executor's
        responsibility to call authorize_full() before task execution.
        """
        return self._bridge.authorize_ce(
            CE, G, A, self.min_authority, reserve_resources=False
        ).authorized

    def authorize_full(
        self, CE: CoordinationEvent, G: Graph, A: Authority
    ) -> AuthResult:
        """Full authorization with resource pre-deduction.

        Used by Executor before task execution.
        If this returns authorized=True, resource_reserved > 0 for execute_task CEs.
        The caller MUST call refund_resource() if the task subsequently fails.
        """
        return self._bridge.authorize_ce(
            CE, G, A, self.min_authority, reserve_resources=True
        )

    def check_capability(self, agent_id: str, capability: str) -> bool:
        return self._bridge.check_capability(agent_id, capability)

    def deduct_resource(self, agent_id: str, resource: str, amount: float) -> bool:
        return self._bridge.deduct_resource(agent_id, resource, amount)

    def refund_resource(self, agent_id: str, resource: str, amount: float) -> None:
        self._bridge.refund_resource(agent_id, resource, amount)

    def grant_capability(self, agent_id: str, capability: str) -> None:
        """Grant a capability via Lux.  System-init only; never called by Emergo CEs."""
        self._bridge.grant_capability(agent_id, capability)

    def audit(
        self,
        ce_type: str,
        agent_ids: tuple,
        success: bool,
        details: Optional[dict] = None,
        resource_deducted: float = 0.0,
    ) -> str:
        """Write immutable audit record.  Returns audit_id."""
        return self._bridge.audit(ce_type, agent_ids, success, details, resource_deducted)
