"""Lux Bridge — formal contract between Emergo and the Lux governance layer.

LuxBridge defines exactly what Emergo requires from Lux:
  - Capability verification  (INV-5: Proposal-Only Authority)
  - Atomic resource ledger   (INV-6: Resource Conservation)
  - CE composite authorization
  - Immutable audit trail    (INV-7: Observable + Fail-Closed)

SimulatedLuxBridge: in-memory implementation for development and testing.
  All ledger operations are protected by a threading.Lock (atomic).

RealLuxBridge: thin adapter for actual Lux Python bindings.
  All operations fail-closed: any exception returns unauthorized / False.

Design principle: the boundary between Emergo and Lux is this file.
Nothing in Emergo may touch capabilities or ledgers except through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
import threading
import time
import uuid

from emergo.types import Authority, CoordinationEvent, Graph

logger = logging.getLogger(__name__)

_ADD_AGENT_AUTHORITY_THRESHOLD: float = float(os.getenv("EMERGO_ADD_AGENT_THRESHOLD", "0.6"))


@dataclass(frozen=True)
class AuthResult:
    """Full result of a Lux authorization check.

    resource_reserved > 0 means the Lux ledger has already been charged.
    The caller MUST call refund_resource() if the CE subsequently fails.
    """

    authorized: bool
    reason: str
    capability_verified: bool  # True iff a named capability was checked and confirmed
    resource_reserved: float  # amount pre-deducted from the agent's ledger (0 if none)
    audit_id: str | None = None


class LuxBridge(ABC):
    """Abstract contract between Emergo and the Lux governance layer.

    Implementations MUST be fail-closed: when in doubt, deny.
    """

    @abstractmethod
    def authorize_ce(
        self,
        CE: CoordinationEvent,
        G: Graph,
        A: Authority,
        min_authority: float = 0.1,
        reserve_resources: bool = False,
    ) -> AuthResult:
        """Composite authorization: capability + authority + optional resource pre-deduction.

        reserve_resources=False (used by ce_execute):  check-only, no ledger deduction.
        reserve_resources=True  (used by Executor):    pre-deduct resource on success.

        On failure: AuthResult.authorized=False, resource_reserved=0.0 always.
        """

    @abstractmethod
    def check_capability(self, agent_id: str, capability: str) -> bool:
        """Return True iff agent currently holds the named capability."""

    @abstractmethod
    def deduct_resource(self, agent_id: str, resource: str, amount: float) -> bool:
        """Atomically deduct resource from agent's ledger.

        Returns False (without deducting) if balance insufficient.
        This is the ONLY path through which resources leave an agent's budget.
        """

    @abstractmethod
    def refund_resource(self, agent_id: str, resource: str, amount: float) -> None:
        """Refund a previously deducted amount (CE rollback / failure path).

        Must be called whenever resource_reserved > 0 and the CE did not complete.
        """

    @abstractmethod
    def grant_capability(self, agent_id: str, capability: str) -> None:
        """Grant a capability to an agent.

        Only called by system initializers, never by Emergo CEs directly (INV-5).
        """

    @abstractmethod
    def audit(
        self,
        ce_type: str,
        agent_ids: tuple,
        success: bool,
        details: dict | None = None,
        resource_deducted: float = 0.0,
    ) -> str:
        """Write an immutable audit record.  Returns audit_id.

        Must succeed (or raise LuxError) — never silently swallow failures.
        """


class LuxError(Exception):
    """Raised when Lux is unavailable or returns an unexpected error."""


class LuxConnectionError(LuxError):
    """Raised when the Lux service cannot be reached."""


class LuxTimeoutError(LuxError):
    """Raised when a Lux operation exceeds the timeout threshold."""


class LuxPermissionError(LuxError):
    """Raised when an operation is denied due to permissions (not just authority)."""


class SimulatedLuxBridge(LuxBridge):
    """In-memory Lux implementation for development and testing.

    Capabilities: registered via grant_capability().
    Ledger: each (agent, resource) pair initializes at initial_budget on first access.
    Atomicity: all ledger mutations hold self._lock.
    """

    def __init__(self, initial_budget: float = 100.0) -> None:
        self._initial_budget = initial_budget
        self._capabilities: dict[str, set[str]] = {}
        self._ledger: dict[str, dict[str, float]] = {}
        self._audit_log: list = []
        self._lock = threading.Lock()

    # --- Capability management ---

    def grant_capability(self, agent_id: str, capability: str) -> None:
        with self._lock:
            self._capabilities.setdefault(agent_id, set()).add(capability)
        logger.debug("SimulatedLux: granted capability %r to %r", capability, agent_id)

    def revoke_capability(self, agent_id: str, capability: str) -> None:
        with self._lock:
            self._capabilities.get(agent_id, set()).discard(capability)
        logger.debug("SimulatedLux: revoked capability %r from %r", capability, agent_id)

    def check_capability(self, agent_id: str, capability: str) -> bool:
        with self._lock:
            return capability in self._capabilities.get(agent_id, set())

    # --- Resource ledger ---

    def _ensure_ledger(self, agent_id: str, resource: str) -> None:
        """Initialize ledger entry to initial_budget if not present.  MUST hold lock."""
        self._ledger.setdefault(agent_id, {}).setdefault(resource, self._initial_budget)

    def get_balance(self, agent_id: str, resource: str = "compute") -> float:
        with self._lock:
            self._ensure_ledger(agent_id, resource)
            return self._ledger[agent_id][resource]

    def deduct_resource(self, agent_id: str, resource: str, amount: float) -> bool:
        with self._lock:
            self._ensure_ledger(agent_id, resource)
            balance = self._ledger[agent_id][resource]
            if balance < amount:
                logger.debug(
                    "SimulatedLux: deduct DENIED %r %r %.3f (balance=%.3f)",
                    agent_id,
                    resource,
                    amount,
                    balance,
                )
                return False
            self._ledger[agent_id][resource] = balance - amount
            logger.debug(
                "SimulatedLux: deducted %r %r %.3f (remaining=%.3f)",
                agent_id,
                resource,
                amount,
                self._ledger[agent_id][resource],
            )
            return True

    def refund_resource(self, agent_id: str, resource: str, amount: float) -> None:
        with self._lock:
            self._ensure_ledger(agent_id, resource)
            self._ledger[agent_id][resource] += amount
        logger.debug("SimulatedLux: refunded %r %r %.3f", agent_id, resource, amount)

    def get_resource_snapshot(self) -> dict[str, dict[str, float]]:
        """Return a deep copy of the current ledger state (for observability)."""
        import copy

        with self._lock:
            return copy.deepcopy(self._ledger)

    # --- Audit ---

    def audit(
        self,
        ce_type: str,
        agent_ids: tuple,
        success: bool,
        details: dict | None = None,
        resource_deducted: float = 0.0,
    ) -> str:
        record_id = str(uuid.uuid4())
        record = {
            "audit_id": record_id,
            "timestamp": time.time(),
            "ce_type": ce_type,
            "agent_ids": agent_ids,
            "success": success,
            "resource_deducted": resource_deducted,
            # Deep-copy details so callers cannot mutate stored records (INV-7).
            "details": dict(details or {}),
        }
        with self._lock:
            import copy

            self._audit_log.append(copy.deepcopy(record))
        logger.debug(
            "SimulatedLux: audit %s ce_type=%r success=%s agents=%s",
            record_id[:8],
            ce_type,
            success,
            agent_ids,
        )
        return record_id

    def get_audit_log(self) -> list:
        """Return a deep-copied snapshot of all audit records.

        Callers may freely mutate the returned dicts without affecting stored records.
        """
        import copy

        with self._lock:
            return copy.deepcopy(self._audit_log)

    # --- CE Authorization ---

    def authorize_ce(
        self,
        CE: CoordinationEvent,
        G: Graph,
        A: Authority,
        min_authority: float = 0.1,
        reserve_resources: bool = False,
    ) -> AuthResult:
        """Composite authorization.

        Evaluation order (fail-fast):
          1. Structural checks (participants exist, CE-specific preconditions)
          2. Authority threshold per participant
          3. Capability check (execute_task only)
          4. Resource pre-deduction (execute_task + reserve_resources=True only)
        """
        params = dict(CE.params)

        # --- add_agent: no existing participants required ---
        if CE.event_type == "add_agent":
            new_id = params.get("agent_id")
            if not new_id:
                return AuthResult(False, "add_agent requires agent_id param", False, 0.0)
            if new_id in G.agent_ids:
                return AuthResult(False, f"Agent {new_id!r} already exists in graph", False, 0.0)
            # Gate on proposer authority when a proposer is specified (R5).
            if CE.participants:
                proposer = CE.participants[0]
                if A.get(proposer) < _ADD_AGENT_AUTHORITY_THRESHOLD:
                    return AuthResult(
                        False,
                        f"Proposer {proposer!r} authority {A.get(proposer):.3f}"
                        f" < {_ADD_AGENT_AUTHORITY_THRESHOLD} required for add_agent",
                        False,
                        0.0,
                    )
            return AuthResult(True, "add_agent authorized", False, 0.0)

        # --- All other types: participants must exist and meet authority threshold ---
        for agent_id in CE.participants:
            if agent_id not in G.agent_ids:
                return AuthResult(False, f"Participant {agent_id!r} not in graph", False, 0.0)
            if A.get(agent_id) < min_authority:
                return AuthResult(
                    False,
                    f"Agent {agent_id!r} authority {A.get(agent_id):.3f} < {min_authority}",
                    False,
                    0.0,
                )

        # --- CE-specific structural preconditions ---
        if CE.event_type in ("add_edge", "remove_edge") and len(CE.participants) < 2:
            return AuthResult(False, "Edge CE requires 2 participants", False, 0.0)

        if CE.event_type == "remove_agent" and len(CE.participants) < 1:
            return AuthResult(False, "remove_agent requires 1 participant", False, 0.0)

        if CE.event_type == "update_capabilities":
            if len(CE.participants) < 1:
                return AuthResult(False, "update_capabilities requires 1 participant", False, 0.0)
            if params.get("capabilities") is None:
                return AuthResult(
                    False, "update_capabilities requires capabilities param", False, 0.0
                )

        # --- Capability check (execute_task only) ---
        capability_verified = False
        if CE.event_type == "execute_task":
            required_cap = params.get("capability")
            initiator = CE.participants[0] if CE.participants else None
            if not required_cap or not initiator:
                return AuthResult(
                    False, "execute_task requires capability param and initiator", False, 0.0
                )
            if not self.check_capability(initiator, required_cap):
                return AuthResult(
                    False,
                    f"Agent {initiator!r} lacks capability {required_cap!r}",
                    False,
                    0.0,
                )
            capability_verified = True

        # --- Resource pre-deduction (execute_task + reserve_resources only) ---
        resource_reserved = 0.0
        if CE.event_type == "execute_task" and reserve_resources:
            cost = float(params.get("resource_cost", 1.0))
            resource = str(params.get("resource", "compute"))
            initiator = CE.participants[0]
            if not self.deduct_resource(initiator, resource, cost):
                return AuthResult(
                    False,
                    f"Agent {initiator!r} insufficient {resource!r} balance (need {cost:.2f})",
                    capability_verified,
                    0.0,
                )
            resource_reserved = cost

        return AuthResult(True, "authorized", capability_verified, resource_reserved)


class RealLuxBridge(LuxBridge):
    """Adapter for actual Lux Python bindings.

    Set EMERGO_LUX_MODE=real and ensure the lux package is installed.
    ALL operations are fail-closed: any exception → denied / False.

    Retry logic: authorize_ce, check_capability, and deduct_resource each
    retry up to max_retries times on transient errors before failing closed.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.5) -> None:
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        try:
            import lux as _lux

            self._lux = _lux
        except ImportError as exc:
            raise LuxError(
                "Real Lux bindings not available. "
                "Install the 'lux' package or use EMERGO_LUX_MODE=simulated."
            ) from exc

    def _retry(self, fn: object, *args: object, **kwargs: object) -> object:
        """Call fn(*args, **kwargs) up to max_retries times; fail-closed on last failure."""
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)  # type: ignore[operator]
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * (2**attempt))
        raise last_exc

    def authorize_ce(
        self,
        CE: CoordinationEvent,
        G: Graph,
        A: Authority,
        min_authority: float = 0.1,
        reserve_resources: bool = False,
    ) -> AuthResult:
        try:
            return self._retry(self._lux.authorize_ce, CE, G, A, min_authority, reserve_resources)  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("RealLuxBridge.authorize_ce failed (fail-closed): %s", exc)
            return AuthResult(False, f"Lux error: {exc}", False, 0.0)

    def check_capability(self, agent_id: str, capability: str) -> bool:
        try:
            return bool(self._retry(self._lux.check_capability, agent_id, capability))
        except Exception:
            return False

    def deduct_resource(self, agent_id: str, resource: str, amount: float) -> bool:
        try:
            return bool(self._retry(self._lux.deduct_resource, agent_id, resource, amount))
        except Exception:
            return False

    def refund_resource(self, agent_id: str, resource: str, amount: float) -> None:
        try:
            self._lux.refund_resource(agent_id, resource, amount)
        except Exception as exc:
            logger.warning("RealLuxBridge.refund_resource failed (best-effort): %s", exc)

    def grant_capability(self, agent_id: str, capability: str) -> None:
        try:
            self._lux.grant_capability(agent_id, capability)
        except Exception as exc:
            raise LuxError(f"grant_capability failed: {exc}") from exc

    def audit(
        self,
        ce_type: str,
        agent_ids: tuple,
        success: bool,
        details: object = None,
        resource_deducted: float = 0.0,
    ) -> str:
        try:
            return str(self._lux.audit(ce_type, agent_ids, success, details, resource_deducted))
        except Exception as exc:
            raise LuxError(f"Audit write failed: {exc}") from exc


def make_lux_bridge(mode: str | None = None, **kwargs: object) -> LuxBridge:
    """Factory.  Respects EMERGO_LUX_MODE env var; kwargs forwarded to constructor."""
    from emergo.config import LUX_MODE

    effective_mode = mode or LUX_MODE
    if effective_mode == "real":
        return RealLuxBridge()
    return SimulatedLuxBridge(**kwargs)  # type: ignore[arg-type]


def validate_bridge(bridge: LuxBridge) -> bool:
    """Smoke-test a LuxBridge implementation for basic correctness.

    Runs a minimal sequence of operations against a sentinel agent and checks
    that the contract is upheld.  Returns True on success, raises LuxError on
    any violation so callers can fail-fast at startup rather than mid-run.

    Use this in integration tests or startup health checks to verify that a
    RealLuxBridge actually talks to a live Lux service.
    """
    _AGENT = "__validate_probe__"
    _CAP = "__validate_cap__"
    _RESOURCE = "compute"
    _AMOUNT = 0.01

    # 1. Grant + check capability
    try:
        bridge.grant_capability(_AGENT, _CAP)
    except Exception as exc:
        raise LuxError(f"validate_bridge: grant_capability failed: {exc}") from exc

    if not bridge.check_capability(_AGENT, _CAP):
        raise LuxError("validate_bridge: check_capability returned False immediately after grant")

    # 2. Deduct + refund resource (no-op amounts — just tests plumbing)
    ok = bridge.deduct_resource(_AGENT, _RESOURCE, _AMOUNT)
    if not ok:
        raise LuxError("validate_bridge: deduct_resource denied on fresh agent")
    bridge.refund_resource(_AGENT, _RESOURCE, _AMOUNT)

    # 3. Audit round-trip
    try:
        audit_id = bridge.audit("validate_probe", (_AGENT,), True, {"probe": True}, 0.0)
        if not isinstance(audit_id, str) or not audit_id:
            raise LuxError("validate_bridge: audit returned empty or non-string id")
    except LuxError:
        raise
    except Exception as exc:
        raise LuxError(f"validate_bridge: audit raised unexpected exception: {exc}") from exc

    logger.info("validate_bridge: all checks passed for %s", type(bridge).__name__)
    return True
