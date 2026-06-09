"""Rate limiting and blast-radius controls for the Lux authorization layer.

Provides ``RateLimitedLuxBridge`` — a transparent decorator that wraps any
``LuxBridge`` implementation and enforces per-agent proposal rate limits using
the token-bucket algorithm.

This module adds no mandatory dependencies.  All rate-limit state is kept in
memory and is reset on process restart.  For production use with multiple
processes, wire a shared Redis/database backend instead (see ``RateLimitConfig``).

Usage::

    from emergo.lux_bridge import SimulatedLuxBridge
    from emergo.rate_limiting import RateLimitedLuxBridge, RateLimitConfig
    from emergo.lux import Lux

    config = RateLimitConfig(
        max_proposals_per_second=10,   # sustained rate per agent
        burst_allowance=20,            # allow short spikes
        max_in_flight_per_agent=5,     # simultaneous pending CEs
    )
    bridge = RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)
    lux = Lux(bridge=bridge)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

from emergo.lux_bridge import AuthResult, LuxBridge
from emergo.types import Authority, CoordinationEvent, Graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    """Configuration for per-agent rate limiting and blast-radius controls.

    Attributes:
        max_proposals_per_second:  Sustained CE proposal rate per agent (token bucket refill rate).
        burst_allowance:           Maximum burst tokens above the sustained rate.  An agent that
                                   has been idle will accumulate up to this many tokens before
                                   being capped.
        max_in_flight_per_agent:   Maximum pending (accepted but not yet resolved) CEs per agent.
                                   0 = disabled.  Prevents a single agent from flooding the graph
                                   with overlapping mutations.
        window_seconds:            Observation window for rate calculations (informational only).
    """

    max_proposals_per_second: float = 10.0
    burst_allowance: float = 20.0
    max_in_flight_per_agent: int = 5
    window_seconds: float = 60.0


# ---------------------------------------------------------------------------
# Token-bucket implementation
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Thread-safe token bucket for a single agent.

    Refills at `rate` tokens/second up to `capacity`.
    Each call to `consume()` atomically checks-and-deducts one token.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Consume one token.  Returns True on success, False if empty."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = elapsed * self._rate
            self._tokens = min(self._capacity, self._tokens + refill)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def available(self) -> float:
        """Current token count (approximately; may have changed by the time you read it)."""
        with self._lock:
            return self._tokens


# ---------------------------------------------------------------------------
# In-flight tracker
# ---------------------------------------------------------------------------


class _InFlightTracker:
    """Thread-safe tracker for in-flight CE count per agent."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def increment(self, agent_id: str) -> None:
        with self._lock:
            self._counts[agent_id] = self._counts.get(agent_id, 0) + 1

    def decrement(self, agent_id: str) -> None:
        with self._lock:
            count = self._counts.get(agent_id, 0)
            self._counts[agent_id] = max(0, count - 1)

    def count(self, agent_id: str) -> int:
        with self._lock:
            return self._counts.get(agent_id, 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


# ---------------------------------------------------------------------------
# Rate-limited bridge
# ---------------------------------------------------------------------------


class RateLimitedLuxBridge(LuxBridge):
    """LuxBridge wrapper that adds per-agent rate limiting and blast-radius controls.

    Wraps any ``LuxBridge`` implementation.  All non-CE methods delegate directly
    to the inner bridge.  ``authorize_ce`` applies rate-limit checks first; if the
    agent is over limit, authorization is denied immediately (fail-fast) without
    consulting the inner bridge.

    Thread-safe: separate ``_TokenBucket`` per agent, protected by individual locks.

    Args:
        bridge:  Inner bridge to delegate to.
        config:  Rate-limit configuration.  Defaults to ``RateLimitConfig()``.
    """

    def __init__(
        self,
        bridge: LuxBridge,
        config: RateLimitConfig | None = None,
    ) -> None:
        self._inner = bridge
        self._config = config or RateLimitConfig()
        self._buckets: dict[str, _TokenBucket] = {}
        self._in_flight = _InFlightTracker()
        self._bucket_lock = threading.Lock()

        # Statistics (monotonic counters, never reset)
        self._rate_limit_denials = 0
        self._blast_radius_denials = 0
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_for(self, agent_id: str) -> _TokenBucket:
        """Return (or create) the token bucket for an agent."""
        with self._bucket_lock:
            if agent_id not in self._buckets:
                self._buckets[agent_id] = _TokenBucket(
                    rate=self._config.max_proposals_per_second,
                    capacity=self._config.burst_allowance,
                )
            return self._buckets[agent_id]

    def _check_rate_limit(self, agent_id: str) -> bool:
        """Return True if the agent may proceed (token available), False if rate-limited."""
        return self._bucket_for(agent_id).consume()

    def _check_blast_radius(self, agent_id: str) -> bool:
        """Return True if the agent is below the max-in-flight threshold."""
        limit = self._config.max_in_flight_per_agent
        if limit <= 0:
            return True
        return self._in_flight.count(agent_id) < limit

    # ------------------------------------------------------------------
    # LuxBridge implementation
    # ------------------------------------------------------------------

    def authorize_ce(
        self,
        CE: CoordinationEvent,
        G: Graph,
        A: Authority,
        min_authority: float = 0.1,
        reserve_resources: bool = False,
    ) -> AuthResult:
        """Apply rate-limit + blast-radius checks, then delegate to inner bridge."""
        proposer = CE.participants[0] if CE.participants else None

        if proposer:
            # 1. Rate limit check
            if not self._check_rate_limit(proposer):
                with self._stats_lock:
                    self._rate_limit_denials += 1
                avail = self._bucket_for(proposer).available
                logger.debug(
                    "RateLimitedLux: rate-limit DENY %r (%.2f tokens available, need 1)",
                    proposer,
                    avail,
                )
                return AuthResult(
                    authorized=False,
                    reason=f"Rate limit exceeded for agent {proposer!r} "
                    f"(max {self._config.max_proposals_per_second:.1f}/s)",
                    capability_verified=False,
                    resource_reserved=0.0,
                )

            # 2. Blast-radius check
            if not self._check_blast_radius(proposer):
                with self._stats_lock:
                    self._blast_radius_denials += 1
                logger.debug(
                    "RateLimitedLux: blast-radius DENY %r (%d in-flight >= %d)",
                    proposer,
                    self._in_flight.count(proposer),
                    self._config.max_in_flight_per_agent,
                )
                return AuthResult(
                    authorized=False,
                    reason=f"Blast-radius limit reached for agent {proposer!r} "
                    f"({self._in_flight.count(proposer)} in-flight, "
                    f"max {self._config.max_in_flight_per_agent})",
                    capability_verified=False,
                    resource_reserved=0.0,
                )

        # 3. Delegate to inner bridge
        result = self._inner.authorize_ce(CE, G, A, min_authority, reserve_resources)

        # Track in-flight count for accepted CEs
        if result.authorized and proposer:
            self._in_flight.increment(proposer)

        return result

    def check_capability(self, agent_id: str, capability: str) -> bool:
        return self._inner.check_capability(agent_id, capability)

    def deduct_resource(self, agent_id: str, resource: str, amount: float) -> bool:
        return self._inner.deduct_resource(agent_id, resource, amount)

    def refund_resource(self, agent_id: str, resource: str, amount: float) -> None:
        self._inner.refund_resource(agent_id, resource, amount)
        # CE rolled back: decrement in-flight count
        self._in_flight.decrement(agent_id)

    def grant_capability(self, agent_id: str, capability: str) -> None:
        self._inner.grant_capability(agent_id, capability)

    def audit(
        self,
        ce_type: str,
        agent_ids: tuple,
        success: bool,
        details: dict[str, Any] | None = None,
        resource_deducted: float = 0.0,
    ) -> str:
        audit_id = self._inner.audit(ce_type, agent_ids, success, details, resource_deducted)
        # CE resolved (success or failure): decrement in-flight for all participants
        for agent_id in agent_ids:
            self._in_flight.decrement(str(agent_id))
        return audit_id

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def rate_limit_stats(self) -> dict[str, Any]:
        """Return a snapshot of rate-limiting statistics."""
        with self._stats_lock:
            denials = self._rate_limit_denials
            blast = self._blast_radius_denials
        with self._bucket_lock:
            buckets = {
                aid: {"tokens_available": round(b.available, 2)} for aid, b in self._buckets.items()
            }
        return {
            "rate_limit_denials": denials,
            "blast_radius_denials": blast,
            "in_flight": self._in_flight.snapshot(),
            "token_buckets": buckets,
        }

    def reset_rate_limits(self, agent_id: str | None = None) -> None:
        """Reset token buckets.  Pass agent_id to reset one agent, or None for all."""
        with self._bucket_lock:
            if agent_id is not None:
                self._buckets.pop(agent_id, None)
            else:
                self._buckets.clear()
