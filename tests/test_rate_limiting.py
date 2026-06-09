"""Tests for emergo.rate_limiting — per-agent rate limiting and blast-radius controls."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from emergo import Graph, make_initial_authority
from emergo.lux_bridge import SimulatedLuxBridge
from emergo.rate_limiting import (
    RateLimitConfig,
    RateLimitedLuxBridge,
    _InFlightTracker,
    _TokenBucket,
)
from emergo.types import Authority, CoordinationEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(n: int = 3) -> Graph:
    ids = tuple(f"a{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
    caps = np.ones((n, 2))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _make_authority(agent_ids: tuple, score: float = 0.5) -> Authority:
    return make_initial_authority(agent_ids, baseline=score)


def _add_edge_ce(from_id: str, to_id: str) -> CoordinationEvent:
    return CoordinationEvent(
        event_type="add_edge",
        participants=(from_id, to_id),
        params=frozenset([("weight", 0.5)]),
    )


# ---------------------------------------------------------------------------
# _TokenBucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_starts_full(self) -> None:
        bucket = _TokenBucket(rate=10.0, capacity=5.0)
        assert bucket.available == pytest.approx(5.0)

    def test_consume_succeeds_when_full(self) -> None:
        bucket = _TokenBucket(rate=10.0, capacity=5.0)
        assert bucket.consume() is True

    def test_consume_depletes_tokens(self) -> None:
        bucket = _TokenBucket(rate=1.0, capacity=3.0)
        for _ in range(3):
            assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refills_over_time(self) -> None:
        bucket = _TokenBucket(rate=100.0, capacity=5.0)
        for _ in range(5):
            bucket.consume()
        assert bucket.available == pytest.approx(0.0, abs=0.5)
        time.sleep(0.05)  # 50ms → ~5 tokens at 100/s
        assert bucket.consume() is True

    def test_capacity_cap(self) -> None:
        bucket = _TokenBucket(rate=1000.0, capacity=3.0)
        time.sleep(0.01)
        assert bucket.available <= 3.0 + 1e-9

    def test_thread_safe_concurrent_consume(self) -> None:
        bucket = _TokenBucket(rate=1000.0, capacity=100.0)
        results = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(10):
                ok = bucket.consume()
                with lock:
                    results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 50 attempts with 100 capacity — all should succeed
        assert all(results)


# ---------------------------------------------------------------------------
# _InFlightTracker
# ---------------------------------------------------------------------------


class TestInFlightTracker:
    def test_starts_at_zero(self) -> None:
        tracker = _InFlightTracker()
        assert tracker.count("a0") == 0

    def test_increment_and_count(self) -> None:
        tracker = _InFlightTracker()
        tracker.increment("a0")
        tracker.increment("a0")
        assert tracker.count("a0") == 2

    def test_decrement_clamps_at_zero(self) -> None:
        tracker = _InFlightTracker()
        tracker.decrement("a0")  # below zero — should clamp
        assert tracker.count("a0") == 0

    def test_snapshot(self) -> None:
        tracker = _InFlightTracker()
        tracker.increment("a0")
        tracker.increment("a1")
        snap = tracker.snapshot()
        assert snap["a0"] == 1
        assert snap["a1"] == 1

    def test_decrement_reduces_count(self) -> None:
        tracker = _InFlightTracker()
        tracker.increment("a0")
        tracker.increment("a0")
        tracker.decrement("a0")
        assert tracker.count("a0") == 1


# ---------------------------------------------------------------------------
# RateLimitedLuxBridge — rate limit behavior
# ---------------------------------------------------------------------------


class TestRateLimitedLuxBridgeRateLimit:
    def _make_bridge(self, rate: float = 10.0, burst: float = 3.0) -> RateLimitedLuxBridge:
        config = RateLimitConfig(
            max_proposals_per_second=rate,
            burst_allowance=burst,
            max_in_flight_per_agent=0,  # disabled for these tests
        )
        return RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)

    def test_allows_within_burst(self) -> None:
        bridge = self._make_bridge(rate=5.0, burst=3.0)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        # First 3 should succeed (burst_allowance=3)
        results = [bridge.authorize_ce(ce, G, A) for _ in range(3)]
        assert all(r.authorized for r in results)

    def test_denies_when_burst_exhausted(self) -> None:
        bridge = self._make_bridge(rate=0.01, burst=2.0)  # very slow refill
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        # Exhaust burst
        bridge.authorize_ce(ce, G, A)
        bridge.authorize_ce(ce, G, A)
        # Third should be denied
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is False
        assert "Rate limit" in result.reason

    def test_rate_limit_denial_increments_stats(self) -> None:
        bridge = self._make_bridge(rate=0.01, burst=1.0)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)  # consumes burst
        bridge.authorize_ce(ce, G, A)  # first denial

        stats = bridge.rate_limit_stats()
        assert stats["rate_limit_denials"] >= 1

    def test_independent_buckets_per_agent(self) -> None:
        bridge = self._make_bridge(rate=0.01, burst=1.0)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce_a0 = _add_edge_ce("a0", "a1")
        ce_a1 = _add_edge_ce("a1", "a0")

        # a0 exhausts its bucket
        bridge.authorize_ce(ce_a0, G, A)
        bridge.authorize_ce(ce_a0, G, A)  # denied

        # a1's bucket is independent — first attempt should succeed
        result_a1 = bridge.authorize_ce(ce_a1, G, A)
        assert result_a1.authorized is True

    def test_reset_rate_limits_single_agent(self) -> None:
        bridge = self._make_bridge(rate=0.01, burst=1.0)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)  # exhaust
        bridge.authorize_ce(ce, G, A)  # denied

        bridge.reset_rate_limits("a0")
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is True

    def test_reset_rate_limits_all_agents(self) -> None:
        bridge = self._make_bridge(rate=0.01, burst=1.0)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce0 = _add_edge_ce("a0", "a1")
        ce1 = _add_edge_ce("a1", "a0")

        bridge.authorize_ce(ce0, G, A)  # exhaust a0
        bridge.authorize_ce(ce1, G, A)  # exhaust a1

        bridge.reset_rate_limits()  # reset all
        assert bridge.authorize_ce(ce0, G, A).authorized is True
        assert bridge.authorize_ce(ce1, G, A).authorized is True


# ---------------------------------------------------------------------------
# RateLimitedLuxBridge — blast-radius behavior
# ---------------------------------------------------------------------------


class TestRateLimitedLuxBridgeBlastRadius:
    def _make_bridge(self, max_in_flight: int = 2) -> RateLimitedLuxBridge:
        config = RateLimitConfig(
            max_proposals_per_second=1000.0,  # effectively unlimited
            burst_allowance=1000.0,
            max_in_flight_per_agent=max_in_flight,
        )
        return RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)

    def test_allows_up_to_limit(self) -> None:
        bridge = self._make_bridge(max_in_flight=3)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        results = [bridge.authorize_ce(ce, G, A) for _ in range(3)]
        assert all(r.authorized for r in results)

    def test_denies_over_limit(self) -> None:
        bridge = self._make_bridge(max_in_flight=2)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)
        bridge.authorize_ce(ce, G, A)
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is False
        assert "Blast-radius" in result.reason

    def test_blast_radius_denial_increments_stats(self) -> None:
        bridge = self._make_bridge(max_in_flight=1)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)  # fills slot
        bridge.authorize_ce(ce, G, A)  # denied

        stats = bridge.rate_limit_stats()
        assert stats["blast_radius_denials"] >= 1

    def test_refund_decrements_in_flight(self) -> None:
        bridge = self._make_bridge(max_in_flight=1)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)  # in-flight = 1
        bridge.refund_resource("a0", "compute", 0.0)  # simulates CE rollback → in-flight = 0

        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is True

    def test_audit_decrements_in_flight(self) -> None:
        bridge = self._make_bridge(max_in_flight=1)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)  # in-flight = 1
        bridge.audit("add_edge", ("a0",), True)  # CE resolved → in-flight = 0

        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is True

    def test_disabled_when_max_in_flight_zero(self) -> None:
        config = RateLimitConfig(
            max_proposals_per_second=1000.0, burst_allowance=1000.0, max_in_flight_per_agent=0
        )
        bridge = RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        # Should never deny due to blast-radius (disabled)
        results = [bridge.authorize_ce(ce, G, A) for _ in range(20)]
        assert all(r.authorized for r in results)


# ---------------------------------------------------------------------------
# RateLimitedLuxBridge — delegation
# ---------------------------------------------------------------------------


class TestRateLimitedLuxBridgeDelegation:
    def _make_bridge(self) -> RateLimitedLuxBridge:
        config = RateLimitConfig(
            max_proposals_per_second=1000.0,
            burst_allowance=1000.0,
            max_in_flight_per_agent=0,
        )
        return RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)

    def test_capability_grant_and_check(self) -> None:
        bridge = self._make_bridge()
        bridge.grant_capability("a0", "compute")
        assert bridge.check_capability("a0", "compute") is True
        assert bridge.check_capability("a0", "fly") is False

    def test_resource_deduct_and_refund(self) -> None:
        bridge = self._make_bridge()
        assert bridge.deduct_resource("a0", "compute", 1.0) is True
        bridge.refund_resource("a0", "compute", 1.0)

    def test_audit_returns_string_id(self) -> None:
        bridge = self._make_bridge()
        audit_id = bridge.audit("add_edge", ("a0", "a1"), True)
        assert isinstance(audit_id, str) and len(audit_id) > 0

    def test_inner_auth_still_applies(self) -> None:
        """Inner bridge authority check still fires after rate-limit passes."""
        config = RateLimitConfig(
            max_proposals_per_second=1000.0,
            burst_allowance=1000.0,
            max_in_flight_per_agent=0,
        )
        bridge = RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)
        G = _make_graph(3)
        # Set authority for a0 below min_authority threshold
        A = Authority(scores={"a0": 0.05, "a1": 0.5, "a2": 0.5}, baseline=0.5)
        ce = _add_edge_ce("a0", "a1")

        result = bridge.authorize_ce(ce, G, A, min_authority=0.1)
        assert result.authorized is False  # inner bridge rejects for low authority


# ---------------------------------------------------------------------------
# rate_limit_stats
# ---------------------------------------------------------------------------


class TestRateLimitStats:
    def test_stats_initial_state(self) -> None:
        bridge = RateLimitedLuxBridge(SimulatedLuxBridge())
        stats = bridge.rate_limit_stats()
        assert stats["rate_limit_denials"] == 0
        assert stats["blast_radius_denials"] == 0
        assert stats["in_flight"] == {}
        assert stats["token_buckets"] == {}

    def test_stats_after_activity(self) -> None:
        config = RateLimitConfig(
            max_proposals_per_second=1000.0,
            burst_allowance=1000.0,
            max_in_flight_per_agent=0,
        )
        bridge = RateLimitedLuxBridge(SimulatedLuxBridge(), config=config)
        G = _make_graph(3)
        A = _make_authority(G.agent_ids)
        ce = _add_edge_ce("a0", "a1")

        bridge.authorize_ce(ce, G, A)
        stats = bridge.rate_limit_stats()
        assert "a0" in stats["token_buckets"]
        assert stats["token_buckets"]["a0"]["tokens_available"] >= 0
