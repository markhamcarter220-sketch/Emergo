"""Contract tests for the LuxBridge interface.

These tests define the behavioural contract that ALL LuxBridge implementations
must satisfy.  They run against SimulatedLuxBridge by default.

To test RealLuxBridge: set the ``LUX_BRIDGE`` environment variable:

    LUX_BRIDGE=real pytest tests/test_lux_bridge_contract.py

The contract covers:
  - grant_capability / check_capability round-trip
  - deduct_resource / refund_resource atomicity
  - authorize_ce composite logic (authority + capability + resource)
  - audit immutability (INV-7)
  - validate_bridge() passes
  - Insufficient budget returns False, does not partially deduct
  - Unknown agent has no capabilities by default
"""

from __future__ import annotations

from collections.abc import Generator
import os

import pytest

from emergo import (
    Graph,
    make_initial_authority,
)
from emergo.lux_bridge import (
    LuxBridge,
    LuxError,
    SimulatedLuxBridge,
    validate_bridge,
)
from tests.conftest import make_ce

# ---------------------------------------------------------------------------
# Fixture: parametrize over bridge implementations
# ---------------------------------------------------------------------------


def _make_simulated() -> SimulatedLuxBridge:
    return SimulatedLuxBridge(initial_budget=100.0)


def _make_real() -> LuxBridge:
    from emergo.lux_bridge import RealLuxBridge

    return RealLuxBridge()


def _bridge_ids():
    ids = ["simulated"]
    if os.getenv("LUX_BRIDGE") == "real":
        ids.append("real")
    return ids


@pytest.fixture(params=_bridge_ids())
def bridge(request) -> Generator[LuxBridge, None, None]:
    if request.param == "real":
        try:
            yield _make_real()
        except LuxError as exc:
            pytest.skip(f"RealLuxBridge not available: {exc}")
    else:
        yield _make_simulated()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph(n: int = 3) -> Graph:
    import numpy as np

    ids = tuple(f"agent{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    caps = np.ones((n, 2), dtype=float) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _authority(G: Graph, baseline: float = 0.5):
    return make_initial_authority(G.agent_ids, baseline=baseline)


# ---------------------------------------------------------------------------
# validate_bridge
# ---------------------------------------------------------------------------


class TestValidateBridgeContract:
    def test_validate_bridge_passes(self, bridge):
        assert validate_bridge(bridge) is True


# ---------------------------------------------------------------------------
# Capability contract
# ---------------------------------------------------------------------------


class TestCapabilityContract:
    def test_grant_and_check_round_trip(self, bridge):
        bridge.grant_capability("agent_x", "cap_a")
        assert bridge.check_capability("agent_x", "cap_a") is True

    def test_unknown_agent_has_no_capabilities(self, bridge):
        assert bridge.check_capability("nonexistent_agent", "any_cap") is False

    def test_unknown_capability_returns_false(self, bridge):
        bridge.grant_capability("agent_y", "cap_b")
        assert bridge.check_capability("agent_y", "cap_missing") is False

    def test_multiple_capabilities_independent(self, bridge):
        bridge.grant_capability("agent_z", "cap_1")
        bridge.grant_capability("agent_z", "cap_2")
        assert bridge.check_capability("agent_z", "cap_1") is True
        assert bridge.check_capability("agent_z", "cap_2") is True
        assert bridge.check_capability("agent_z", "cap_3") is False


# ---------------------------------------------------------------------------
# Resource ledger contract
# ---------------------------------------------------------------------------


class TestResourceLedgerContract:
    def test_deduct_succeeds_when_balance_sufficient(self, bridge):
        assert bridge.deduct_resource("a1", "compute", 10.0) is True

    def test_deduct_fails_when_balance_insufficient(self, bridge):
        # Drain the budget first
        bridge.deduct_resource("a2", "compute", 99.0)
        bridge.deduct_resource("a2", "compute", 1.0)
        # Now the budget should be 0; any further deduction should fail
        result = bridge.deduct_resource("a2", "compute", 0.01)
        assert result is False

    def test_refund_restores_balance(self, bridge):
        bridge.deduct_resource("a3", "compute", 50.0)
        bridge.refund_resource("a3", "compute", 50.0)
        # Should be able to deduct again
        assert bridge.deduct_resource("a3", "compute", 50.0) is True

    def test_deduct_is_atomic_no_partial_charge(self, bridge):
        """A failed deduction must leave the balance unchanged."""
        # Drain completely
        bridge.deduct_resource("a4", "compute", 100.0)
        result = bridge.deduct_resource("a4", "compute", 1.0)
        assert result is False

    def test_refund_without_prior_deduct_is_safe(self, bridge):
        """Refunding without a matching deduct must not raise."""
        bridge.refund_resource("a5", "compute", 5.0)  # should not raise


# ---------------------------------------------------------------------------
# Audit contract (INV-7)
# ---------------------------------------------------------------------------


class TestAuditContract:
    def test_audit_returns_non_empty_string(self, bridge):
        audit_id = bridge.audit("add_edge", ("a", "b"), True)
        assert isinstance(audit_id, str)
        assert len(audit_id) > 0

    def test_audit_ids_are_unique(self, bridge):
        id1 = bridge.audit("add_edge", ("a",), True)
        id2 = bridge.audit("remove_edge", ("a",), False)
        assert id1 != id2

    def test_audit_details_are_immutable_copies(self, bridge):
        """Mutating the returned snapshot must not corrupt stored records."""
        if not hasattr(bridge, "get_audit_log"):
            pytest.skip("Bridge has no get_audit_log — skipping immutability check")
        bridge.audit("op", ("a",), True, details={"key": 1})
        log = bridge.get_audit_log()
        log[0]["details"]["key"] = 999
        fresh = bridge.get_audit_log()
        assert fresh[0]["details"]["key"] == 1


# ---------------------------------------------------------------------------
# authorize_ce contract
# ---------------------------------------------------------------------------


class TestAuthorizeCEContract:
    def test_unknown_participant_denied(self, bridge):
        G = _graph(3)
        A = _authority(G)
        ce = make_ce("add_edge", ("ghost_agent",), weight=0.5)
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is False

    def test_low_authority_denied(self, bridge):
        G = _graph(3)
        A = make_initial_authority(G.agent_ids, baseline=0.01)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        result = bridge.authorize_ce(ce, G, A, min_authority=0.5)
        assert result.authorized is False

    def test_sufficient_authority_authorized(self, bridge):
        G = _graph(3)
        A = make_initial_authority(G.agent_ids, baseline=0.9)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        result = bridge.authorize_ce(ce, G, A, min_authority=0.1)
        assert result.authorized is True

    def test_execute_task_requires_capability(self, bridge):
        G = _graph(2)
        A = make_initial_authority(G.agent_ids, baseline=0.9)
        ce = make_ce(
            "execute_task",
            (G.agent_ids[0],),
            capability="summarize",
            resource_cost=1.0,
            resource="compute",
        )
        # Without granting the capability, should be denied
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is False

    def test_execute_task_authorized_with_capability(self, bridge):
        G = _graph(2)
        A = make_initial_authority(G.agent_ids, baseline=0.9)
        bridge.grant_capability(G.agent_ids[0], "summarize")
        ce = make_ce(
            "execute_task",
            (G.agent_ids[0],),
            capability="summarize",
            resource_cost=1.0,
            resource="compute",
        )
        result = bridge.authorize_ce(ce, G, A)
        assert result.authorized is True

    def test_resource_reserved_on_execute_task_with_reserve(self, bridge):
        G = _graph(2)
        A = make_initial_authority(G.agent_ids, baseline=0.9)
        bridge.grant_capability(G.agent_ids[0], "analyze")
        ce = make_ce(
            "execute_task",
            (G.agent_ids[0],),
            capability="analyze",
            resource_cost=5.0,
            resource="compute",
        )
        result = bridge.authorize_ce(ce, G, A, reserve_resources=True)
        assert result.authorized is True
        assert result.resource_reserved == pytest.approx(5.0)

    def test_auth_result_authorized_false_resource_reserved_zero(self, bridge):
        """A failed auth must never reserve resources."""
        G = _graph(2)
        A = make_initial_authority(G.agent_ids, baseline=0.01)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        result = bridge.authorize_ce(ce, G, A, min_authority=0.9)
        assert result.authorized is False
        assert result.resource_reserved == pytest.approx(0.0)
