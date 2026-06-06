"""Tests for LuxBridge / SimulatedLuxBridge.

Covers: capability management, atomic resource ledger, CE authorization,
audit trail, fail-closed behavior, and the reserve_resources flag.
"""
import threading
import numpy as np
import pytest

from emergo.lux_bridge import SimulatedLuxBridge, AuthResult
from emergo.types import Authority, CoordinationEvent, Graph
from tests.conftest import make_ce


def _graph() -> Graph:
    adj = np.array([[0, 1], [1, 0]], dtype=float)
    caps = np.ones((2, 2)) * 0.5
    return Graph(agent_ids=("A", "B"), adjacency=adj, capabilities=caps)


def _auth(scores=None) -> Authority:
    scores = scores or {"A": 0.5, "B": 0.5}
    return Authority(scores=scores, baseline=0.5)


# ---------------------------------------------------------------------------
# Capability management
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_grant_and_check(self):
        b = SimulatedLuxBridge()
        b.grant_capability("A", "summarize")
        assert b.check_capability("A", "summarize")

    def test_check_returns_false_without_grant(self):
        b = SimulatedLuxBridge()
        assert not b.check_capability("A", "summarize")

    def test_revoke_removes_capability(self):
        b = SimulatedLuxBridge()
        b.grant_capability("A", "summarize")
        b.revoke_capability("A", "summarize")
        assert not b.check_capability("A", "summarize")

    def test_different_agents_independent(self):
        b = SimulatedLuxBridge()
        b.grant_capability("A", "cap_x")
        assert not b.check_capability("B", "cap_x")


# ---------------------------------------------------------------------------
# Resource ledger
# ---------------------------------------------------------------------------

class TestResourceLedger:
    def test_initial_budget(self):
        b = SimulatedLuxBridge(initial_budget=50.0)
        assert b.get_balance("A") == pytest.approx(50.0)

    def test_deduct_reduces_balance(self):
        b = SimulatedLuxBridge(initial_budget=100.0)
        ok = b.deduct_resource("A", "compute", 30.0)
        assert ok
        assert b.get_balance("A") == pytest.approx(70.0)

    def test_deduct_fails_when_insufficient(self):
        b = SimulatedLuxBridge(initial_budget=10.0)
        ok = b.deduct_resource("A", "compute", 20.0)
        assert not ok
        assert b.get_balance("A") == pytest.approx(10.0)  # unchanged

    def test_refund_restores_balance(self):
        b = SimulatedLuxBridge(initial_budget=100.0)
        b.deduct_resource("A", "compute", 40.0)
        b.refund_resource("A", "compute", 40.0)
        assert b.get_balance("A") == pytest.approx(100.0)

    def test_deduct_refund_net_zero(self):
        b = SimulatedLuxBridge(initial_budget=100.0)
        b.deduct_resource("A", "compute", 25.0)
        b.refund_resource("A", "compute", 25.0)
        assert b.get_balance("A") == pytest.approx(100.0)

    def test_exact_balance_deduction_succeeds(self):
        b = SimulatedLuxBridge(initial_budget=5.0)
        ok = b.deduct_resource("A", "compute", 5.0)
        assert ok
        assert b.get_balance("A") == pytest.approx(0.0)

    def test_deduct_is_atomic_under_concurrent_threads(self):
        """Two threads racing to deduct the same budget: only one should win."""
        b = SimulatedLuxBridge(initial_budget=1.0)
        results = []

        def try_deduct():
            results.append(b.deduct_resource("A", "compute", 1.0))

        t1 = threading.Thread(target=try_deduct)
        t2 = threading.Thread(target=try_deduct)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Exactly one thread should have succeeded
        assert results.count(True) == 1
        assert results.count(False) == 1
        assert b.get_balance("A") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CE Authorization — structural CEs
# ---------------------------------------------------------------------------

class TestAuthorizeStructural:
    def test_add_edge_authorized(self):
        b = SimulatedLuxBridge()
        G = _graph()
        A = _auth({"A": 0.5, "B": 0.5})
        ce = make_ce("add_edge", ("A", "B"), weight=1.0)
        r = b.authorize_ce(ce, G, A)
        assert r.authorized
        assert r.resource_reserved == pytest.approx(0.0)

    def test_add_edge_low_authority_denied(self):
        b = SimulatedLuxBridge()
        G = _graph()
        A = _auth({"A": 0.05, "B": 0.5})
        ce = make_ce("add_edge", ("A", "B"), weight=1.0)
        r = b.authorize_ce(ce, G, A, min_authority=0.1)
        assert not r.authorized

    def test_add_edge_nonexistent_agent_denied(self):
        b = SimulatedLuxBridge()
        G = _graph()
        A = _auth({"A": 0.5, "Z": 0.5})
        ce = make_ce("add_edge", ("A", "Z"), weight=1.0)
        r = b.authorize_ce(ce, G, A)
        assert not r.authorized

    def test_add_agent_succeeds_for_new_id(self):
        b = SimulatedLuxBridge()
        G = _graph()
        A = _auth()
        ce = make_ce("add_agent", (), agent_id="C")
        r = b.authorize_ce(ce, G, A)
        assert r.authorized

    def test_add_agent_fails_for_existing_id(self):
        b = SimulatedLuxBridge()
        G = _graph()
        A = _auth()
        ce = make_ce("add_agent", (), agent_id="A")
        r = b.authorize_ce(ce, G, A)
        assert not r.authorized


# ---------------------------------------------------------------------------
# CE Authorization — execute_task CEs
# ---------------------------------------------------------------------------

class TestAuthorizeExecuteTask:
    def test_execute_task_authorized_with_capability_and_budget(self):
        b = SimulatedLuxBridge(initial_budget=100.0)
        b.grant_capability("A", "cap_x")
        G = _graph()
        A = _auth()
        ce = make_ce(
            "execute_task", ("A",),
            capability="cap_x", resource_cost=5.0, resource="compute",
        )
        r = b.authorize_ce(ce, G, A, reserve_resources=True)
        assert r.authorized
        assert r.capability_verified
        assert r.resource_reserved == pytest.approx(5.0)

    def test_execute_task_denied_without_capability(self):
        b = SimulatedLuxBridge(initial_budget=100.0)
        G = _graph()
        A = _auth()
        ce = make_ce(
            "execute_task", ("A",),
            capability="cap_x", resource_cost=1.0, resource="compute",
        )
        r = b.authorize_ce(ce, G, A, reserve_resources=True)
        assert not r.authorized
        assert r.resource_reserved == pytest.approx(0.0)  # nothing deducted

    def test_execute_task_denied_insufficient_budget(self):
        b = SimulatedLuxBridge(initial_budget=2.0)
        b.grant_capability("A", "cap_x")
        G = _graph()
        A = _auth()
        ce = make_ce(
            "execute_task", ("A",),
            capability="cap_x", resource_cost=10.0, resource="compute",
        )
        r = b.authorize_ce(ce, G, A, reserve_resources=True)
        assert not r.authorized
        assert r.resource_reserved == pytest.approx(0.0)
        assert b.get_balance("A") == pytest.approx(2.0)  # unchanged

    def test_execute_task_check_only_does_not_deduct(self):
        """reserve_resources=False: capability checked but no ledger deduction."""
        b = SimulatedLuxBridge(initial_budget=100.0)
        b.grant_capability("A", "cap_x")
        G = _graph()
        A = _auth()
        ce = make_ce(
            "execute_task", ("A",),
            capability="cap_x", resource_cost=5.0, resource="compute",
        )
        r = b.authorize_ce(ce, G, A, reserve_resources=False)
        # Authorized (capability present) but no deduction
        assert r.authorized
        assert r.resource_reserved == pytest.approx(0.0)
        assert b.get_balance("A") == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_audit_produces_record(self):
        b = SimulatedLuxBridge()
        aid = b.audit("add_edge", ("A", "B"), True, details={"weight": 0.5})
        log = b.get_audit_log()
        assert len(log) == 1
        assert log[0]["audit_id"] == aid
        assert log[0]["success"] is True

    def test_audit_log_is_append_only(self):
        b = SimulatedLuxBridge()
        b.audit("add_edge", ("A",), True)
        b.audit("remove_edge", ("A", "B"), False)
        log = b.get_audit_log()
        assert len(log) == 2
        # Modifying the returned copy does not affect the internal log
        log.clear()
        assert len(b.get_audit_log()) == 2

    def test_failed_audit_records_failure(self):
        b = SimulatedLuxBridge()
        b.audit("execute_task", ("A",), False, details={"reason": "capability_missing"})
        log = b.get_audit_log()
        assert log[0]["success"] is False
        assert log[0]["details"]["reason"] == "capability_missing"

    def test_audit_records_resource_deducted(self):
        b = SimulatedLuxBridge()
        b.audit("execute_task", ("A",), True, resource_deducted=3.5)
        log = b.get_audit_log()
        assert log[0]["resource_deducted"] == pytest.approx(3.5)
