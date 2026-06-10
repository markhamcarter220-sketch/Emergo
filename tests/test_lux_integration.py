"""Integration tests for RealLuxBridge ↔ PyLuxGate.

These tests require `lux_kernel` to be built from the Lux-V1.0 repo.
They are automatically skipped in environments where the extension is absent.

Build:
    cd /path/to/Lux-V1.0 && maturin develop --features python

Run:
    python -m pytest tests/test_lux_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# Auto-skip entire module if lux_kernel not built
lux_kernel = pytest.importorskip("lux_kernel")

from emergo.lux_bridge import RealLuxBridge  # noqa: E402
from emergo.types import Authority, CoordinationEvent, Graph  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph(n: int = 3) -> Graph:
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = 0.4
    caps = np.ones((n, 4)) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _auth(scores: dict[str, float] | None = None, n: int = 3) -> Authority:
    ids = tuple(f"agent_{i}" for i in range(n))
    s = scores or {a: 0.5 for a in ids}
    return Authority(scores=s, baseline=0.5)


def _ce(event_type: str, *participants: str, **params: float) -> CoordinationEvent:
    return CoordinationEvent(
        event_type=event_type,
        participants=tuple(participants),
        params=frozenset(params.items()),
    )


@pytest.fixture
def bridge() -> RealLuxBridge:
    return RealLuxBridge(
        authority_threshold=0.3,
        add_agent_threshold=0.6,
        max_agents=10,
    )


# ---------------------------------------------------------------------------
# Authorization: standard CEs
# ---------------------------------------------------------------------------


class TestRealBridgeAuthorize:
    def test_high_authority_add_edge_approved(self, bridge: RealLuxBridge) -> None:
        G = _graph()
        A = _auth({"agent_0": 0.7, "agent_1": 0.5, "agent_2": 0.5})
        CE = _ce("add_edge", "agent_0", "agent_1", weight=0.5)
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is True
        assert result.reason == "authorized"

    def test_low_authority_add_edge_denied(self, bridge: RealLuxBridge) -> None:
        G = _graph()
        A = _auth({"agent_0": 0.1, "agent_1": 0.5, "agent_2": 0.5})
        CE = _ce("add_edge", "agent_0", "agent_1", weight=0.5)
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False
        assert "authority" in result.reason

    def test_auth_result_fields_complete(self, bridge: RealLuxBridge) -> None:
        """AuthResult must have all four required fields populated."""
        G = _graph()
        A = _auth()
        CE = _ce("add_edge", "agent_0", "agent_1", weight=0.5)
        result = bridge.authorize_ce(CE, G, A)
        assert isinstance(result.authorized, bool)
        assert isinstance(result.reason, str)
        assert isinstance(result.capability_verified, bool)
        assert isinstance(result.resource_reserved, float)

    def test_unknown_event_type_denied(self, bridge: RealLuxBridge) -> None:
        G = _graph()
        A = _auth({"agent_0": 0.9, "agent_1": 0.9, "agent_2": 0.9})
        CE = _ce("unknown_type", "agent_0", "agent_1")
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False

    def test_empty_participants_denied(self, bridge: RealLuxBridge) -> None:
        G = _graph()
        A = _auth()
        CE = CoordinationEvent(
            event_type="add_edge",
            participants=(),
            params=frozenset(),
        )
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False


# ---------------------------------------------------------------------------
# Authorization: add_agent (topology-bounded + elevated threshold)
# ---------------------------------------------------------------------------


class TestRealBridgeAddAgent:
    def test_add_agent_high_authority_approved(self, bridge: RealLuxBridge) -> None:
        G = _graph(3)
        A = _auth({"agent_0": 0.7, "agent_1": 0.5, "agent_2": 0.5})
        CE = _ce("add_agent", "agent_0")
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is True

    def test_add_agent_insufficient_elevated_authority_denied(
        self, bridge: RealLuxBridge
    ) -> None:
        G = _graph(3)
        A = _auth({"agent_0": 0.4, "agent_1": 0.5, "agent_2": 0.5})
        # 0.4 > authority_threshold(0.3) but < add_agent_threshold(0.6)
        CE = _ce("add_agent", "agent_0")
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False
        assert "add_agent" in result.reason

    def test_add_agent_topology_bound_denied(self, bridge: RealLuxBridge) -> None:
        """graph_size == max_agents must be denied."""
        G = _graph(10)  # max_agents=10
        A = _auth({f"agent_{i}": 0.8 for i in range(10)}, n=10)
        CE = _ce("add_agent", "agent_0")
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False
        assert "topology" in result.reason


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


class TestRealBridgeFailClosed:
    def test_returns_authresult_on_exception(self, bridge: RealLuxBridge) -> None:
        """Corrupted inputs must return AuthResult(False), never raise."""
        G = _graph()
        A = _auth()
        # Participant not in authority_scores → triggers proposer-unknown path
        CE = _ce("add_edge", "ghost_agent", "agent_1")
        result = bridge.authorize_ce(CE, G, A)
        assert result.authorized is False
        assert isinstance(result.reason, str)
