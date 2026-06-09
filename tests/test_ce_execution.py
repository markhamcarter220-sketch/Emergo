"""Tests for CE_Execution: atomicity, determinism, all CE types."""

import numpy as np
import pytest

from emergo.ce_execution import ce_execute
from tests.conftest import make_ce


class TestAddEdge:
    def test_adds_directed_edge(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_edge", ("A", "C"), weight=0.7)
        G_next, ok, _parts = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        i = G_next.agent_index("A")
        j = G_next.agent_index("C")
        assert G_next.adjacency[i, j] == pytest.approx(0.7)

    def test_does_not_modify_source(self, three_agent_graph, default_authority, lux):
        original_adj = three_agent_graph.adjacency.copy()
        ce = make_ce("add_edge", ("A", "C"), weight=0.7)
        ce_execute(three_agent_graph, ce, lux, default_authority)
        np.testing.assert_array_equal(three_agent_graph.adjacency, original_adj)

    def test_other_edges_unchanged(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_edge", ("A", "C"), weight=0.7)
        G_next, _, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        i_a, i_b = G_next.agent_index("A"), G_next.agent_index("B")
        assert G_next.adjacency[i_a, i_b] == pytest.approx(1.0)


class TestRemoveEdge:
    def test_zeroes_edge(self, three_agent_graph, default_authority, lux):
        ce = make_ce("remove_edge", ("A", "B"))
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        i, j = G_next.agent_index("A"), G_next.agent_index("B")
        assert G_next.adjacency[i, j] == pytest.approx(0.0)

    def test_nonexistent_edge_remove_still_succeeds(
        self, three_agent_graph, default_authority, lux
    ):
        ce = make_ce("remove_edge", ("A", "C"))  # A→C does not exist
        _G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok  # idempotent: setting 0 to 0 is fine


class TestUpdateCapabilities:
    def test_updates_agent_capabilities(self, three_agent_graph, default_authority, lux):
        new_caps = [0.9, 0.1]
        ce = make_ce("update_capabilities", ("B",), capabilities=new_caps)
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        i = G_next.agent_index("B")
        np.testing.assert_allclose(G_next.capabilities[i], new_caps)


class TestAddAgent:
    def test_adds_node_with_zero_adjacency(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_agent", (), agent_id="D", capabilities=[0.3, 0.7])
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        assert "D" in G_next.agent_ids
        assert G_next.n_agents == 4
        d_idx = G_next.agent_index("D")
        np.testing.assert_array_equal(G_next.adjacency[d_idx, :], 0)
        np.testing.assert_array_equal(G_next.adjacency[:, d_idx], 0)

    def test_duplicate_agent_fails(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_agent", (), agent_id="A")  # A already exists
        _, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert not ok


class TestRemoveAgent:
    def test_removes_agent_and_edges(self, three_agent_graph, default_authority, lux):
        ce = make_ce("remove_agent", ("B",))
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        assert "B" not in G_next.agent_ids
        assert G_next.n_agents == 2
        assert G_next.adjacency.shape == (2, 2)


class TestAuthorization:
    def test_low_authority_blocks_ce(self, three_agent_graph, lux):
        from emergo.types import Authority

        low_auth = Authority(scores={"A": 0.05, "B": 0.5, "C": 0.5}, baseline=0.5)
        ce = make_ce("add_edge", ("A", "C"), weight=1.0)
        _, ok, _ = ce_execute(three_agent_graph, ce, lux, low_auth)
        assert not ok

    def test_unknown_participant_blocked(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_edge", ("A", "Z"), weight=1.0)
        _, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert not ok


class TestImmutability:
    def test_output_adjacency_is_readonly(self, three_agent_graph, default_authority, lux):
        ce = make_ce("add_edge", ("A", "C"), weight=0.5)
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        assert not G_next.adjacency.flags.writeable

    def test_output_capabilities_is_readonly(self, three_agent_graph, default_authority, lux):
        ce = make_ce("update_capabilities", ("A",), capabilities=[0.1, 0.9])
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok
        assert not G_next.capabilities.flags.writeable


class TestSelfLoopRejection:
    def test_add_edge_self_loop_rejected(self, three_agent_graph, default_authority, lux):
        """INV-12: self-loops must be rejected before any graph mutation."""
        ce = make_ce("add_edge", ("A", "A"), weight=1.0)
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok is False
        assert G_next is three_agent_graph

    def test_add_edge_self_loop_graph_unchanged(self, three_agent_graph, default_authority, lux):
        """INV-12: graph must be byte-identical after self-loop rejection."""
        original_adj = three_agent_graph.adjacency.copy()
        ce = make_ce("add_edge", ("B", "B"), weight=0.5)
        G_next, ok, _ = ce_execute(three_agent_graph, ce, lux, default_authority)
        assert ok is False
        np.testing.assert_array_equal(G_next.adjacency, original_adj)
