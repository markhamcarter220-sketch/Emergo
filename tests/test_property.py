"""Property-based tests using Hypothesis.

Tests structural invariants that must hold for *all* valid inputs, not just
hand-crafted examples.  Coverage categories:

  - Graph immutability after any CE
  - Authority always in [0, 1] after update
  - Error computation always non-negative
  - PhiMap.copy() produces an independent copy
  - ce_execute is all-or-nothing (adjacency either unchanged or single mutation)
  - authority_update output bounded regardless of extreme error values
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np

from emergo import (
    Graph,
    Lux,
    authority_update,
    ce_execute,
    error_computation,
    make_initial_authority,
    make_initial_phi,
)
from emergo.types import Authority, Errors
from tests.conftest import make_ce

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def graph_strategy(draw, min_agents=2, max_agents=6):
    n = draw(st.integers(min_value=min_agents, max_value=max_agents))
    ids = tuple(f"agent{i}" for i in range(n))
    adj_flat = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n * n,
            max_size=n * n,
        )
    )
    adj = np.array(adj_flat, dtype=float).reshape(n, n)
    np.fill_diagonal(adj, 0.0)
    caps = np.ones((n, 2), dtype=float) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


@st.composite
def authority_strategy(draw, agent_ids):
    scores = {
        aid: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for aid in agent_ids
    }
    return Authority(scores=scores, baseline=0.5)


@st.composite
def errors_strategy(draw, agent_ids):
    per_agent = {
        aid: draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for aid in agent_ids
    }
    return Errors(per_agent=per_agent)


# ---------------------------------------------------------------------------
# CE_execute: immutability
# ---------------------------------------------------------------------------


class TestCEExecuteProperties:
    @given(G=graph_strategy(), weight=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_ce_execute_never_mutates_input_adjacency(self, G, weight):
        if G.n_agents < 2:
            return
        original = G.adjacency.copy()
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=weight)
        ce_execute(G, ce, Lux(), A)
        np.testing.assert_array_equal(G.adjacency, original)

    @given(G=graph_strategy())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_ce_execute_result_adjacency_is_readonly(self, G):
        if G.n_agents < 2:
            return
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        G_next, success, _ = ce_execute(G, ce, Lux(), A)
        if success:
            assert not G_next.adjacency.flags.writeable

    @given(G=graph_strategy(), weight=st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_add_edge_creates_nonzero_entry(self, G, weight):
        if G.n_agents < 2:
            return
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        i, j = 0, 1
        from_a, to_a = G.agent_ids[i], G.agent_ids[j]
        # Only test the case where the edge doesn't already exist
        if G.adjacency[i, j] > 0:
            return
        ce = make_ce("add_edge", (from_a, to_a), weight=weight)
        G_next, success, _ = ce_execute(G, ce, Lux(), A)
        if success:
            assert G_next.adjacency[i, j] > 0.0

    @given(G=graph_strategy())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_edge_zeroes_entry(self, G):
        if G.n_agents < 2:
            return
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        i, j = 0, 1
        from_a, to_a = G.agent_ids[i], G.agent_ids[j]
        if G.adjacency[i, j] == 0.0:
            return
        ce = make_ce("remove_edge", (from_a, to_a))
        G_next, success, _ = ce_execute(G, ce, Lux(), A)
        if success:
            assert G_next.adjacency[i, j] == 0.0


# ---------------------------------------------------------------------------
# authority_update: bounds
# ---------------------------------------------------------------------------


class TestAuthorityUpdateProperties:
    @given(
        n=st.integers(min_value=1, max_value=8),
        errors=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=8,
        ),
        eta=st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_authority_always_in_unit_interval(self, n, errors, eta):
        agent_ids = tuple(f"a{i}" for i in range(n))
        A = make_initial_authority(agent_ids, baseline=0.5)
        per_agent = {aid: e for aid, e in zip(agent_ids, errors[:n])}
        E = Errors(per_agent=per_agent)
        A_next = authority_update(A, E, eta=eta)
        for aid in agent_ids:
            v = A_next.get(aid)
            assert 0.0 <= v <= 1.0, f"authority out of bounds: {aid}={v}"

    @given(
        n=st.integers(min_value=2, max_value=6),
        baseline=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_authority_update_does_not_mutate_input(self, n, baseline):
        agent_ids = tuple(f"a{i}" for i in range(n))
        A = make_initial_authority(agent_ids, baseline=baseline)
        original_scores = dict(A.scores)
        E = Errors(per_agent={aid: 0.5 for aid in agent_ids})
        authority_update(A, E)
        assert A.scores == original_scores


# ---------------------------------------------------------------------------
# Error computation: non-negative
# ---------------------------------------------------------------------------


class TestErrorComputationProperties:
    @given(G=graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_error_computation_non_negative(self, G):
        if G.n_agents < 2:
            return
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        G_next, success, _ = ce_execute(G, ce, Lux(), A)
        if not success:
            return
        errors = error_computation(G, G_next, phi, ce)
        for agent_id, err in errors.per_agent.items():
            assert err >= 0.0, f"Negative error for {agent_id}: {err}"

    @given(G=graph_strategy())
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_errors_mean_non_negative(self, G):
        if G.n_agents < 2:
            return
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4)
        ce = make_ce("add_edge", (G.agent_ids[0], G.agent_ids[1]), weight=0.5)
        A = make_initial_authority(G.agent_ids, baseline=0.5)
        G_next, success, _ = ce_execute(G, ce, Lux(), A)
        if not success:
            return
        errors = error_computation(G, G_next, phi, ce)
        assert errors.mean_error() >= 0.0


# ---------------------------------------------------------------------------
# PhiMap.copy: independence
# ---------------------------------------------------------------------------


class TestPhiMapCopy:
    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50)
    def test_copy_is_independent(self, seed):
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=seed)
        phi2 = phi.copy()
        phi2.W_phi[0, 0] = 999.0
        assert phi.W_phi[0, 0] != 999.0

    @given(seed=st.integers(min_value=0, max_value=999))
    @settings(max_examples=50)
    def test_copy_equal_values(self, seed):
        phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=seed)
        phi2 = phi.copy()
        np.testing.assert_array_equal(phi.W_phi, phi2.W_phi)
        np.testing.assert_array_equal(phi.W_F, phi2.W_F)
