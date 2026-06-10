"""Hypothesis property tests for core Emergo invariants.

Tests structural invariants across randomly generated inputs, catching
edge cases that hand-written tests cannot exhaustively cover.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np

from emergo import make_initial_phi
from emergo.authority_update import authority_update
from emergo.error_computation import error_computation
from emergo.types import Authority, CoordinationEvent, Errors, ErrorScales, Graph

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def agent_ids_st(draw, min_n: int = 1, max_n: int = 6) -> tuple:
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    return tuple(f"agent_{i}" for i in range(n))


@st.composite
def graph_st(draw, min_n: int = 1, max_n: int = 6) -> Graph:
    ids = draw(agent_ids_st(min_n=min_n, max_n=max_n))
    n = len(ids)
    adj_vals = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n * n,
            max_size=n * n,
        )
    )
    cap_vals = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n * 4,
            max_size=n * 4,
        )
    )
    return Graph(
        agent_ids=ids,
        adjacency=np.array(adj_vals, dtype=float).reshape(n, n),
        capabilities=np.array(cap_vals, dtype=float).reshape(n, 4),
    )


@st.composite
def authority_st(draw, agent_ids: tuple | None = None) -> Authority:
    if agent_ids is None:
        agent_ids = draw(agent_ids_st())
    scores = {
        a: draw(
            st.floats(min_value=0.0, max_value=0.8, allow_nan=False, allow_infinity=False)
        )
        for a in agent_ids
    }
    baseline = draw(
        st.floats(min_value=0.1, max_value=0.7, allow_nan=False, allow_infinity=False)
    )
    return Authority(scores=scores, baseline=baseline)


@st.composite
def errors_st(draw, agent_ids: tuple | None = None) -> Errors:
    if agent_ids is None:
        agent_ids = draw(agent_ids_st())
    per_agent = {
        a: draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
        for a in agent_ids
    }
    proposer = (
        draw(st.sampled_from(list(agent_ids))) if agent_ids else None
    )
    return Errors(per_agent=per_agent, proposer_id=proposer)


@st.composite
def error_scales_st(draw) -> ErrorScales:
    return ErrorScales(
        global_scale=draw(
            st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        local_scale=draw(
            st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        alpha=draw(
            st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
        ),
    )


# ---------------------------------------------------------------------------
# Authority update properties
# ---------------------------------------------------------------------------


class TestAuthorityUpdateProperties:
    @given(authority_st(), errors_st(), error_scales_st())
    def test_scores_stay_in_bounds(self, A: Authority, errors: Errors, scales: ErrorScales):
        """INV-11: all scores in [0, 0.8] after any update."""
        result = authority_update(A, errors, scales)
        for agent_id in result.scores:
            score = result.get(agent_id)
            assert 0.0 <= score <= 0.8, f"Score {score} out of [0, 0.8] for {agent_id}"

    @given(authority_st(), error_scales_st())
    def test_empty_errors_unchanged(self, A: Authority, scales: ErrorScales):
        """Empty errors must leave all scores exactly unchanged."""
        empty = Errors(per_agent={}, proposer_id=None)
        result = authority_update(A, empty, scales)
        for agent_id in A.scores:
            assert result.get(agent_id) == A.get(agent_id)

    @given(authority_st(), errors_st(), error_scales_st())
    def test_nonparticipants_unchanged(
        self, A: Authority, errors: Errors, scales: ErrorScales
    ):
        """Agents not in errors.per_agent must be unchanged."""
        result = authority_update(A, errors, scales)
        for agent_id in A.scores:
            if agent_id not in errors.per_agent:
                assert result.get(agent_id) == A.get(agent_id)

    @given(authority_st(), errors_st(), error_scales_st())
    def test_update_is_finite(self, A: Authority, errors: Errors, scales: ErrorScales):
        """No NaN or inf scores after any update."""
        result = authority_update(A, errors, scales)
        for score in result.scores.values():
            assert np.isfinite(score), f"Non-finite score: {score}"

    @given(authority_st(), errors_st(), error_scales_st())
    def test_original_authority_not_mutated(
        self, A: Authority, errors: Errors, scales: ErrorScales
    ):
        """authority_update must not mutate its input."""
        scores_before = dict(A.scores)
        authority_update(A, errors, scales)
        assert A.scores == scores_before


# ---------------------------------------------------------------------------
# ErrorScales properties
# ---------------------------------------------------------------------------


class TestErrorScalesProperties:
    @given(
        error_scales_st(),
        st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=0, max_size=5),
        st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=0, max_size=5),
    )
    def test_scales_stay_above_floor(
        self, scales: ErrorScales, global_errs: list, local_errs: list
    ):
        """Scales must never fall below the 1e-6 floor."""
        updated = scales.update(global_errs, local_errs)
        assert updated.global_scale >= 1e-6
        assert updated.local_scale >= 1e-6

    @given(
        error_scales_st(),
        st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=1, max_size=5),
        st.lists(st.floats(0.0, 10.0, allow_nan=False), min_size=1, max_size=5),
    )
    def test_scales_are_finite(
        self, scales: ErrorScales, global_errs: list, local_errs: list
    ):
        """Updated scales must be finite."""
        updated = scales.update(global_errs, local_errs)
        assert np.isfinite(updated.global_scale)
        assert np.isfinite(updated.local_scale)

    @given(error_scales_st())
    def test_empty_lists_leave_scales_unchanged(self, scales: ErrorScales):
        """Updating with no observations must leave both scales unchanged."""
        updated = scales.update([], [])
        assert updated.global_scale == scales.global_scale
        assert updated.local_scale == scales.local_scale

    @given(error_scales_st())
    def test_update_returns_new_instance(self, scales: ErrorScales):
        """update() must always return a new ErrorScales object."""
        updated = scales.update([1.0], [1.0])
        assert updated is not scales


# ---------------------------------------------------------------------------
# Error computation properties
# ---------------------------------------------------------------------------


_FIXED_PHI = make_initial_phi(d_latent=4, d_features=8, d_ce=4, seed=0)


class TestErrorComputationProperties:
    @given(graph_st(min_n=2))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_errors_nonnegative(self, G: Graph):
        """All per-agent errors must be >= 0."""
        ce = CoordinationEvent(
            event_type="add_edge",
            participants=(G.agent_ids[0], G.agent_ids[1]),
            params=frozenset([("weight", 0.5)]),
        )
        errors = error_computation(G, G, _FIXED_PHI, ce)
        for agent_id, err in errors.per_agent.items():
            assert err >= 0.0, f"Negative error {err} for {agent_id}"

    @given(graph_st(min_n=2))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_proposer_id_equals_first_participant(self, G: Graph):
        """proposer_id must equal the first participant when participants exist."""
        ce = CoordinationEvent(
            event_type="add_edge",
            participants=(G.agent_ids[0], G.agent_ids[1]),
            params=frozenset([("weight", 0.5)]),
        )
        errors = error_computation(G, G, _FIXED_PHI, ce)
        assert errors.proposer_id == G.agent_ids[0]

    @given(graph_st(min_n=2))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_errors_finite(self, G: Graph):
        """All per-agent error values must be finite (no NaN/inf)."""
        ce = CoordinationEvent(
            event_type="add_edge",
            participants=(G.agent_ids[0], G.agent_ids[1]),
            params=frozenset([("weight", 0.5)]),
        )
        errors = error_computation(G, G, _FIXED_PHI, ce)
        for agent_id, err in errors.per_agent.items():
            assert np.isfinite(err), f"Non-finite error {err} for {agent_id}"

    @given(graph_st(min_n=1, max_n=1))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_empty_participants_gives_empty_errors(self, G: Graph):
        """CE with no participants yields empty Errors."""
        ce = CoordinationEvent(
            event_type="add_edge",
            participants=(),
            params=frozenset(),
        )
        errors = error_computation(G, G, _FIXED_PHI, ce)
        assert errors.per_agent == {}
        assert errors.proposer_id is None
