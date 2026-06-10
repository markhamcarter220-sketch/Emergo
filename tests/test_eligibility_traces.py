"""Tests for EligibilityTraces and its integration with authority_update."""

from __future__ import annotations

import pytest

from emergo.authority_update import authority_update
from emergo.types import Authority, EligibilityTraces, Errors, ErrorScales


# ---------------------------------------------------------------------------
# Basic EligibilityTraces behaviour
# ---------------------------------------------------------------------------


class TestEligibilityTracesBasic:
    def test_uniform_initializes_all_at_one(self) -> None:
        ids = ("a", "b", "c")
        et = EligibilityTraces.uniform(ids)
        for aid in ids:
            assert et.get(aid) == 1.0

    def test_decay_reduces_all_traces(self) -> None:
        ids = ("a", "b")
        et = EligibilityTraces.uniform(ids, decay=0.5)
        et2 = et.step()
        assert et2.get("a") == pytest.approx(0.5)
        assert et2.get("b") == pytest.approx(0.5)

    def test_step_with_participants_resets_accepted_to_one(self) -> None:
        ids = ("a", "b", "c")
        et = EligibilityTraces.uniform(ids, decay=0.5)
        et2 = et.step(accepted_participants=("a",))
        assert et2.get("a") == 1.0          # reset
        assert et2.get("b") == pytest.approx(0.5)  # decayed
        assert et2.get("c") == pytest.approx(0.5)  # decayed

    def test_step_none_participants_only_decays(self) -> None:
        ids = ("a", "b")
        et = EligibilityTraces.uniform(ids, decay=0.8)
        et2 = et.step(accepted_participants=None)
        assert et2.get("a") == pytest.approx(0.8)
        assert et2.get("b") == pytest.approx(0.8)

    def test_get_unknown_agent_returns_one(self) -> None:
        et = EligibilityTraces(traces={"a": 0.5}, decay=0.8)
        assert et.get("unknown") == 1.0

    def test_frozen_prevents_attribute_mutation(self) -> None:
        et = EligibilityTraces.uniform(("a",))
        with pytest.raises((AttributeError, TypeError)):
            et.decay = 0.5  # type: ignore[misc]

    def test_decay_compounds_over_multiple_steps(self) -> None:
        et = EligibilityTraces.uniform(("a",), decay=0.5)
        et2 = et.step().step()  # two steps without acceptance
        assert et2.get("a") == pytest.approx(0.25)  # 1.0 × 0.5²

    def test_step_does_not_mutate_original(self) -> None:
        et = EligibilityTraces.uniform(("a", "b"), decay=0.5)
        _ = et.step()
        assert et.get("a") == 1.0  # original unchanged

    def test_hashable(self) -> None:
        et = EligibilityTraces.uniform(("a",), decay=0.8)
        h = hash(et)
        assert isinstance(h, int)

    def test_empty_traces_step_is_empty(self) -> None:
        et = EligibilityTraces(traces={}, decay=0.8)
        et2 = et.step(accepted_participants=("a",))
        # "a" not in traces, so step has nothing to reset
        assert et2.traces == {}


# ---------------------------------------------------------------------------
# Integration with authority_update
# ---------------------------------------------------------------------------


class TestEligibilityTracesAuthorityUpdate:
    def _perfect_errors(self, agent_id: str) -> Errors:
        """Zero error (perfect prediction) for a single agent."""
        return Errors(per_agent={agent_id: 0.0}, proposer_id=agent_id)

    def test_none_traces_same_as_uniform(self) -> None:
        """traces=None must be numerically identical to uniform traces."""
        A = Authority(scores={"a": 0.5, "b": 0.5}, baseline=0.5)
        errors = Errors(per_agent={"a": 0.0, "b": 0.3}, proposer_id="a")
        scales = ErrorScales(global_scale=1.0, local_scale=1.0)
        result_none = authority_update(A, errors, scales, traces=None)
        result_uniform = authority_update(
            A, errors, scales, traces=EligibilityTraces.uniform(("a", "b"))
        )
        assert result_none.get("a") == pytest.approx(result_uniform.get("a"))
        assert result_none.get("b") == pytest.approx(result_uniform.get("b"))

    def test_trace_weight_scales_authority_delta(self) -> None:
        """Half trace weight damps the authority delta toward baseline."""
        A = Authority(scores={"a": 0.5}, baseline=0.5)
        errors = self._perfect_errors("a")
        scales = ErrorScales(global_scale=1.0, local_scale=1.0)
        # correctness=1.0, baseline=0.5
        # full trace: delta = 1.0 × 1.0 − 0.5 = +0.5 → authority increases
        # half trace: delta = 0.5 × 1.0 − 0.5 = 0.0 → authority unchanged
        r_full = authority_update(A, errors, scales, traces=EligibilityTraces({"a": 1.0}, 0.8))
        r_half = authority_update(A, errors, scales, traces=EligibilityTraces({"a": 0.5}, 0.8))
        assert r_full.get("a") > r_half.get("a")

    def test_zero_trace_gives_negative_delta(self) -> None:
        """Zero trace weight means delta = 0 − baseline < 0 (authority decreases)."""
        A = Authority(scores={"a": 0.5}, baseline=0.5)
        errors = self._perfect_errors("a")  # perfect correctness
        scales = ErrorScales(global_scale=1.0, local_scale=1.0)
        result = authority_update(A, errors, scales, traces=EligibilityTraces({"a": 0.0}, 0.8))
        # delta = 0.0 × 1.0 − 0.5 = −0.5 → score decreases
        assert result.get("a") < 0.5

    def test_high_trace_amplifies_positive_delta(self) -> None:
        """Agents with higher trace weight earn more authority for the same error."""
        A = Authority(scores={"a": 0.4}, baseline=0.5)
        errors = self._perfect_errors("a")
        scales = ErrorScales(global_scale=1.0, local_scale=1.0)
        r_high = authority_update(A, errors, scales, traces=EligibilityTraces({"a": 1.0}, 0.8))
        r_low = authority_update(A, errors, scales, traces=EligibilityTraces({"a": 0.3}, 0.8))
        assert r_high.get("a") > r_low.get("a")
