"""Tests for AuthorityUpdate: clamping, calibration, continuous updates."""

import pytest

from emergo.authority_update import authority_update
from emergo.types import Authority, Errors, ErrorScales


def make_errors(**kwargs) -> Errors:
    return Errors(per_agent={k: float(v) for k, v in kwargs.items()})


class TestAuthorityUpdate:
    def test_scores_remain_in_unit_interval(self):
        A = Authority(scores={"A": 0.9, "B": 0.1}, baseline=0.5)
        errors = make_errors(A=0.0, B=10.0)
        A_next = authority_update(A, errors, eta=1.0)
        for v in A_next.scores.values():
            assert 0.0 <= v <= 1.0

    def test_perfect_predictor_gains_authority(self):
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        errors = make_errors(A=0.0, B=1.0)
        A_next = authority_update(A, errors, eta=0.1)
        assert A_next.get("A") > A.get("A"), "perfect predictor should gain authority"

    def test_worst_predictor_loses_authority(self):
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        errors = make_errors(A=0.0, B=1.0)
        A_next = authority_update(A, errors, eta=0.1)
        assert A_next.get("B") < A.get("B"), "worst predictor should lose authority"

    def test_equal_errors_equal_correctness(self):
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        errors = make_errors(A=0.5, B=0.5)
        A_next = authority_update(A, errors, eta=0.1)
        assert abs(A_next.get("A") - A_next.get("B")) < 1e-9

    def test_nonparticipants_unchanged(self):
        A = Authority(scores={"A": 0.5, "B": 0.7, "C": 0.3}, baseline=0.5)
        errors = make_errors(A=0.2)  # only A participated
        A_next = authority_update(A, errors, eta=0.1)
        assert A_next.get("B") == pytest.approx(0.7)
        assert A_next.get("C") == pytest.approx(0.3)

    def test_empty_errors_returns_unchanged_authority(self):
        A = Authority(scores={"A": 0.6}, baseline=0.5)
        A_next = authority_update(A, Errors(per_agent={}), eta=0.1)
        assert A_next.get("A") == pytest.approx(0.6)

    def test_high_eta_still_clamps(self):
        A = Authority(scores={"A": 0.99}, baseline=0.5)
        errors = make_errors(A=0.0)  # perfect predictor, large eta
        A_next = authority_update(A, errors, eta=100.0)
        assert A_next.get("A") <= 1.0

    def test_original_authority_not_mutated(self):
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=0.3)
        _ = authority_update(A, errors)
        assert A.get("A") == pytest.approx(0.5)

    def test_update_is_continuous(self):
        """Authority change should scale smoothly with eta."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=0.0)
        A1 = authority_update(A, errors, eta=0.01)
        A2 = authority_update(A, errors, eta=0.02)
        assert A2.get("A") > A1.get("A")


class TestDualChannelNormalization:
    """Tests for dual-channel ErrorScales normalization."""

    def test_proposer_uses_global_scale(self):
        """Agent matching proposer_id is normalized against global_scale."""
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        # A is proposer with global error = global_scale → correctness=0 → lose
        errors = Errors(per_agent={"A": 4.0, "B": 0.0}, proposer_id="A")
        scales = ErrorScales(global_scale=4.0, local_scale=1.0)
        A_next = authority_update(A, errors, scales, eta=0.1)
        assert A_next.get("A") < 0.5, "proposer at scale should lose authority"
        assert A_next.get("B") > 0.5, "participant with zero local error should gain"

    def test_participant_uses_local_scale(self):
        """Non-proposing agents are normalized against local_scale."""
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        # B is participant; error = 0.5 * local_scale → correctness = 0.5 = neutral
        errors = Errors(per_agent={"A": 0.0, "B": 1.0}, proposer_id="A")
        scales = ErrorScales(global_scale=1.0, local_scale=2.0)
        A_next = authority_update(A, errors, scales, eta=0.1)
        # B: correctness = 1 - 1/2 = 0.5 = baseline → delta = 0 → neutral
        assert A_next.get("B") == pytest.approx(0.5)

    def test_zero_error_max_gain(self):
        """Zero error → correctness=1 → gain by eta*(1-baseline)."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = Errors(per_agent={"A": 0.0}, proposer_id="A")
        scales = ErrorScales(global_scale=5.0)
        A_next = authority_update(A, errors, scales, eta=0.1)
        # delta = 1 - 0.5 = 0.5; new = 0.5 + 0.1*0.5 = 0.55
        assert A_next.get("A") == pytest.approx(0.55)

    def test_error_at_scale_min_correctness(self):
        """Error >= scale → correctness=0 → lose by baseline * eta."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = Errors(per_agent={"A": 10.0}, proposer_id="A")
        scales = ErrorScales(global_scale=2.0)
        A_next = authority_update(A, errors, scales, eta=0.1)
        # correctness=0, delta=-0.5, new=0.5-0.05=0.45
        assert A_next.get("A") == pytest.approx(0.45)

    def test_single_participant_can_gain(self):
        """Single-participant CE: proposer with error < scale can gain authority."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = Errors(per_agent={"A": 0.5}, proposer_id="A")
        scales = ErrorScales(global_scale=4.0)
        # correctness = 1 - 0.5/4 = 0.875; delta = 0.375 > 0 → gain
        A_next = authority_update(A, errors, scales, eta=0.1)
        assert A_next.get("A") > 0.5

    def test_different_channels_give_different_corrections(self):
        """Proposer and participant with same absolute error, different channel scales."""
        A = Authority(scores={"P": 0.5, "Q": 0.5}, baseline=0.5)
        # Both have error=2.0; proposer uses global_scale=4, participant uses local_scale=1
        errors = Errors(per_agent={"P": 2.0, "Q": 2.0}, proposer_id="P")
        scales = ErrorScales(global_scale=4.0, local_scale=1.0)
        A_next = authority_update(A, errors, scales, eta=0.1)
        # P (proposer): correctness = 1 - 0.5 = 0.5 = neutral → no change
        assert A_next.get("P") == pytest.approx(0.5)
        # Q (participant): correctness = 1 - min(2,1) = 0 → lose
        assert A_next.get("Q") < 0.5

    def test_default_scales_backward_compatible(self):
        """Default ErrorScales() with scale=1.0 matches old batch-max for errors in [0,1]."""
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        errors = Errors(per_agent={"A": 0.0, "B": 1.0})
        A_next = authority_update(A, errors, eta=0.1)  # uses default ErrorScales()
        assert A_next.get("A") > 0.5, "zero-error agent gains with default scales"
        assert A_next.get("B") < 0.5, "max-error agent loses with default scales"


class TestErrorScales:
    """Tests for the ErrorScales dataclass."""

    def test_default_construction(self):
        s = ErrorScales()
        assert s.global_scale == pytest.approx(1.0)
        assert s.local_scale == pytest.approx(1.0)
        assert s.alpha == pytest.approx(0.1)

    def test_update_global_only(self):
        s = ErrorScales(global_scale=1.0, local_scale=1.0, alpha=0.1)
        s2 = s.update(global_errors=[3.0], local_errors=[])
        # global: 0.1*3.0 + 0.9*1.0 = 1.2
        assert s2.global_scale == pytest.approx(1.2)
        # local unchanged
        assert s2.local_scale == pytest.approx(1.0)

    def test_update_local_only(self):
        s = ErrorScales(global_scale=1.0, local_scale=1.0, alpha=0.1)
        s2 = s.update(global_errors=[], local_errors=[0.5])
        assert s2.global_scale == pytest.approx(1.0)
        # local: 0.1*0.5 + 0.9*1.0 = 0.95
        assert s2.local_scale == pytest.approx(0.95)

    def test_update_both_channels(self):
        s = ErrorScales(global_scale=1.0, local_scale=1.0, alpha=0.2)
        s2 = s.update(global_errors=[5.0], local_errors=[0.1])
        assert s2.global_scale == pytest.approx(0.2 * 5.0 + 0.8 * 1.0)
        assert s2.local_scale == pytest.approx(0.2 * 0.1 + 0.8 * 1.0)

    def test_update_enforces_floor(self):
        s = ErrorScales(global_scale=1.0, local_scale=1.0, alpha=1.0)
        s2 = s.update(global_errors=[0.0], local_errors=[0.0])
        assert s2.global_scale >= 1e-6
        assert s2.local_scale >= 1e-6

    def test_frozen_immutable(self):
        s = ErrorScales()
        with pytest.raises((AttributeError, TypeError)):
            s.global_scale = 2.0  # type: ignore[misc]

    def test_update_returns_new_instance(self):
        s = ErrorScales()
        s2 = s.update([1.0], [0.5])
        assert s is not s2
        assert s.global_scale == pytest.approx(1.0)  # original unchanged
