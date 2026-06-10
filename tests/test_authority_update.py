"""Tests for AuthorityUpdate: clamping, calibration, continuous updates."""

import pytest

from emergo.authority_update import authority_update
from emergo.types import Authority, Errors


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


class TestEMANormalization:
    """Tests for EMA error_scale parameter (Critical 1.1)."""

    def test_error_scale_neutral_at_mean(self):
        """When error equals error_scale, correctness=0 → no authority change."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=2.0)
        A_next = authority_update(A, errors, eta=0.1, error_scale=2.0)
        assert A_next.get("A") == pytest.approx(0.5)

    def test_error_below_scale_gains_authority(self):
        """Error < scale → correctness > 0 → authority increases."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=1.0)
        A_next = authority_update(A, errors, eta=0.1, error_scale=4.0)
        assert A_next.get("A") > 0.5

    def test_error_above_scale_loses_authority(self):
        """Error > scale → correctness < 0 → authority decreases."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=6.0)
        A_next = authority_update(A, errors, eta=0.1, error_scale=2.0)
        assert A_next.get("A") < 0.5

    def test_single_participant_can_gain_with_ema(self):
        """Single-participant CE: with EMA scale, authority can increase."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=0.5)
        # error=0.5 < scale=2.0 → correctness=+0.75 → gain
        A_next = authority_update(A, errors, eta=0.1, error_scale=2.0)
        assert A_next.get("A") > 0.5

    def test_none_scale_falls_back_to_batch_max(self):
        """error_scale=None uses batch-max (backward-compatible)."""
        A = Authority(scores={"A": 0.5, "B": 0.5}, baseline=0.5)
        errors = make_errors(A=0.0, B=1.0)
        A_none = authority_update(A, errors, eta=0.1, error_scale=None)
        A_batch = authority_update(A, errors, eta=0.1)  # default
        assert A_none.get("A") == pytest.approx(A_batch.get("A"))
        assert A_none.get("B") == pytest.approx(A_batch.get("B"))

    def test_ema_correctness_clamped_to_minus_one(self):
        """Error = 3 * scale → raw correctness = -2, clamped to -1."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=30.0)
        A_next = authority_update(A, errors, eta=0.1, error_scale=10.0)
        # max loss: 0.5 - 0.1 = 0.4
        assert A_next.get("A") == pytest.approx(0.4)

    def test_ema_correctness_clamped_to_one(self):
        """Zero error → correctness clamped to 1."""
        A = Authority(scores={"A": 0.5}, baseline=0.5)
        errors = make_errors(A=0.0)
        A_next = authority_update(A, errors, eta=0.1, error_scale=5.0)
        # max gain: 0.5 + 0.1 = 0.6
        assert A_next.get("A") == pytest.approx(0.6)
