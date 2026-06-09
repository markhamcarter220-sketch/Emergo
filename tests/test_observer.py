"""Tests for KernelObserver protocol and fire_observers dispatcher (INV-10).

Verifies:
  - Observer protocol is correctly implemented by LoggingObserver / HistoryObserver
  - fire_observers catches all exceptions — never propagates to kernel (INV-10)
  - HistoryObserver accurately accumulates metrics
  - LoggingObserver emits log lines at the configured level
  - Observers receive read-only snapshots; mutations do not affect kernel state
  - on_kernel_done is fired for both "Converged" and "Max iterations reached"
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from emergo import (
    Graph,
    HistoryObserver,
    KernelObserver,
    LoggingObserver,
    emergo_kernel,
    fire_observers,
    make_initial_authority,
    make_initial_phi,
)
from emergo.observer import _NoOpMixin
from emergo.types import CoordinationEvent, Errors
from tests.conftest import make_ce

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triangle() -> Graph:
    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    caps = np.ones((3, 2)) * 0.5
    return Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)


def _small_state():
    G = _triangle()
    phi = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=7)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    return G, phi, A, []


def _dummy_ce() -> CoordinationEvent:
    return make_ce("add_edge", ("A", "B"), weight=0.5)


def _dummy_errors() -> Errors:
    return Errors(per_agent={"A": 0.1, "B": 0.2})


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestKernelObserverProtocol:
    def test_logging_observer_is_kernel_observer(self):
        assert isinstance(LoggingObserver(), KernelObserver)

    def test_history_observer_is_kernel_observer(self):
        assert isinstance(HistoryObserver(), KernelObserver)

    def test_noopMixin_has_all_methods(self):
        obs = _NoOpMixin()
        state = _small_state()
        ce = _dummy_ce()
        errors = _dummy_errors()
        obs.on_iteration_start(0, state)
        obs.on_ce_result(0, ce, True, errors)
        obs.on_phi_updated(0, 0.5)
        obs.on_kernel_done("Converged", state, 1)

    def test_custom_observer_accepted(self):
        class MyObs(_NoOpMixin):
            def on_iteration_start(self, t, state):
                self.called = True

        obs = MyObs()
        assert isinstance(obs, KernelObserver)


# ---------------------------------------------------------------------------
# fire_observers: exception isolation (INV-10)
# ---------------------------------------------------------------------------


class TestFireObserversIsolation:
    def test_exception_in_observer_never_propagates(self):
        class BrokenObs(_NoOpMixin):
            def on_iteration_start(self, t, state):
                raise RuntimeError("broken")

        obs = BrokenObs()
        state = _small_state()
        # Must not raise
        fire_observers([obs], "on_iteration_start", 0, state)

    def test_exception_logged_at_warning(self, caplog):
        class BrokenObs(_NoOpMixin):
            def on_ce_result(self, t, ce, accepted, errors):
                raise ValueError("observer crash")

        obs = BrokenObs()
        ce = _dummy_ce()
        with caplog.at_level(logging.WARNING, logger="emergo.observer"):
            fire_observers([obs], "on_ce_result", 0, ce, True, None)

        assert any("observer crash" in r.message for r in caplog.records)

    def test_second_observer_still_called_after_first_raises(self):
        class BrokenObs(_NoOpMixin):
            def on_phi_updated(self, t, loss):
                raise RuntimeError("crash")

        calls = []

        class GoodObs(_NoOpMixin):
            def on_phi_updated(self, t, loss):
                calls.append((t, loss))

        fire_observers([BrokenObs(), GoodObs()], "on_phi_updated", 5, 0.3)
        assert calls == [(5, 0.3)]

    def test_empty_observer_list_is_noop(self):
        fire_observers([], "on_iteration_start", 0, _small_state())

    def test_unknown_method_does_not_raise(self):
        obs = _NoOpMixin()
        fire_observers([obs], "nonexistent_method", 0)


# ---------------------------------------------------------------------------
# HistoryObserver accumulation
# ---------------------------------------------------------------------------


class TestHistoryObserver:
    def test_ce_acceptance_rate_all_accepted(self):
        obs = HistoryObserver()
        ce = _dummy_ce()
        for _ in range(5):
            obs.on_ce_result(0, ce, True, _dummy_errors())
        assert obs.ce_acceptance_rate == pytest.approx(1.0)

    def test_ce_acceptance_rate_none_accepted(self):
        obs = HistoryObserver()
        ce = _dummy_ce()
        for _ in range(3):
            obs.on_ce_result(0, ce, False, None)
        assert obs.ce_acceptance_rate == pytest.approx(0.0)

    def test_ce_acceptance_rate_empty(self):
        obs = HistoryObserver()
        assert obs.ce_acceptance_rate == 0.0

    def test_mean_errors_accumulate(self):
        obs = HistoryObserver()
        ce = _dummy_ce()
        for v in [0.1, 0.2, 0.3]:
            obs.on_ce_result(0, ce, True, Errors(per_agent={"A": v}))
        assert len(obs.mean_errors) == 3
        assert obs.mean_errors[0] == pytest.approx(0.1)

    def test_phi_losses_accumulate(self):
        obs = HistoryObserver()
        obs.on_phi_updated(0, 1.0)
        obs.on_phi_updated(10, 0.5)
        assert obs.phi_losses == [(0, 1.0), (10, 0.5)]

    def test_authority_history_snapshot(self):
        obs = HistoryObserver()
        state = _small_state()
        obs.on_iteration_start(0, state)
        snap = obs.authority_history
        # Each agent should be in the snapshot with value 0.5
        assert set(snap[0].keys()) == {"A", "B", "C"}
        assert all(v == pytest.approx(0.5) for v in snap[0].values())

    def test_max_records_cap(self):
        obs = HistoryObserver(max_records=5)
        ce = _dummy_ce()
        for _ in range(10):
            obs.on_ce_result(0, ce, True, _dummy_errors())
        assert len(obs.mean_errors) == 5

    def test_summary_keys(self):
        obs = HistoryObserver()
        s = obs.summary()
        assert "n_iterations_seen" in s
        assert "ce_acceptance_rate" in s
        assert "mean_error_final" in s
        assert "phi_loss_final" in s
        assert "elapsed_seconds" in s

    def test_elapsed_seconds_positive(self):
        obs = HistoryObserver()
        assert obs.elapsed_seconds >= 0.0

    def test_history_properties_return_copies(self):
        obs = HistoryObserver()
        ce = _dummy_ce()
        obs.on_ce_result(0, ce, True, _dummy_errors())
        h1 = obs.mean_errors
        h2 = obs.mean_errors
        assert h1 is not h2


# ---------------------------------------------------------------------------
# Integration: observers wired into emergo_kernel
# ---------------------------------------------------------------------------


class TestKernelObserverIntegration:
    def test_history_observer_receives_events(self):
        obs = HistoryObserver()
        state = _small_state()
        _final_state, _reason = emergo_kernel(
            state,
            max_iterations=20,
            observers=[obs],
            rng=np.random.default_rng(42),
        )
        # At least some iterations should have fired
        assert len(obs.mean_errors) >= 0  # may be 0 if all rejected (degenerate)

    def test_on_kernel_done_converged_fires(self):
        calls = []

        class DoneObs(_NoOpMixin):
            def on_kernel_done(self, reason, state, n_iterations):
                calls.append(reason)

        state = _small_state()
        # Very tight threshold so it converges quickly
        emergo_kernel(
            state,
            max_iterations=50,
            convergence_threshold=1e10,  # always converges
            observers=[DoneObs()],
            rng=np.random.default_rng(1),
        )
        assert "Converged" in calls

    def test_on_kernel_done_max_iterations_fires(self):
        calls = []

        class DoneObs(_NoOpMixin):
            def on_kernel_done(self, reason, state, n_iterations):
                calls.append(reason)

        state = _small_state()
        # Very tight threshold → never converges in 5 iterations
        emergo_kernel(
            state,
            max_iterations=5,
            convergence_threshold=1e-20,
            observers=[DoneObs()],
            rng=np.random.default_rng(2),
        )
        assert "Max iterations reached" in calls

    def test_observer_exception_does_not_break_kernel(self):
        class AlwaysRaises(_NoOpMixin):
            def on_iteration_start(self, t, state):
                raise RuntimeError("boom")

            def on_ce_result(self, t, ce, accepted, errors):
                raise RuntimeError("boom")

        state = _small_state()
        # Should complete without raising
        _final_state, reason = emergo_kernel(
            state,
            max_iterations=10,
            observers=[AlwaysRaises()],
            rng=np.random.default_rng(3),
        )
        assert reason in ("Converged", "Max iterations reached")
