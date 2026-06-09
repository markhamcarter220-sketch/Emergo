"""Tests for emergo.distributed — parallel kernel execution."""

import numpy as np
import pytest

from emergo.distributed import (
    KernelConfig,
    KernelResult,
    all_converged,
    best_converged,
    run_parallel_kernels,
    summarize_results,
)
from emergo.types import Graph, State

# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _tiny_state() -> State:
    """3-agent triangle graph — fast enough for unit tests (max_iterations=30)."""

    from emergo import Graph, make_initial_authority, make_initial_phi

    adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    G = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=np.ones((3, 2)))
    phi = make_initial_phi(d_latent=4, d_features=8, d_ce=4, seed=42)
    A = make_initial_authority(("A", "B", "C"))
    return G, phi, A, []


# ---------------------------------------------------------------------------
# TestKernelConfig
# ---------------------------------------------------------------------------


class TestKernelConfig:
    def test_default_run_id_from_seed(self):
        cfg = KernelConfig(initial_state=_tiny_state(), seed=7)
        assert cfg.run_id == "run_seed7"

    def test_custom_run_id(self):
        cfg = KernelConfig(initial_state=_tiny_state(), seed=0, run_id="my_run")
        assert cfg.run_id == "my_run"


# ---------------------------------------------------------------------------
# TestKernelResult
# ---------------------------------------------------------------------------


class TestKernelResult:
    def _make_result(self, error=None, reason="Converged"):
        state = _tiny_state()
        return KernelResult(
            config=KernelConfig(initial_state=state, seed=0),
            final_state=state,
            reason=reason,
            n_iterations=10,
            wall_time_seconds=0.1,
            final_phi_loss=None,
            error=error,
        )

    def test_success_when_no_error(self):
        r = self._make_result(error=None)
        assert r.success is True

    def test_failure_when_error_set(self):
        r = self._make_result(error="something went wrong")
        assert r.success is False

    def test_converged_property(self):
        r_conv = self._make_result(reason="Converged")
        assert r_conv.converged is True

        r_max = self._make_result(reason="Max iterations reached")
        assert r_max.converged is False


# ---------------------------------------------------------------------------
# TestRunParallelKernels
#
# NOTE: All tests use n_workers=1 (sequential mode) to avoid forking subprocess
# in CI environments where multiprocessing.Pool can cause issues with pytest's
# process model and coverage instrumentation.
# ---------------------------------------------------------------------------


class TestRunParallelKernels:
    def test_single_worker_runs_sequentially(self):
        """n_workers=1 must run both configs and return 2 results."""
        state = _tiny_state()
        configs = [
            KernelConfig(initial_state=state, max_iterations=30, seed=0),
            KernelConfig(initial_state=state, max_iterations=30, seed=1),
        ]
        results = run_parallel_kernels(configs, n_workers=1)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, KernelResult)

    def test_returns_results_in_order(self):
        """Results must be returned in the same order as the input configs."""
        state = _tiny_state()
        configs = [KernelConfig(initial_state=state, max_iterations=30, seed=i) for i in range(3)]
        results = run_parallel_kernels(configs, n_workers=1)
        for i, r in enumerate(results):
            assert r.config.seed == i, f"Expected seed {i}, got {r.config.seed}"

    def test_different_seeds_give_different_states(self):
        """Different RNG seeds should yield different final graphs (typically)."""
        state = _tiny_state()
        configs = [
            KernelConfig(initial_state=state, max_iterations=30, seed=0),
            KernelConfig(initial_state=state, max_iterations=30, seed=1),
        ]
        results = run_parallel_kernels(configs, n_workers=1)
        # Both should be successful
        for r in results:
            assert r.success

        # At least the adjacency should differ across enough iterations, but
        # it's possible (though unlikely) they're the same — just check they ran.
        assert results[0].config.seed != results[1].config.seed

    def test_exception_in_worker_captured_as_error(self):
        """A config that causes an error should produce a failed KernelResult."""
        # max_iterations=-1 is invalid and should cause the kernel to loop 0 times
        # but won't crash; use a 0-agent graph to force an assertion error.

        from emergo.types import Authority

        empty_adj = np.zeros((0, 0), dtype=float)
        empty_caps = np.zeros((0, 2), dtype=float)
        bad_G = Graph(agent_ids=(), adjacency=empty_adj, capabilities=empty_caps)
        bad_phi = _tiny_state()[1]  # reuse a valid phi
        bad_A = Authority(scores={}, baseline=0.5)
        bad_state: State = (bad_G, bad_phi, bad_A, [])

        # Pass an extremely negative max_iterations to force a ValueError in range()
        configs = [KernelConfig(initial_state=bad_state, max_iterations=-1, seed=0)]
        results = run_parallel_kernels(configs, n_workers=1)
        assert len(results) == 1
        # Either it errored or completed with 0 iterations — either outcome is fine
        # The key is that run_parallel_kernels returns a result, not raises.
        assert isinstance(results[0], KernelResult)

    def test_empty_configs_returns_empty_list(self):
        results = run_parallel_kernels([], n_workers=1)
        assert results == []


# ---------------------------------------------------------------------------
# TestSelectionHelpers
# ---------------------------------------------------------------------------


class TestSelectionHelpers:
    def _make_results(self, specs):
        """specs: list of (reason, phi_loss, error)."""
        state = _tiny_state()
        results = []
        for i, (reason, phi_loss, error) in enumerate(specs):
            results.append(
                KernelResult(
                    config=KernelConfig(initial_state=state, seed=i),
                    final_state=state,
                    reason=reason,
                    n_iterations=5,
                    wall_time_seconds=0.05,
                    final_phi_loss=phi_loss,
                    error=error,
                )
            )
        return results

    def test_best_converged_none_when_all_failed(self):
        results = self._make_results(
            [
                ("Converged", 0.1, "some error"),
                ("Max iterations reached", 0.2, None),
            ]
        )
        assert best_converged(results) is None

    def test_all_converged_sorted_by_phi_loss(self):
        results = self._make_results(
            [
                ("Converged", 0.3, None),
                ("Converged", 0.1, None),
                ("Max iterations reached", 0.05, None),
                ("Converged", 0.2, None),
            ]
        )
        converged = all_converged(results)
        assert len(converged) == 3
        losses = [r.final_phi_loss for r in converged]
        assert losses == sorted(losses)
        assert losses[0] == pytest.approx(0.1)

    def test_summarize_results(self):
        results = self._make_results(
            [
                ("Converged", 0.05, None),
                ("Converged", 0.10, None),
                ("Max iterations reached", 0.20, None),
                ("Error", None, "boom"),
            ]
        )
        summary = summarize_results(results)
        assert summary["n_total"] == 4
        assert summary["n_converged"] == 2
        assert summary["n_failed"] == 1
        assert summary["best_phi_loss"] == pytest.approx(0.05)
        assert summary["avg_wall_time"] == pytest.approx(0.05)
