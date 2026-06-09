"""Distributed kernel execution — run multiple Emergo kernel loops in parallel.

Uses Python's multiprocessing module for CPU-bound parallelism.  Each worker
runs an independent emergo_kernel() call; results are collected and returned
after all workers finish.

This module is CPU-parallel (not GPU/cluster), suitable for:
  - Population-based hyperparameter search (different lr, phi_update_interval, etc.)
  - Ensemble runs with different RNG seeds (find the best converging trajectory)
  - Parallel goal execution across independent swarms

Usage::

    from emergo.distributed import run_parallel_kernels, KernelConfig

    configs = [
        KernelConfig(initial_state=state, max_iterations=500, seed=i)
        for i in range(8)
    ]
    results = run_parallel_kernels(configs, n_workers=4)
    best = min(results, key=lambda r: r.final_phi_loss or float("inf"))
    print(f"Best run: seed={best.config.seed}, reason={best.reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing
import time

import numpy as np

from emergo.types import State


@dataclass
class KernelConfig:
    """Configuration for a single parallel kernel run."""

    initial_state: State
    max_iterations: int = 1000
    convergence_threshold: float = 1e-4
    phi_update_interval: int = 10
    phi_optimizer: str = "sgd"
    phi_lr: float | None = None
    phi_early_stop_patience: int = 5
    phi_force_adapt_interval: int = 1000
    seed: int = 0
    run_id: str = field(default="")  # optional label; auto-generated from seed if empty

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"run_seed{self.seed}"


@dataclass
class KernelResult:
    """Result from a single parallel kernel run."""

    config: KernelConfig
    final_state: State
    reason: str  # "Converged" | "Max iterations reached"
    n_iterations: int  # actual iterations run (always available)
    wall_time_seconds: float
    final_phi_loss: float | None  # last phi_update loss if available
    error: str | None = None  # set if the run raised an exception

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def converged(self) -> bool:
        return self.reason == "Converged"


def _kernel_worker(config: KernelConfig) -> KernelResult:
    """Run a single kernel; returns KernelResult.  Top-level for pickle compatibility."""
    # Import here to avoid any circular-import issues when the worker is forked
    from emergo.kernel import emergo_kernel  # NOT from emergo package

    start = time.monotonic()
    try:
        rng = np.random.default_rng(config.seed)
        result = emergo_kernel(
            config.initial_state,
            max_iterations=config.max_iterations,
            convergence_threshold=config.convergence_threshold,
            rng=rng,
            phi_update_interval=config.phi_update_interval,
            phi_optimizer=config.phi_optimizer,
            phi_lr=config.phi_lr,
            phi_early_stop_patience=config.phi_early_stop_patience,
            collect_diagnostics=False,
        )
        final_state, reason = result[0], result[1]
        wall_time = time.monotonic() - start

        # Derive n_iterations from error history length
        _, _, _, e_history = final_state
        n_iterations = len(e_history)

        # Try to get the last phi loss from diagnostics if available;
        # otherwise leave as None (we ran without collect_diagnostics).
        final_phi_loss: float | None = None

        return KernelResult(
            config=config,
            final_state=final_state,
            reason=reason,
            n_iterations=n_iterations,
            wall_time_seconds=wall_time,
            final_phi_loss=final_phi_loss,
            error=None,
        )
    except Exception as exc:
        wall_time = time.monotonic() - start
        # Return a failed result rather than propagating the exception so that
        # pool.map() can collect all results even when some workers crash.
        dummy_state = config.initial_state
        return KernelResult(
            config=config,
            final_state=dummy_state,
            reason="Error",
            n_iterations=0,
            wall_time_seconds=wall_time,
            final_phi_loss=None,
            error=str(exc),
        )


def run_parallel_kernels(
    configs: list[KernelConfig],
    n_workers: int | None = None,
) -> list[KernelResult]:
    """Run multiple kernel configurations in parallel.

    Args:
        configs:    List of KernelConfig, each defining one independent run.
        n_workers:  Number of worker processes. None = min(len(configs), cpu_count()).
                    Use n_workers=1 for debugging (runs sequentially without fork).

    Returns:
        List of KernelResult, in the same order as `configs`.
        Results with exceptions have result.success=False and result.error set.
    """
    if not configs:
        return []

    if n_workers is None:
        n_workers = min(len(configs), multiprocessing.cpu_count())

    # Sequential path — no fork overhead, safe for test environments
    if n_workers == 1 or len(configs) == 1:
        return [_kernel_worker(cfg) for cfg in configs]

    # Parallel path via multiprocessing.Pool
    with multiprocessing.Pool(processes=n_workers) as pool:
        results: list[KernelResult] = pool.map(_kernel_worker, configs)
    return results


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def best_converged(results: list[KernelResult]) -> KernelResult | None:
    """Return the converged result with lowest final_phi_loss, or None if none converged."""
    converged = [r for r in results if r.converged and r.success]
    if not converged:
        return None
    # Sort by phi_loss (treat None as infinity so non-null values rank first)
    return min(
        converged, key=lambda r: r.final_phi_loss if r.final_phi_loss is not None else float("inf")
    )


def all_converged(results: list[KernelResult]) -> list[KernelResult]:
    """Return only converged results, sorted by final_phi_loss ascending."""
    converged = [r for r in results if r.converged and r.success]
    return sorted(
        converged,
        key=lambda r: r.final_phi_loss if r.final_phi_loss is not None else float("inf"),
    )


def summarize_results(results: list[KernelResult]) -> dict:
    """Return summary dict: n_total, n_converged, n_failed, best_phi_loss, avg_wall_time."""
    n_total = len(results)
    n_converged = sum(1 for r in results if r.converged and r.success)
    n_failed = sum(1 for r in results if not r.success)
    phi_losses = [r.final_phi_loss for r in results if r.final_phi_loss is not None and r.success]
    best_phi_loss: float | None = min(phi_losses) if phi_losses else None
    avg_wall_time = sum(r.wall_time_seconds for r in results) / n_total if n_total > 0 else 0.0
    return {
        "n_total": n_total,
        "n_converged": n_converged,
        "n_failed": n_failed,
        "best_phi_loss": best_phi_loss,
        "avg_wall_time": avg_wall_time,
    }
