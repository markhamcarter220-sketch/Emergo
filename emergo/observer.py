"""KernelObserver protocol — non-mutating hooks into the kernel fixed-point loop.

INV-10 (Observer Isolation): Observers are strictly read-only.  They receive
snapshots of state data but cannot reach into kernel internals or mutate any
shared object.  Observer exceptions are caught by `fire_observers()` and
logged at WARNING level — they never propagate to the kernel (blast radius: zero).

Two concrete observers are provided:
  LoggingObserver  — writes iteration summaries to Python logging.
  HistoryObserver  — accumulates metrics in-memory for post-hoc analysis.

Usage::

    from emergo.observer import HistoryObserver, LoggingObserver
    from emergo import emergo_kernel

    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        initial_state, max_iterations=200, observers=[obs]
    )
    print(f"Acceptance rate: {obs.ce_acceptance_rate:.0%}")
    print(f"Final mean error: {obs.mean_errors[-1]:.4f}")
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from emergo.types import CoordinationEvent, Errors, State

_obs_logger = logging.getLogger("emergo.observer")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class KernelObserver(Protocol):
    """Read-only hook interface for kernel loop callbacks.

    All methods receive copies / snapshots — mutating the arguments does not
    affect kernel state.  Implementations may be stateful (accumulate history)
    but must never reach back into the kernel or modify Lux state.
    """

    def on_iteration_start(self, t: int, state: State) -> None:
        """Called at the start of each kernel iteration, before CE sampling."""

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Optional[Errors],
    ) -> None:
        """Called after CE execution.  errors is None for rejected CEs."""

    def on_phi_updated(self, t: int, loss: float) -> None:
        """Called when phi_update runs (every phi_update_interval steps)."""

    def on_kernel_done(self, reason: str, state: State, n_iterations: int) -> None:
        """Called once when the kernel exits (converged or max_iterations)."""


# ---------------------------------------------------------------------------
# Mixin with default no-op implementations
# ---------------------------------------------------------------------------

class _NoOpMixin:
    """Default no-op implementations — subclass and override what you need."""

    def on_iteration_start(self, t: int, state: State) -> None:  # noqa: D401
        pass

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Optional[Errors],
    ) -> None:
        pass

    def on_phi_updated(self, t: int, loss: float) -> None:
        pass

    def on_kernel_done(self, reason: str, state: State, n_iterations: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Concrete observers
# ---------------------------------------------------------------------------

class LoggingObserver(_NoOpMixin):
    """Writes periodic iteration summaries to Python logging.

    Args:
        log_every: Emit a log line every this many iterations.
        level:     Python logging level (default: INFO).
        logger_name: Logger name (default: "emergo.kernel").
    """

    def __init__(
        self,
        log_every: int = 10,
        level: int = logging.INFO,
        logger_name: str = "emergo.kernel",
    ) -> None:
        self._log_every = log_every
        self._level = level
        self._logger = logging.getLogger(logger_name)
        self._n_accepted = 0
        self._n_total = 0

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Optional[Errors],
    ) -> None:
        self._n_total += 1
        if accepted:
            self._n_accepted += 1
        if t % self._log_every == 0:
            rate = 100 * self._n_accepted / max(1, self._n_total)
            err_str = f" err={errors.mean_error():.4f}" if errors else ""
            self._logger.log(
                self._level,
                "t=%d ce=%s accepted=%s accept_rate=%.0f%%%s",
                t, ce.event_type, accepted, rate, err_str,
            )

    def on_kernel_done(self, reason: str, state: State, n_iterations: int) -> None:
        self._logger.log(
            self._level,
            "Kernel done: reason=%r n_iter=%d accepted=%d/%d",
            reason, n_iterations, self._n_accepted, self._n_total,
        )


class HistoryObserver(_NoOpMixin):
    """Accumulates per-iteration metrics for post-hoc analysis.

    All data is stored in plain Python lists; the observer never holds
    references to mutable kernel objects (INV-10).

    Args:
        max_records: Cap on stored records (older records are dropped).
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._max = max_records
        self._authority_history: List[Dict[str, float]] = []
        self._ce_history: List[Tuple[str, bool]] = []   # (event_type, accepted)
        self._error_history: List[float] = []
        self._phi_losses: List[Tuple[int, float]] = []
        self._edge_counts: List[int] = []
        self._start = time.monotonic()

    # ----- hooks -----

    def on_iteration_start(self, t: int, state: State) -> None:
        if len(self._authority_history) >= self._max:
            return
        G, phi, A, E = state
        # Shallow copy of scores dict — no live reference retained
        self._authority_history.append(dict(A.scores))
        self._edge_counts.append(int((G.adjacency > 0).sum()))

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Optional[Errors],
    ) -> None:
        if len(self._ce_history) >= self._max:
            return
        self._ce_history.append((ce.event_type, accepted))
        if errors is not None:
            self._error_history.append(errors.mean_error())

    def on_phi_updated(self, t: int, loss: float) -> None:
        self._phi_losses.append((t, float(loss)))

    # ----- properties -----

    @property
    def authority_history(self) -> List[Dict[str, float]]:
        """Per-iteration authority score snapshots (list of dicts)."""
        return list(self._authority_history)

    @property
    def edge_count_history(self) -> List[int]:
        """Active edge count per accepted iteration."""
        return list(self._edge_counts)

    @property
    def ce_acceptance_rate(self) -> float:
        """Fraction of attempted CEs that were accepted."""
        accepted = sum(1 for _, ok in self._ce_history if ok)
        return accepted / len(self._ce_history) if self._ce_history else 0.0

    @property
    def mean_errors(self) -> List[float]:
        """Mean prediction error per accepted iteration."""
        return list(self._error_history)

    @property
    def phi_losses(self) -> List[Tuple[int, float]]:
        """(iteration, loss) pairs from each phi_update call."""
        return list(self._phi_losses)

    @property
    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since this observer was created."""
        return time.monotonic() - self._start

    def summary(self) -> Dict[str, object]:
        """Return a plain-dict summary suitable for logging or JSON export."""
        return {
            "n_iterations_seen": len(self._ce_history),
            "ce_acceptance_rate": self.ce_acceptance_rate,
            "mean_error_final": self._error_history[-1] if self._error_history else None,
            "phi_loss_final": self._phi_losses[-1][1] if self._phi_losses else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


# ---------------------------------------------------------------------------
# Dispatcher (called by kernel — INV-10 enforcement)
# ---------------------------------------------------------------------------

def fire_observers(observers: List, method: str, *args, **kwargs) -> None:
    """Call method on each observer, catching all exceptions (INV-10).

    Exceptions are logged at WARNING level and never re-raised.
    The kernel's execution is never interrupted by observer failures.
    """
    if not observers:
        return
    for obs in observers:
        try:
            getattr(obs, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _obs_logger.warning(
                "Observer %s.%s raised (suppressed per INV-10): %s",
                type(obs).__name__, method, exc,
            )
