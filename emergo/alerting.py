"""Real-time alerting for Emergo kernel execution.

Provides AlertManager — a KernelObserver that watches for threshold violations
and invariant anomalies during kernel runs, then fires configurable callbacks.

Usage::

    from emergo.alerting import AlertManager, AlertLevel, authority_monopoly_rule

    alerts = []
    mgr = AlertManager(
        rules=[authority_monopoly_rule(threshold=0.75)],
        on_alert=lambda evt: alerts.append(evt),
    )
    final_state, reason = emergo_kernel(initial_state, observers=[mgr])
    for a in alerts:
        print(f"[{a.level.name}] {a.message} at t={a.iteration}")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Callable, Protocol

from emergo.observer import _NoOpMixin
from emergo.types import CoordinationEvent, Errors, State

logger = logging.getLogger("emergo.alerting")


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class AlertEvent:
    rule_name: str
    level: AlertLevel
    message: str
    iteration: int
    details: dict[str, object]  # arbitrary context for the alert


# ---------------------------------------------------------------------------
# AlertRule protocol
# ---------------------------------------------------------------------------


class AlertRule(Protocol):
    """A callable rule: given iteration + state snapshot, maybe return an AlertEvent."""

    name: str
    level: AlertLevel

    def check(
        self,
        iteration: int,
        state: State,
        *,
        last_ce_accepted: bool = False,
        last_errors: Errors | None = None,
        phi_loss: float | None = None,
    ) -> AlertEvent | None:
        """Return an AlertEvent if the rule fires, else None."""
        ...


# ---------------------------------------------------------------------------
# Default callback
# ---------------------------------------------------------------------------


def _default_on_alert(event: AlertEvent) -> None:
    logger.warning(
        "ALERT [%s] %s: %s (iteration=%d)",
        event.level.name,
        event.rule_name,
        event.message,
        event.iteration,
    )


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager(_NoOpMixin):
    """KernelObserver that checks AlertRules each iteration and fires callbacks.

    Args:
        rules:       List of AlertRule instances to check each iteration.
        on_alert:    Callback called with AlertEvent when a rule fires.
                     Default: log at WARNING level.
        cooldown:    Minimum iterations between repeated alerts from the same rule.
                     Default: 10 (prevents alert storms).
    """

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        on_alert: Callable[[AlertEvent], None] | None = None,
        cooldown: int = 10,
    ) -> None:
        self._rules: list[AlertRule] = list(rules) if rules else []
        self._on_alert: Callable[[AlertEvent], None] = (
            on_alert if on_alert is not None else _default_on_alert
        )
        self._cooldown = cooldown
        self._history: list[AlertEvent] = []
        # Maps rule_name -> last iteration it fired
        self._last_fired: dict[str, int] = {}
        # Per-iteration state accumulated from other hooks
        self._last_ce_accepted: bool = False
        self._last_errors: Errors | None = None
        self._phi_loss: float | None = None

    def on_iteration_start(self, t: int, state: State) -> None:
        """Check all rules, fire on_alert for any that return an event."""
        for rule in self._rules:
            last = self._last_fired.get(rule.name)
            if last is not None and (t - last) < self._cooldown:
                continue
            event = rule.check(
                t,
                state,
                last_ce_accepted=self._last_ce_accepted,
                last_errors=self._last_errors,
                phi_loss=self._phi_loss,
            )
            if event is not None:
                self._last_fired[rule.name] = t
                self._history.append(event)
                self._on_alert(event)

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Errors | None,
    ) -> None:
        """Store last_ce_accepted and last_errors for use in on_iteration_start."""
        self._last_ce_accepted = accepted
        self._last_errors = errors

    def on_phi_updated(self, t: int, loss: float) -> None:
        """Store phi_loss."""
        self._phi_loss = loss

    @property
    def alert_history(self) -> list[AlertEvent]:
        """All alerts fired so far (read-only copy)."""
        return list(self._history)

    def reset(self) -> None:
        """Clear alert history and cooldown state."""
        self._history.clear()
        self._last_fired.clear()
        self._last_ce_accepted = False
        self._last_errors = None
        self._phi_loss = None


# ---------------------------------------------------------------------------
# Built-in rule implementations
# ---------------------------------------------------------------------------


@dataclass
class _AuthorityMonopolyRule:
    """Fires when any agent's authority exceeds threshold."""

    threshold: float = 0.75
    level: AlertLevel = AlertLevel.WARNING
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"authority_monopoly(threshold={self.threshold})"

    def check(
        self,
        iteration: int,
        state: State,
        *,
        last_ce_accepted: bool = False,
        last_errors: Errors | None = None,
        phi_loss: float | None = None,
    ) -> AlertEvent | None:
        _, _, A, _ = state
        violators = {
            agent_id: score for agent_id, score in A.scores.items() if score > self.threshold
        }
        if violators:
            worst_agent = max(violators, key=lambda k: violators[k])
            worst_score = violators[worst_agent]
            return AlertEvent(
                rule_name=self.name,
                level=self.level,
                message=(
                    f"Agent '{worst_agent}' authority={worst_score:.3f} "
                    f"exceeds threshold={self.threshold:.3f} (INV-11 hard cap is 0.8)"
                ),
                iteration=iteration,
                details={"violators": dict(violators), "threshold": self.threshold},
            )
        return None


@dataclass
class _ConvergenceStallRule:
    """Fires when no CE has been accepted in the last stall_iterations iterations."""

    stall_iterations: int = 200
    level: AlertLevel = AlertLevel.WARNING
    name: str = field(init=False)
    _acceptance_window: deque[bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"convergence_stall(stall_iterations={self.stall_iterations})"
        self._acceptance_window: deque[bool] = deque(maxlen=self.stall_iterations)

    def check(
        self,
        iteration: int,
        state: State,
        *,
        last_ce_accepted: bool = False,
        last_errors: Errors | None = None,
        phi_loss: float | None = None,
    ) -> AlertEvent | None:
        self._acceptance_window.append(last_ce_accepted)
        if len(self._acceptance_window) < self.stall_iterations:
            return None
        if not any(self._acceptance_window):
            return AlertEvent(
                rule_name=self.name,
                level=self.level,
                message=(
                    f"No CE accepted in the last {self.stall_iterations} iterations "
                    f"(convergence may be stalled)"
                ),
                iteration=iteration,
                details={"stall_iterations": self.stall_iterations},
            )
        return None


@dataclass
class _HighRejectionRateRule:
    """Fires when CE rejection rate over the last window attempts exceeds threshold."""

    window: int = 50
    threshold: float = 0.9
    level: AlertLevel = AlertLevel.WARNING
    name: str = field(init=False)
    _attempt_window: deque[bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"high_rejection_rate(window={self.window},threshold={self.threshold})"
        self._attempt_window: deque[bool] = deque(maxlen=self.window)

    def check(
        self,
        iteration: int,
        state: State,
        *,
        last_ce_accepted: bool = False,
        last_errors: Errors | None = None,
        phi_loss: float | None = None,
    ) -> AlertEvent | None:
        # Record last CE result (accepted=True → not rejected)
        self._attempt_window.append(last_ce_accepted)
        if len(self._attempt_window) < self.window:
            return None
        n_rejected = sum(1 for acc in self._attempt_window if not acc)
        rejection_rate = n_rejected / len(self._attempt_window)
        if rejection_rate > self.threshold:
            return AlertEvent(
                rule_name=self.name,
                level=self.level,
                message=(
                    f"CE rejection rate {rejection_rate:.1%} over last {self.window} "
                    f"attempts exceeds threshold={self.threshold:.1%}"
                ),
                iteration=iteration,
                details={
                    "rejection_rate": rejection_rate,
                    "window": self.window,
                    "threshold": self.threshold,
                },
            )
        return None


@dataclass
class _PhiLossSpikeRule:
    """Fires when phi_loss spikes by more than spike_factor x baseline."""

    spike_factor: float = 10.0
    level: AlertLevel = AlertLevel.CRITICAL
    name: str = field(init=False)
    _baseline: float | None = field(init=False, default=None, repr=False)
    _n_samples: int = field(init=False, default=0, repr=False)
    _running_sum: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.name = f"phi_loss_spike(spike_factor={self.spike_factor})"

    def check(
        self,
        iteration: int,
        state: State,
        *,
        last_ce_accepted: bool = False,
        last_errors: Errors | None = None,
        phi_loss: float | None = None,
    ) -> AlertEvent | None:
        if phi_loss is None:
            return None
        if self._n_samples == 0:
            # First sample — establish baseline, never fire
            self._running_sum = phi_loss
            self._n_samples = 1
            self._baseline = phi_loss
            return None
        # Compute baseline from *previous* samples before adding the current value
        baseline = self._running_sum / self._n_samples
        self._baseline = baseline
        # Now update running stats with the current sample
        self._running_sum += phi_loss
        self._n_samples += 1
        if baseline > 0 and phi_loss > self.spike_factor * baseline:
            return AlertEvent(
                rule_name=self.name,
                level=self.level,
                message=(
                    f"phi_loss={phi_loss:.4f} is {phi_loss / baseline:.1f}x baseline "
                    f"{baseline:.4f} (spike_factor threshold={self.spike_factor}x)"
                ),
                iteration=iteration,
                details={
                    "phi_loss": phi_loss,
                    "baseline": baseline,
                    "spike_factor": self.spike_factor,
                    "ratio": phi_loss / baseline,
                },
            )
        return None


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def authority_monopoly_rule(
    threshold: float = 0.75,
    level: AlertLevel = AlertLevel.WARNING,
) -> AlertRule:
    """Fire when any agent's authority exceeds `threshold`.

    Note: the hard cap is 0.8 (INV-11). This rule fires earlier at 0.75 to warn
    before the hard cap is reached.
    """
    return _AuthorityMonopolyRule(threshold=threshold, level=level)


def convergence_stall_rule(
    stall_iterations: int = 200,
    level: AlertLevel = AlertLevel.WARNING,
) -> AlertRule:
    """Fire when no CE has been accepted in the last `stall_iterations` iterations."""
    return _ConvergenceStallRule(stall_iterations=stall_iterations, level=level)


def high_rejection_rate_rule(
    window: int = 50,
    threshold: float = 0.9,
    level: AlertLevel = AlertLevel.WARNING,
) -> AlertRule:
    """Fire when CE rejection rate over the last `window` attempts exceeds `threshold`."""
    return _HighRejectionRateRule(window=window, threshold=threshold, level=level)


def phi_loss_spike_rule(
    spike_factor: float = 10.0,
    level: AlertLevel = AlertLevel.CRITICAL,
) -> AlertRule:
    """Fire when phi_loss spikes by more than `spike_factor` x baseline."""
    return _PhiLossSpikeRule(spike_factor=spike_factor, level=level)
