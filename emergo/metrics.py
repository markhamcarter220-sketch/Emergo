"""Optional metrics / telemetry module for Emergo.

Provides `MetricsObserver`, a `KernelObserver` that accumulates counters and
gauges and can export them as:

  - Plain Python dict (always available)
  - Prometheus text exposition format (no deps beyond stdlib)
  - OpenTelemetry SDK metrics (if ``opentelemetry-sdk`` is installed)

The core Emergo package has no mandatory telemetry dependency.  Import this
module and attach `MetricsObserver` to `emergo_kernel` to enable monitoring.

Usage::

    from emergo.metrics import MetricsObserver
    from emergo import emergo_kernel

    obs = MetricsObserver(namespace="myapp")
    final_state, reason = emergo_kernel(initial_state, observers=[obs])

    # Plain dict export
    print(obs.snapshot())

    # Prometheus text (no extra deps)
    print(obs.prometheus_text())

    # OpenTelemetry (requires opentelemetry-sdk, opentelemetry-exporter-otlp)
    obs.enable_otel(endpoint="http://localhost:4317")
    obs.flush_otel()
"""

from __future__ import annotations

import logging
import time
from typing import Any

from emergo.observer import _NoOpMixin
from emergo.types import CoordinationEvent, Errors, State

logger = logging.getLogger(__name__)


class MetricsObserver(_NoOpMixin):
    """KernelObserver that accumulates Prometheus/OTEL-compatible metrics.

    Counters (monotonically increasing):
        ce_attempts_total         — all CE attempts (accepted + rejected)
        ce_accepted_total         — CEs that passed Lux and mutated the graph
        ce_rejected_total         — CEs rejected by Lux
        phi_updates_total         — times phi_update ran
        kernel_runs_total         — times on_kernel_done was fired
        convergence_total         — runs that terminated with "Converged"

    Gauges (last-seen value):
        authority_mean            — mean authority score across agents (last iteration)
        authority_min             — minimum authority score (last iteration)
        authority_max             — maximum authority score (last iteration)
        mean_error_last           — most recent mean φ-prediction error
        phi_loss_last             — most recent phi_update loss
        edge_count_last           — active edge count (last accepted CE)
        kernel_duration_seconds   — wall time of last kernel run
        ce_acceptance_rate        — accepted / total (rolling)

    Args:
        namespace:  String prefix for all metric names in exported text.
    """

    def __init__(self, namespace: str = "emergo") -> None:
        self._ns = namespace
        self._otel_meter: Any = None
        self._otel_instruments: dict[str, Any] = {}
        self._start: float | None = None

        # Counters
        self._ce_attempts = 0
        self._ce_accepted = 0
        self._ce_rejected = 0
        self._phi_updates = 0
        self._kernel_runs = 0
        self._convergence_count = 0

        # Gauges
        self._authority_mean: float | None = None
        self._authority_min: float | None = None
        self._authority_max: float | None = None
        self._mean_error_last: float | None = None
        self._phi_loss_last: float | None = None
        self._edge_count_last: int | None = None
        self._kernel_duration_seconds: float | None = None

    # ------------------------------------------------------------------
    # KernelObserver hooks
    # ------------------------------------------------------------------

    def on_iteration_start(self, t: int, state: State) -> None:
        if self._start is None:
            self._start = time.monotonic()
        _, _, A, _ = state
        if A.scores:
            vals = [A.get(a) for a in A.scores]
            self._authority_mean = sum(vals) / len(vals)
            self._authority_min = min(vals)
            self._authority_max = max(vals)

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Errors | None,
    ) -> None:
        self._ce_attempts += 1
        if accepted:
            self._ce_accepted += 1
            if errors is not None:
                self._mean_error_last = errors.mean_error()
        else:
            self._ce_rejected += 1

    def on_phi_updated(self, t: int, loss: float) -> None:
        self._phi_updates += 1
        self._phi_loss_last = loss

    def on_kernel_done(self, reason: str, state: State, n_iterations: int) -> None:
        self._kernel_runs += 1
        if reason == "Converged":
            self._convergence_count += 1
        if self._start is not None:
            self._kernel_duration_seconds = time.monotonic() - self._start
            self._start = None
        if self._otel_meter is not None:
            self.flush_otel()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return all metrics as a flat dict."""
        rate = self._ce_accepted / self._ce_attempts if self._ce_attempts > 0 else 0.0
        return {
            f"{self._ns}_ce_attempts_total": self._ce_attempts,
            f"{self._ns}_ce_accepted_total": self._ce_accepted,
            f"{self._ns}_ce_rejected_total": self._ce_rejected,
            f"{self._ns}_phi_updates_total": self._phi_updates,
            f"{self._ns}_kernel_runs_total": self._kernel_runs,
            f"{self._ns}_convergence_total": self._convergence_count,
            f"{self._ns}_authority_mean": self._authority_mean,
            f"{self._ns}_authority_min": self._authority_min,
            f"{self._ns}_authority_max": self._authority_max,
            f"{self._ns}_mean_error_last": self._mean_error_last,
            f"{self._ns}_phi_loss_last": self._phi_loss_last,
            f"{self._ns}_edge_count_last": self._edge_count_last,
            f"{self._ns}_kernel_duration_seconds": self._kernel_duration_seconds,
            f"{self._ns}_ce_acceptance_rate": rate,
        }

    def prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format (no deps).

        Example output::

            # HELP emergo_ce_attempts_total Total CE attempts
            # TYPE emergo_ce_attempts_total counter
            emergo_ce_attempts_total 42
        """
        lines: list[str] = []
        _HELP = {
            "ce_attempts_total": ("counter", "Total CE attempts"),
            "ce_accepted_total": ("counter", "CEs accepted by Lux"),
            "ce_rejected_total": ("counter", "CEs rejected by Lux"),
            "phi_updates_total": ("counter", "Number of phi_update calls"),
            "kernel_runs_total": ("counter", "Kernel run completions"),
            "convergence_total": ("counter", "Runs that converged"),
            "authority_mean": ("gauge", "Mean authority score (last iter)"),
            "authority_min": ("gauge", "Minimum authority score (last iter)"),
            "authority_max": ("gauge", "Maximum authority score (last iter)"),
            "mean_error_last": ("gauge", "Last mean phi-prediction error"),
            "phi_loss_last": ("gauge", "Last phi_update loss"),
            "edge_count_last": ("gauge", "Active edge count (last accepted CE)"),
            "kernel_duration_seconds": ("gauge", "Wall time of last kernel run (s)"),
            "ce_acceptance_rate": ("gauge", "Fraction of CEs accepted"),
        }
        snap = self.snapshot()
        for suffix, (mtype, help_text) in _HELP.items():
            full_name = f"{self._ns}_{suffix}"
            value = snap.get(full_name)
            if value is None:
                continue
            lines.append(f"# HELP {full_name} {help_text}")
            lines.append(f"# TYPE {full_name} {mtype}")
            lines.append(f"{full_name} {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Optional OpenTelemetry export
    # ------------------------------------------------------------------

    def enable_otel(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "emergo",
        *,
        insecure: bool = True,
    ) -> bool:
        """Connect to an OTLP endpoint.

        Requires ``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp-proto-grpc``
        (or -http).  Returns True on success, False if the packages are absent.
        """
        try:
            from opentelemetry import metrics as otel_metrics
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            logger.warning(
                "emergo.metrics: opentelemetry-sdk not installed — OTEL export disabled. "
                "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
            )
            return False

        resource = Resource({"service.name": service_name})
        exporter = OTLPMetricExporter(endpoint=endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(provider)
        self._otel_meter = otel_metrics.get_meter(self._ns)

        # Register instruments once
        self._otel_instruments = {
            "ce_attempts_total": self._otel_meter.create_counter(
                f"{self._ns}.ce_attempts_total", description="Total CE attempts"
            ),
            "ce_accepted_total": self._otel_meter.create_counter(
                f"{self._ns}.ce_accepted_total", description="CEs accepted"
            ),
            "phi_updates_total": self._otel_meter.create_counter(
                f"{self._ns}.phi_updates_total", description="phi_update calls"
            ),
            "convergence_total": self._otel_meter.create_counter(
                f"{self._ns}.convergence_total", description="Convergence events"
            ),
        }
        logger.info("emergo.metrics: OTEL export enabled → %s", endpoint)
        return True

    def flush_otel(self) -> None:
        """Push latest counter deltas to the OTEL pipeline (call after kernel_done)."""
        if self._otel_meter is None:
            return
        snap = self.snapshot()
        instr = self._otel_instruments
        # Counters: add current total (provider tracks monotonicity)
        for key in (
            "ce_attempts_total",
            "ce_accepted_total",
            "phi_updates_total",
            "convergence_total",
        ):
            full = f"{self._ns}_{key}"
            val = snap.get(full, 0) or 0
            if key in instr:
                instr[key].add(int(val))
