"""Kernel diagnostics — iteration records and failure-mode detectors.

KernelDiagnostics accumulates per-iteration IterationRecord objects produced
by the kernel when collect_diagnostics=True.  The 8 DetectorResult functions
analyse those records to surface known failure modes.

This module must NOT import from kernel.py (circular import).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emergo.types import State

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    iteration: int
    ce_attempted: bool
    ce_accepted: bool
    ce_type: str | None = None
    ce_proposer: str | None = None  # initiating agent (first participant)
    ce_participants: tuple[str, ...] | None = None
    authority_scores: dict[str, float] = field(default_factory=dict)
    authority_delta: dict[str, float] = field(default_factory=dict)  # A_next - A_t
    topology_entropy: float = 0.0
    edge_count: int = 0
    error_mean: float | None = None
    phi_loss: float | None = None


@dataclass
class KernelDiagnostics:
    records: list[IterationRecord]
    agent_ids: tuple[str, ...]

    def authority_matrix(self) -> np.ndarray:
        """(n_records, n_agents) — authority score at each recorded iteration."""
        n = len(self.records)
        m = len(self.agent_ids)
        mat = np.full((n, m), np.nan)
        for i, rec in enumerate(self.records):
            for j, aid in enumerate(self.agent_ids):
                if aid in rec.authority_scores:
                    mat[i, j] = rec.authority_scores[aid]
        return mat

    def ce_acceptance_rates(self, window: int = 50) -> np.ndarray:
        """Rolling CE acceptance rate over a sliding window."""
        n = len(self.records)
        if n == 0:
            return np.array([])
        accepted = np.array([float(r.ce_accepted) for r in self.records])
        rates = []
        for i in range(n):
            start = max(0, i - window + 1)
            rates.append(float(np.mean(accepted[start : i + 1])))
        return np.array(rates)

    def phi_losses(self) -> np.ndarray:
        """Non-None, finite phi_loss values."""
        vals = [
            r.phi_loss for r in self.records if r.phi_loss is not None and np.isfinite(r.phi_loss)
        ]
        return np.array(vals, dtype=float)

    def edge_counts(self) -> np.ndarray:
        return np.array([r.edge_count for r in self.records], dtype=float)

    def error_means(self) -> np.ndarray:
        vals = [r.error_mean for r in self.records if r.error_mean is not None]
        return np.array(vals, dtype=float)


@dataclass
class DetectorResult:
    name: str
    failure_detected: bool
    severity: float  # 0.0 = healthy, 1.0 = critical
    evidence: str
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def topology_entropy(adjacency: np.ndarray) -> float:
    """Entropy of the edge-weight distribution of an adjacency matrix."""
    flat = adjacency.ravel().astype(float)
    total = float(np.sum(flat))
    if total <= 0.0:
        return 0.0
    p = flat / total
    # Only over positive entries to avoid log(0)
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


# ---------------------------------------------------------------------------
# 8 failure detectors
# ---------------------------------------------------------------------------


def detect_authority_collapse(diag: KernelDiagnostics) -> DetectorResult:
    """Detects whether all agents' authority has collapsed toward zero."""
    mat = diag.authority_matrix()
    n = len(mat)
    if n == 0:
        return DetectorResult(
            name="authority_collapse",
            failure_detected=False,
            severity=0.0,
            evidence="No records available.",
        )

    tail_start = max(0, n - max(1, int(0.2 * n)))
    tail = mat[tail_start:]
    # Replace NaN with 0 for analysis
    tail = np.where(np.isnan(tail), 0.0, tail)
    final_mean = float(np.mean(tail))
    final_std = float(np.std(tail))

    failure = final_mean < 0.1 and final_std < 0.05
    severity = float(np.clip(1.0 - final_mean / 0.1, 0.0, 1.0)) if failure else 0.0

    return DetectorResult(
        name="authority_collapse",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Tail mean authority={final_mean:.4f}, std={final_std:.4f} "
            f"(threshold: mean<0.1, std<0.05)"
        ),
        recommendation=(
            (
                "Introduce per-agent credit differentiation so authority does not "
                "monotonically decrease for all participants."
            )
            if failure
            else ""
        ),
    )


def detect_topology_lock_in(
    diag: KernelDiagnostics,
    threshold: float = 0.01,
) -> DetectorResult:
    """Detects whether CE acceptance has stalled (topology frozen)."""
    n = len(diag.records)
    if n < 2:
        return DetectorResult(
            name="topology_lock_in",
            failure_detected=False,
            severity=0.0,
            evidence="Insufficient records.",
        )

    window = min(50, max(1, n // 5))
    rates = diag.ce_acceptance_rates(window=window)

    # Last 5 windows' worth of entries
    tail_len = min(5 * window, len(rates))
    tail_rates = rates[-tail_len:]
    tail_rate = float(np.mean(tail_rates)) if len(tail_rates) > 0 else 0.0

    failure = tail_rate < threshold
    severity = float(np.clip(1.0 - tail_rate / max(threshold, 1e-10), 0.0, 1.0)) if failure else 0.0

    return DetectorResult(
        name="topology_lock_in",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Tail CE acceptance rate={tail_rate:.4f} " f"(threshold={threshold}, window={window})"
        ),
        recommendation=(
            (
                "Lower Lux min_authority or introduce authority injection to "
                "unfreeze topology exploration."
            )
            if failure
            else ""
        ),
    )


def detect_phi_gaming(
    diag: KernelDiagnostics,
    final_state: State,
    n_random: int = 50,
    ratio_threshold: float = 2.0,
) -> DetectorResult:
    """Detects whether proposed CEs are unusually well-predicted by phi (gaming signal)."""
    from emergo.ce_execution import ce_execute
    from emergo.error_computation import error_computation
    from emergo.kernel import make_initial_authority
    from emergo.lux import Lux
    from emergo.types import CoordinationEvent

    G_final, phi_final, _A_final, _ = final_state

    # Gather last 100 accepted-CE error_means
    accepted_errors = [
        r.error_mean
        for r in diag.records
        if r.ce_accepted and r.error_mean is not None and np.isfinite(r.error_mean)
    ][-100:]

    if len(accepted_errors) < 5:
        return DetectorResult(
            name="phi_gaming",
            failure_detected=False,
            severity=0.0,
            evidence="Fewer than 5 accepted CEs with recorded errors; cannot assess.",
        )

    proposed_mean_error = float(np.mean(accepted_errors))

    # Generate random CEs on G_final
    lux_open = Lux(min_authority=0.0)
    A_open = make_initial_authority(G_final.agent_ids, baseline=1.0)
    rng = np.random.default_rng(99)
    agents = list(G_final.agent_ids)
    random_errors: list[float] = []

    attempts = 0
    while len(random_errors) < n_random and attempts < n_random * 20:
        attempts += 1
        if len(agents) < 2:
            break
        i = int(rng.integers(0, len(agents)))
        j = int(rng.integers(0, len(agents) - 1))
        if j >= i:
            j += 1
        from_agent = agents[i]
        to_agent = agents[j]
        ai = G_final.agent_index(from_agent)
        aj = G_final.agent_index(to_agent)
        if G_final.adjacency[ai, aj] == 0.0:
            ce = CoordinationEvent(
                event_type="add_edge",
                participants=(from_agent, to_agent),
                params=frozenset([("weight", 1.0)]),
            )
        else:
            ce = CoordinationEvent(
                event_type="remove_edge",
                participants=(from_agent, to_agent),
                params=frozenset(),
            )
        G_next_rand, ok, _ = ce_execute(G_final, ce, lux_open, A_open)
        if ok:
            err = error_computation(G_final, G_next_rand, phi_final, ce)
            em = err.mean_error()
            if np.isfinite(em):
                random_errors.append(em)

    if len(random_errors) < 5:
        return DetectorResult(
            name="phi_gaming",
            failure_detected=False,
            severity=0.0,
            evidence="Could not generate enough valid random CEs for comparison.",
        )

    random_mean_error = float(np.mean(random_errors))
    ratio = random_mean_error / (proposed_mean_error + 1e-10)
    failure = ratio > ratio_threshold
    severity = (
        float(np.clip((ratio - ratio_threshold) / ratio_threshold, 0.0, 1.0)) if failure else 0.0
    )

    return DetectorResult(
        name="phi_gaming",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Random CE mean error={random_mean_error:.4f}, "
            f"proposed mean error={proposed_mean_error:.4f}, "
            f"ratio={ratio:.2f} (threshold={ratio_threshold})"
        ),
        recommendation=(
            (
                "Agents may be cherry-picking easy-to-predict CEs. "
                "Introduce a diversity penalty or random CE injection."
            )
            if failure
            else ""
        ),
    )


def detect_speculative_cascades(
    diag: KernelDiagnostics,
    window: int = 50,
    sigma_threshold: float = 3.0,
) -> DetectorResult:
    """Detects authority volatility spikes relative to topology change."""
    mat = diag.authority_matrix()
    n = len(mat)
    edge_counts = diag.edge_counts()

    if n < 2 * window:
        return DetectorResult(
            name="speculative_cascades",
            failure_detected=False,
            severity=0.0,
            evidence=f"Fewer than {2 * window} records; skipping.",
        )

    signals_list: list[float] = []
    for i in range(window, n):
        auth_slice = mat[i - window : i]
        auth_slice = np.where(np.isnan(auth_slice), 0.0, auth_slice)
        auth_vol = float(np.std(auth_slice))
        ec_slice = edge_counts[i - window : i]
        edge_range = float(np.max(ec_slice) - np.min(ec_slice)) + 1e-8
        signals_list.append(auth_vol / edge_range)

    signals = np.array(signals_list)
    if len(signals) == 0:
        return DetectorResult(
            name="speculative_cascades",
            failure_detected=False,
            severity=0.0,
            evidence="No window segments computed.",
        )

    sig_mean = float(np.mean(signals))
    sig_std = float(np.std(signals))

    if sig_std < 1e-12:
        return DetectorResult(
            name="speculative_cascades",
            failure_detected=False,
            severity=0.0,
            evidence="Zero variance in cascade signal; system is uniform.",
        )

    max_signal = float(np.max(signals))
    threshold_val = sig_mean + sigma_threshold * sig_std
    failure = max_signal > threshold_val
    severity = (
        float(np.clip((max_signal - threshold_val) / (sig_std + 1e-10), 0.0, 1.0))
        if failure
        else 0.0
    )

    return DetectorResult(
        name="speculative_cascades",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Max cascade signal={max_signal:.4f}, "
            f"mean={sig_mean:.4f}, std={sig_std:.4f}, "
            f"threshold={threshold_val:.4f}"
        ),
        recommendation=(
            (
                "An authority volatility spike was detected relative to topology change. "
                "Check for runaway authority cascades in the CE sampling path."
            )
            if failure
            else ""
        ),
    )


def detect_lux_bottleneck(
    diag_permissive: KernelDiagnostics,
    diag_strict: KernelDiagnostics,
    rate_ratio_threshold: float = 0.5,
) -> DetectorResult:
    """Detects whether strict Lux settings suppress CE acceptance significantly."""
    if len(diag_permissive.records) == 0 or len(diag_strict.records) == 0:
        return DetectorResult(
            name="lux_bottleneck",
            failure_detected=False,
            severity=0.0,
            evidence="One or both diagnostic sets are empty.",
        )

    rate_perm = float(np.mean([float(r.ce_accepted) for r in diag_permissive.records]))
    rate_strict = float(np.mean([float(r.ce_accepted) for r in diag_strict.records]))
    rate_ratio = rate_strict / (rate_perm + 1e-8)

    # Compare phi_losses
    losses_perm = diag_permissive.phi_losses()
    losses_strict = diag_strict.phi_losses()

    loss_ratio = 1.0
    if len(losses_perm) > 0 and len(losses_strict) > 0:
        mean_loss_perm = float(np.mean(losses_perm))
        mean_loss_strict = float(np.mean(losses_strict))
        loss_ratio = mean_loss_strict / (mean_loss_perm + 1e-10)

    failure = rate_ratio < rate_ratio_threshold or loss_ratio > 2.0
    if failure:
        severity = float(
            np.clip(
                max(1.0 - rate_ratio / rate_ratio_threshold, (loss_ratio - 2.0) / 2.0), 0.0, 1.0
            )
        )
    else:
        severity = 0.0

    return DetectorResult(
        name="lux_bottleneck",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"CE rate (permissive)={rate_perm:.4f}, "
            f"CE rate (strict)={rate_strict:.4f}, "
            f"rate_ratio={rate_ratio:.4f} (threshold={rate_ratio_threshold}), "
            f"loss_ratio={loss_ratio:.4f}"
        ),
        recommendation=(
            (
                "Lux min_authority is suppressing topology exploration. "
                "Lower min_authority or use authority normalization to unblock CEs."
            )
            if failure
            else ""
        ),
    )


def detect_clique_formation(
    diag: KernelDiagnostics,
    final_state: State,
    threshold: float = 2.0,
) -> DetectorResult:
    """Detects whether a clique of high-authority agents dominates."""
    n_agents = len(diag.agent_ids)

    if n_agents < 4:
        return DetectorResult(
            name="clique_formation",
            failure_detected=False,
            severity=0.0,
            evidence=f"Only {n_agents} agents; need >= 4 for clique analysis.",
        )

    # Use final authority scores from the last record
    _, _, A_final, _ = final_state
    scores = np.array([A_final.get(aid) for aid in diag.agent_ids], dtype=float)

    q75 = float(np.percentile(scores, 75))
    high_mask = scores >= q75
    low_mask = ~high_mask

    if not np.any(high_mask) or not np.any(low_mask):
        return DetectorResult(
            name="clique_formation",
            failure_detected=False,
            severity=0.0,
            evidence="Could not split agents into high/low authority groups.",
        )

    mean_high = float(np.mean(scores[high_mask]))
    mean_low = float(np.mean(scores[low_mask]))
    clique_index = mean_high / (mean_low + 1e-8)

    failure = clique_index > threshold
    severity = float(np.clip((clique_index - threshold) / threshold, 0.0, 1.0)) if failure else 0.0

    return DetectorResult(
        name="clique_formation",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Top-25% authority mean={mean_high:.4f}, "
            f"bottom-75% mean={mean_low:.4f}, "
            f"clique_index={clique_index:.4f} (threshold={threshold})"
        ),
        recommendation=(
            (
                "A small clique of agents has disproportionate authority. "
                "Introduce authority decay or anti-monopoly constraints."
            )
            if failure
            else ""
        ),
    )


def detect_credit_assignment_ambiguity(diag: KernelDiagnostics) -> DetectorResult:
    """Detects whether all CE participants receive identical credit (no differentiation)."""
    proposer_deltas = []
    participant_deltas = []

    for rec in diag.records:
        if not rec.ce_accepted:
            continue
        if rec.ce_proposer is None or rec.ce_participants is None:
            continue
        if len(rec.authority_delta) == 0:
            continue

        prop = rec.ce_proposer
        prop_delta = rec.authority_delta.get(prop)
        if prop_delta is None:
            continue

        non_prop_deltas = [
            v
            for k, v in rec.authority_delta.items()
            if k != prop and k in (rec.ce_participants or ())
        ]
        if not non_prop_deltas:
            continue

        proposer_deltas.append(prop_delta)
        participant_deltas.extend(non_prop_deltas)

    if not proposer_deltas or not participant_deltas:
        # All CE with single participant or no participants recorded
        # Still check: if all authority_deltas are identical across all participants
        all_deltas_for_check = []
        for rec in diag.records:
            if not rec.ce_accepted:
                continue
            if rec.ce_participants is None or len(rec.ce_participants) < 2:
                continue
            deltas = [
                rec.authority_delta[p] for p in rec.ce_participants if p in rec.authority_delta
            ]
            if len(deltas) >= 2:
                all_deltas_for_check.append(deltas)

        if not all_deltas_for_check:
            return DetectorResult(
                name="credit_assignment_ambiguity",
                failure_detected=False,
                severity=0.0,
                evidence="No multi-participant CEs found to compare credit assignment.",
            )

        # Check if all participants in each CE get same delta
        all_same = all(len({round(d, 10) for d in deltas}) == 1 for deltas in all_deltas_for_check)
        diff = 0.0
        failure = all_same
        evidence = (
            "All multi-participant CEs have identical authority deltas across participants."
            if failure
            else "No ambiguity detected."
        )
    else:
        mean_prop = float(np.mean(proposer_deltas))
        mean_part = float(np.mean(participant_deltas))
        diff = abs(mean_prop - mean_part)
        failure = diff < 1e-9
        evidence = (
            f"Mean proposer delta={mean_prop:.6f}, "
            f"mean participant delta={mean_part:.6f}, "
            f"|diff|={diff:.2e}"
        )

    severity = 1.0 if failure else 0.0

    return DetectorResult(
        name="credit_assignment_ambiguity",
        failure_detected=failure,
        severity=severity,
        evidence=evidence,
        recommendation=(
            (
                "All CE participants receive identical authority changes. "
                "Implement per-agent error differentiation in error_computation "
                "so initiating agents receive differentiated credit."
            )
            if failure
            else ""
        ),
    )


def detect_emergent_conservatism(
    diag: KernelDiagnostics,
    window: int = 100,
    threshold: float = 0.01,
) -> DetectorResult:
    """Detects long-horizon decline in willingness to propose CEs."""
    n = len(diag.records)
    if n < window:
        return DetectorResult(
            name="emergent_conservatism",
            failure_detected=False,
            severity=0.0,
            evidence=f"Only {n} records (need >= {window}); skipping.",
        )

    rates = diag.ce_acceptance_rates(window=window)
    # Last 5 windows
    tail_len = min(5 * window, len(rates))
    tail_rates = rates[-tail_len:]
    tail_rate = float(np.mean(tail_rates)) if len(tail_rates) > 0 else 0.0

    failure = tail_rate < threshold
    severity = float(np.clip(1.0 - tail_rate / max(threshold, 1e-10), 0.0, 1.0)) if failure else 0.0

    return DetectorResult(
        name="emergent_conservatism",
        failure_detected=failure,
        severity=severity,
        evidence=(
            f"Long-horizon CE acceptance rate={tail_rate:.4f} "
            f"(window={window}, threshold={threshold})"
        ),
        recommendation=(
            (
                "Agents have become systematically conservative. "
                "Introduce authority injection, forced exploration, or reduce min_authority."
            )
            if failure
            else ""
        ),
    )


# ---------------------------------------------------------------------------
# Health check runner
# ---------------------------------------------------------------------------


def run_health_check(
    diag: KernelDiagnostics,
    final_state: State,
    diag_strict: KernelDiagnostics | None = None,
) -> list[DetectorResult]:
    """Apply all detectors. Returns results sorted by severity descending."""
    results: list[DetectorResult] = [
        detect_authority_collapse(diag),
        detect_topology_lock_in(diag),
        detect_phi_gaming(diag, final_state),
        detect_speculative_cascades(diag),
        detect_clique_formation(diag, final_state),
        detect_credit_assignment_ambiguity(diag),
        detect_emergent_conservatism(diag),
    ]

    if diag_strict is not None:
        results.append(detect_lux_bottleneck(diag, diag_strict))
    else:
        # Placeholder result when strict diag not provided
        results.append(
            DetectorResult(
                name="lux_bottleneck",
                failure_detected=False,
                severity=0.0,
                evidence="No strict-Lux diagnostic provided for comparison.",
            )
        )

    results.sort(key=lambda r: r.severity, reverse=True)
    return results
