"""Operation 3 — AuthorityUpdate.

authority_update(A_t, errors, scales, eta) → A_{t+1}

For each agent a in errors.per_agent:
    channel_scale = scales.global_scale if a == proposer else scales.local_scale
    correctness_a = 1 - clamp(error_a / channel_scale, 0, 1)   [∈ [0, 1]]
    Δ_a           = correctness_a - A_t.baseline
    A_{t+1}(a)    = clamp(A_t(a) + η * Δ_a, 0, 0.8)

Each error channel (global φ-prediction, local structural delta) is
normalized against its own EMA scale, not the batch max. This prevents the
channel with larger absolute magnitude from always winning the relative
comparison and collapsing authority.

Invariant: A_{t+1}(a) ∈ [0, 0.8] for all a (INV-11 monopolization cap).
Invariant: updates are continuous — η controls step size.
"""

from __future__ import annotations

import numpy as np

from emergo.types import Authority, EligibilityTraces, Errors, ErrorScales

_DEFAULT_ETA: float = 0.05
_DEFAULT_SCALES = ErrorScales()  # global_scale=1.0, local_scale=1.0 — backward-compatible default


def authority_update(
    A_t: Authority,
    errors: Errors,
    scales: ErrorScales = _DEFAULT_SCALES,
    traces: EligibilityTraces | None = None,
    eta: float = _DEFAULT_ETA,
) -> Authority:
    """Compute A_{t+1} from A_t, per-agent errors, per-channel EMA scales, and eligibility traces.

    When ``traces`` is None (default), every agent gets uniform weight 1.0 —
    identical to the pre-trace behavior; zero regressions.
    """
    A_next = A_t.copy()

    if not errors.per_agent:
        return A_next

    proposer = errors.proposer_id
    max_authority = 0.8  # INV-11: hard monopolization cap

    for agent_id, error_a in errors.per_agent.items():
        channel_scale = (
            scales.global_scale if agent_id == proposer else scales.local_scale
        )
        # Clamp before inverting so correctness stays in [0, 1].
        correctness_a = 1.0 - float(np.clip(error_a / channel_scale, 0.0, 1.0))
        trace_weight = traces.get(agent_id) if traces is not None else 1.0
        delta_a = correctness_a * trace_weight - A_next.baseline
        new_score = float(np.clip(A_t.get(agent_id) + eta * delta_a, 0.0, max_authority))
        A_next.set(agent_id, new_score)

    return A_next
