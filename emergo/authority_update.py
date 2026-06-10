"""Operation 3 — AuthorityUpdate.

authority_update(A_t, errors, eta, error_scale) → A_{t+1}

For each agent a in errors.per_agent:
    correctness_a = clip(1 - error_a / error_scale, -1, 1)  [∈ [-1,1]]
    Δ_a           = correctness_a - A_t.baseline              [calibrated]
    A_{t+1}(a)    = clamp(A_t(a) + η * Δ_a, 0, 0.8)

error_scale should be a running EMA of historical mean errors (provided by the
kernel). When None, falls back to the batch maximum so that at least one agent
per batch gets correctness=1 (backward-compatible).  Using an EMA denominator
rather than batch-max breaks the authority-collapse failure mode: when all
agents share the same error (single-participant CE), the EMA scale allows
correctness to differ from zero, so authority can actually increase.

Agents not in errors.per_agent are unchanged (they did not participate).

Invariant: A_{t+1}(a) ∈ [0, 0.8] for all a; updates are continuous (no jumps).
The 0.8 ceiling enforces INV-11 (no authority monopolization).
"""

from __future__ import annotations

import numpy as np

from emergo.types import Authority, Errors

_DEFAULT_ETA: float = 0.05


def authority_update(
    A_t: Authority,
    errors: Errors,
    eta: float = _DEFAULT_ETA,
    error_scale: float | None = None,
) -> Authority:
    """Compute A_{t+1} from A_t and per-agent prediction errors.

    Args:
        error_scale: Historical EMA of mean errors passed by the kernel.
                     When provided, correctness ∈ [-1, 1] relative to EMA.
                     When None, falls back to batch-max (backward-compatible).
    """
    A_next = A_t.copy()

    if not errors.per_agent:
        return A_next

    if error_scale is not None:
        _scale = max(float(error_scale), 1e-8)
    else:
        _scale = max(errors.max_error(), 1e-8)

    max_authority = 0.8  # INV-11: hard monopolization cap
    use_ema = error_scale is not None
    for agent_id, error_a in errors.per_agent.items():
        correctness_a = float(np.clip(1.0 - error_a / _scale, -1.0, 1.0))
        if use_ema:
            # EMA mode: correctness=0 means "at historical average" → neutral.
            # No baseline subtraction — the zero-crossing is the neutral point.
            delta_a = correctness_a
        else:
            # Batch-max mode: correctness ∈ [0,1]; baseline centres updates.
            delta_a = correctness_a - A_next.baseline
        new_score = float(np.clip(A_t.get(agent_id) + eta * delta_a, 0.0, max_authority))
        A_next.set(agent_id, new_score)

    return A_next
