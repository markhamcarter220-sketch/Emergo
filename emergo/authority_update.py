"""Operation 3 — AuthorityUpdate.

authority_update(A_t, errors, eta) → A_{t+1}

For each agent a in errors.per_agent:
    correctness_a = 1 - (error_a / max_possible_error)   [∈ [0,1]]
    Δ_a           = correctness_a - A_t.baseline          [calibrated]
    A_{t+1}(a)    = clamp(A_t(a) + η * Δ_a, 0, 0.8)

max_possible_error is the maximum observed error in this batch, giving relative
correctness scores: the best-predicting agent gets correctness→1, the worst→0.
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
) -> Authority:
    """Compute A_{t+1} from A_t and per-agent prediction errors."""
    A_next = A_t.copy()

    if not errors.per_agent:
        return A_next

    # Normalize errors to [0,1]: agent with 0 error gets correctness=1.
    # Use the batch max as the denominator; guard against all-zero case.
    max_possible_error = max(errors.max_error(), 1e-8)

    for agent_id, error_a in errors.per_agent.items():
        correctness_a = 1.0 - (error_a / max_possible_error)
        delta_a = correctness_a - A_next.baseline
        max_authority = 0.8  # INV-11: hard monopolization cap
        new_score = float(np.clip(A_t.get(agent_id) + eta * delta_a, 0.0, max_authority))
        A_next.set(agent_id, new_score)

    return A_next
