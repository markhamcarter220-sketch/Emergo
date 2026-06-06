"""Operation 2 — ErrorComputation.

error_computation(G_t, G_next, phi_t, CE_t) → Errors

For each participant in CE_t:
    predicted = F_t(φ_t(G_t), encode(CE_t))   — agent's implicit topology forecast
    actual    = φ_t(G_{t+1})                   — true post-CE latent state
    error_a   = ||predicted - actual||_2        — L2 prediction error

All participants share the same (predicted, actual) pair because F_t is a global
operator: each agent's "implicit model" is the shared transition.  Individual
accountability is preserved by recording the error under each participant's ID,
enabling per-agent authority updates downstream.

Invariant: error_computation is pure and deterministic; it reads phi_t but never
modifies it.
"""
from __future__ import annotations

from emergo.features import encode_ce
from emergo.types import CoordinationEvent, Errors, Graph, PhiMap


def error_computation(
    G_t: Graph,
    G_next: Graph,
    phi_t: PhiMap,
    CE_t: CoordinationEvent,
) -> Errors:
    """Compute per-agent L2 prediction error for one CE execution."""
    ce_enc = encode_ce(CE_t, phi_t.d_ce)

    z_t = phi_t.embed(G_t)
    z_predicted = phi_t.transition(z_t, ce_enc)
    z_actual = phi_t.embed(G_next)

    import numpy as np
    error_magnitude = float(np.linalg.norm(z_predicted - z_actual))

    return Errors(per_agent={agent_id: error_magnitude for agent_id in CE_t.participants})
