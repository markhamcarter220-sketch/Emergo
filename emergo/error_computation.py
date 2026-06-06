"""Operation 2 — ErrorComputation.

error_computation(G_t, G_next, phi_t, CE_t) → Errors

Proposer (CE_t.participants[0]):
    predicted = F_t(φ_t(G_t), encode(CE_t))   — global topology forecast
    actual    = φ_t(G_{t+1})                   — true post-CE latent state
    error     = ||predicted - actual||_2        — global L2 prediction error

Non-proposing participants:
    error     = ||Δrow_i || Δcol_i||_2         — local structural change L2

This differentiation breaks the authority-collapse failure mode: when φ is
poorly calibrated the global error dominates, so proposers lose and participants
gain, preventing monotonic authority decrease.

Invariant: error_computation is pure and deterministic; it reads phi_t but never
modifies it.
"""
from __future__ import annotations

import numpy as np

from emergo.features import encode_ce
from emergo.types import CoordinationEvent, Errors, Graph, PhiMap


def _agent_local_error(G_t: Graph, G_next: Graph, agent_id: str) -> float:
    """L2 norm of change in agent's local adjacency structure (rows + cols)."""
    idx = G_t.agent_index(agent_id)
    row_delta = G_next.adjacency[idx, :] - G_t.adjacency[idx, :]
    col_delta = G_next.adjacency[:, idx] - G_t.adjacency[:, idx]
    return float(np.linalg.norm(np.concatenate([row_delta, col_delta])))


def error_computation(
    G_t: Graph,
    G_next: Graph,
    phi_t: PhiMap,
    CE_t: CoordinationEvent,
) -> Errors:
    """Compute differentiated per-agent prediction error for one CE execution.

    Proposer gets global φ-prediction error; non-proposing participants get
    local structural change error proportional to their adjacency delta.
    """
    if not CE_t.participants:
        return Errors(per_agent={})

    ce_enc = encode_ce(CE_t, phi_t.d_ce)
    z_t = phi_t.embed(G_t)
    z_predicted = phi_t.transition(z_t, ce_enc)
    z_actual = phi_t.embed(G_next)
    global_error = float(np.linalg.norm(z_predicted - z_actual))

    proposer = CE_t.participants[0]
    per_agent: dict = {}
    for agent_id in CE_t.participants:
        if agent_id == proposer:
            per_agent[agent_id] = global_error
        else:
            per_agent[agent_id] = _agent_local_error(G_t, G_next, agent_id)

    return Errors(per_agent=per_agent)
