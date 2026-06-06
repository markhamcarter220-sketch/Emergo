"""Operation 4 — PhiUpdate.

phi_update(phi_t, g_history, ce_history, all_errors, ...) → (phi_{t+1}, fitting_loss)

Jointly optimize φ and F to minimize the latent-space prediction loss:

    L = (1/T) Σ_{t=0}^{T-1}  (1/2) ||φ(G_{t+1}) - F(φ(G_t), encode(CE_t))||²

where:
    φ(G) = W_phi @ f(G) + b_phi        (linear projection, d_features → d_latent)
    F(z, c) = W_F @ [z; c] + b_F      (linear transition, d_latent+d_ce → d_latent)

Gradients are computed analytically (no autograd needed for the linear model).
Gradient clipping prevents blow-up on early poorly-conditioned steps.

Entanglement check (Axiom 5.4.3 guard): if rank(W_phi) < d_latent // 2 after
optimization, the latent map has collapsed to a near-factorized form — reject
the candidate and return the previous phi_t unchanged.

Returns:
    phi_next     — updated PhiMap (= phi_t if entanglement guard fires)
    fitting_loss — final mean loss (used by kernel for convergence detection)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from emergo.features import encode_ce, extract_graph_features
from emergo.types import CoordinationEvent, Errors, Graph, PhiMap

_LEARNING_RATE: float = 1e-3
_GRAD_CLIP: float = 1.0


def phi_update(
    phi_t: PhiMap,
    g_history: List[Graph],
    ce_history: List[CoordinationEvent],
    all_errors: List[Errors],
    n_steps: int = 20,
    lr: float = _LEARNING_RATE,
) -> Tuple[PhiMap, float]:
    """Jointly refine φ and F over the full graph/CE history."""
    T = len(g_history) - 1
    if T < 1 or len(ce_history) < T:
        return phi_t, float("inf")

    phi_candidate = phi_t.copy()

    # Pre-compute raw features once (they don't change during gradient steps)
    features = np.array(
        [extract_graph_features(G, phi_t.d_features) for G in g_history]
    )
    ce_encs = np.array(
        [encode_ce(ce, phi_t.d_ce) for ce in ce_history[:T]]
    )

    for _ in range(n_steps):
        loss, grads = _loss_and_grads(phi_candidate, features, ce_encs, T)

        phi_candidate.W_phi -= lr * np.clip(grads["W_phi"], -_GRAD_CLIP, _GRAD_CLIP)
        phi_candidate.b_phi -= lr * np.clip(grads["b_phi"], -_GRAD_CLIP, _GRAD_CLIP)
        phi_candidate.W_F   -= lr * np.clip(grads["W_F"],   -_GRAD_CLIP, _GRAD_CLIP)
        phi_candidate.b_F   -= lr * np.clip(grads["b_F"],   -_GRAD_CLIP, _GRAD_CLIP)

    final_loss, _ = _loss_and_grads(phi_candidate, features, ce_encs, T)

    # Entanglement guard: reject if W_phi has collapsed to near-degenerate rank.
    # A degenerate W_phi means φ projects all graphs to a low-dimensional subspace,
    # which implies per-agent factorization (some agents stop influencing the embedding).
    rank = np.linalg.matrix_rank(phi_candidate.W_phi, tol=1e-6)
    if rank < max(phi_candidate.d_latent // 2, 1):
        return phi_t, final_loss

    return phi_candidate, final_loss


def _loss_and_grads(
    phi: PhiMap,
    features: np.ndarray,
    ce_encs: np.ndarray,
    T: int,
) -> Tuple[float, dict]:
    """Compute mean prediction loss and exact gradients over T transitions.

    For each t in [0, T):
        z_t     = W_phi @ f_t + b_phi
        z_next  = W_phi @ f_{t+1} + b_phi
        z_pred  = W_F @ [z_t; c_t] + b_F
        res_t   = z_pred - z_next
        loss_t  = 0.5 * ||res_t||²

    Exact gradients:
        ∂L/∂W_F    = (1/T) Σ_t  res_t ⊗ [z_t; c_t]ᵀ
        ∂L/∂b_F    = (1/T) Σ_t  res_t
        ∂L/∂W_phi  = (1/T) Σ_t  (W_F[:,:d]ᵀ @ res_t) ⊗ f_tᵀ  −  res_t ⊗ f_{t+1}ᵀ
        ∂L/∂b_phi  = (1/T) Σ_t  (W_F[:,:d]ᵀ @ res_t)  −  res_t
    """
    d = phi.d_latent

    total_loss = 0.0
    dW_phi = np.zeros_like(phi.W_phi)
    db_phi = np.zeros_like(phi.b_phi)
    dW_F   = np.zeros_like(phi.W_F)
    db_F   = np.zeros_like(phi.b_F)

    for t in range(T):
        f_t    = features[t]
        f_next = features[t + 1]
        c_t    = ce_encs[t]

        z_t    = phi.W_phi @ f_t    + phi.b_phi
        z_next = phi.W_phi @ f_next + phi.b_phi
        zc_t   = np.concatenate([z_t, c_t])
        z_pred = phi.W_F @ zc_t + phi.b_F

        res = z_pred - z_next
        total_loss += 0.5 * float(np.dot(res, res))

        # Gradients w.r.t. F parameters
        dW_F += np.outer(res, zc_t)
        db_F += res

        # Gradient through z_t (input to W_F)
        d_z_t = phi.W_F[:, :d].T @ res
        dW_phi += np.outer(d_z_t, f_t)
        db_phi += d_z_t

        # Gradient through z_next (target embedding, sign flipped)
        dW_phi -= np.outer(res, f_next)
        db_phi -= res

    inv_T = 1.0 / T
    return total_loss * inv_T, {
        "W_phi": dW_phi * inv_T,
        "b_phi": db_phi * inv_T,
        "W_F":   dW_F   * inv_T,
        "b_F":   db_F   * inv_T,
    }
