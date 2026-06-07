"""Operation 4 — PhiUpdate.

phi_update(phi_t, g_history, ce_history, all_errors, ...) → (phi_{t+1}, fitting_loss)

Jointly optimize φ and F to minimize the latent-space prediction loss:

    L = (1/T) Σ_{t=0}^{T-1}  (1/2) ||φ(G_{t+1}) - F(φ(G_t), encode(CE_t))||²

where:
    φ(G) = W_phi @ f(G) + b_phi        (linear projection, d_features → d_latent)
    F(z, c) = W_F @ [z; c] + b_F      (linear transition, d_latent+d_ce → d_latent)

Optimizers
----------
  "sgd"  (default) — vanilla gradient descent with clipping.
  "adam" — Adam (Kingma & Ba 2015) with β₁=0.9, β₂=0.999, ε=1e-8.
           Carries momentum state within the call (no cross-call state needed
           for the existing periodic-refit pattern).

Early stopping
--------------
When `early_stop_patience > 0`, phi_update monitors the per-step loss
improvement.  If the absolute improvement is < `early_stop_delta` for
`early_stop_patience` consecutive steps, optimization terminates early.
Set `early_stop_patience=0` to disable (run all n_steps regardless).

Gradient monitoring
-------------------
After clipping, the module logs a WARNING when any gradient norm exceeds
10 × grad_clip, which signals a numerically ill-conditioned φ update.

Entanglement guard
------------------
If rank(W_phi) < d_latent // 2 after optimization, the latent map has
collapsed — the candidate is rejected and phi_t is returned unchanged.

Returns:
    phi_next     — updated PhiMap (= phi_t if entanglement guard fires)
    fitting_loss — final mean loss (used by kernel for convergence detection)
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from emergo.features import encode_ce, extract_graph_features
from emergo.types import CoordinationEvent, Errors, Graph, PhiMap

logger = logging.getLogger(__name__)

_LEARNING_RATE: float = 1e-3
_GRAD_CLIP: float = 1.0
_GRAD_EXPLODE_WARN_FACTOR: float = 10.0  # warn when norm > factor × clip


def phi_update(
    phi_t: PhiMap,
    g_history: List[Graph],
    ce_history: List[CoordinationEvent],
    all_errors: List[Errors],
    n_steps: int = 20,
    lr: float = _LEARNING_RATE,
    grad_clip: float = _GRAD_CLIP,
    optimizer: str = "sgd",
    early_stop_patience: int = 5,
    early_stop_delta: float = 1e-6,
) -> Tuple[PhiMap, float]:
    """Jointly refine φ and F over the full graph/CE history.

    Args:
        phi_t:                Input PhiMap to refine.
        g_history:            List of Graph snapshots G_0 … G_T.
        ce_history:           List of CEs CE_0 … CE_{T-1} (parallel to transitions).
        all_errors:           Accumulated per-agent errors (unused by the optimizer
                              itself but preserved for future weighted loss).
        n_steps:              Maximum number of gradient steps.
        lr:                   Learning rate (SGD step size, or Adam α).
        grad_clip:            Gradient clipping magnitude (applied element-wise).
        optimizer:            "sgd" or "adam".
        early_stop_patience:  Stop after this many steps with loss improvement
                              < early_stop_delta.  0 = disabled.
        early_stop_delta:     Minimum loss improvement threshold for early stopping.

    Returns:
        (phi_next, fitting_loss) — phi_t unchanged if entanglement guard fires.
    """
    T = len(g_history) - 1
    if T < 1 or len(ce_history) < T:
        return phi_t, float("inf")

    phi_candidate = phi_t.copy()

    # Pre-compute raw features once — they are constant during optimization
    features = np.array(
        [extract_graph_features(G, phi_t.d_features) for G in g_history]
    )
    ce_encs = np.array(
        [encode_ce(ce, phi_t.d_ce) for ce in ce_history[:T]]
    )

    if optimizer == "adam":
        state = _make_adam_state(phi_candidate)
    else:
        state = None

    prev_loss: float = float("inf")
    no_improve_count: int = 0

    for step in range(n_steps):
        loss, grads = _loss_and_grads(phi_candidate, features, ce_encs, T)

        _check_grad_norms(grads, grad_clip, step)

        if optimizer == "adam":
            _apply_adam(phi_candidate, grads, state, step + 1, lr, grad_clip)
        else:
            _apply_sgd(phi_candidate, grads, lr, grad_clip)

        # Early stopping
        if early_stop_patience > 0:
            improvement = prev_loss - loss
            if improvement < early_stop_delta:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    logger.debug(
                        "phi_update: early stop at step %d — loss=%.6f (no improvement for %d steps)",
                        step, loss, early_stop_patience,
                    )
                    break
            else:
                no_improve_count = 0
            prev_loss = loss

    final_loss, _ = _loss_and_grads(phi_candidate, features, ce_encs, T)

    # Entanglement guard: reject if W_phi has collapsed to near-degenerate rank.
    rank = np.linalg.matrix_rank(phi_candidate.W_phi, tol=1e-6)
    if rank < max(phi_candidate.d_latent // 2, 1):
        logger.warning(
            "phi_update: entanglement guard fired (rank=%d < %d) — reverting to phi_t",
            rank, max(phi_candidate.d_latent // 2, 1),
        )
        return phi_t, final_loss

    return phi_candidate, final_loss


# ---------------------------------------------------------------------------
# Gradient application
# ---------------------------------------------------------------------------

def _apply_sgd(
    phi: PhiMap,
    grads: dict,
    lr: float,
    grad_clip: float,
) -> None:
    """In-place SGD step with element-wise gradient clipping."""
    phi.W_phi -= lr * np.clip(grads["W_phi"], -grad_clip, grad_clip)
    phi.b_phi -= lr * np.clip(grads["b_phi"], -grad_clip, grad_clip)
    phi.W_F   -= lr * np.clip(grads["W_F"],   -grad_clip, grad_clip)
    phi.b_F   -= lr * np.clip(grads["b_F"],   -grad_clip, grad_clip)


def _make_adam_state(phi: PhiMap) -> dict:
    """Initialise zero first- and second-moment buffers for Adam."""
    return {
        "m": {k: np.zeros_like(v) for k, v in _phi_params(phi)},
        "v": {k: np.zeros_like(v) for k, v in _phi_params(phi)},
    }


def _apply_adam(
    phi: PhiMap,
    grads: dict,
    state: dict,
    step: int,
    lr: float,
    grad_clip: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """In-place Adam step with element-wise gradient clipping.

    Clipping is applied before the moment update so that the momentum
    buffers track clipped (numerically safe) gradients only.
    """
    m, v = state["m"], state["v"]
    for key in ("W_phi", "b_phi", "W_F", "b_F"):
        g = np.clip(grads[key], -grad_clip, grad_clip)
        m[key] = beta1 * m[key] + (1 - beta1) * g
        v[key] = beta2 * v[key] + (1 - beta2) * g ** 2
        m_hat = m[key] / (1 - beta1 ** step)
        v_hat = v[key] / (1 - beta2 ** step)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        getattr(phi, key)[...] -= update


def _phi_params(phi: PhiMap):
    yield "W_phi", phi.W_phi
    yield "b_phi", phi.b_phi
    yield "W_F",   phi.W_F
    yield "b_F",   phi.b_F


# ---------------------------------------------------------------------------
# Gradient monitoring
# ---------------------------------------------------------------------------

def _check_grad_norms(grads: dict, grad_clip: float, step: int) -> None:
    """Log WARNING when any gradient norm greatly exceeds the clip threshold."""
    threshold = _GRAD_EXPLODE_WARN_FACTOR * grad_clip
    for key, g in grads.items():
        norm = float(np.linalg.norm(g))
        if norm > threshold:
            logger.warning(
                "phi_update: large gradient at step %d — %s norm=%.3f > %.1f×clip=%.3f "
                "(possible exploding gradient)",
                step, key, norm, _GRAD_EXPLODE_WARN_FACTOR, grad_clip,
            )


# ---------------------------------------------------------------------------
# Loss & exact gradients (unchanged from original)
# ---------------------------------------------------------------------------

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

        dW_F += np.outer(res, zc_t)
        db_F += res

        d_z_t = phi.W_F[:, :d].T @ res
        dW_phi += np.outer(d_z_t, f_t)
        db_phi += d_z_t

        dW_phi -= np.outer(res, f_next)
        db_phi -= res

    inv_T = 1.0 / T
    return total_loss * inv_T, {
        "W_phi": dW_phi * inv_T,
        "b_phi": db_phi * inv_T,
        "W_F":   dW_F   * inv_T,
        "b_F":   db_F   * inv_T,
    }
