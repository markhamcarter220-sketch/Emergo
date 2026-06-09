"""Operation 4 — PhiUpdate.

phi_update(phi_t, g_history, ce_history, all_errors, ...) → (phi_{t+1}, fitting_loss)

Jointly optimize φ and F to minimize the latent-space prediction loss:

    L = (1/T) Σ_{t=0}^{T-1}  (1/2) ||φ(G_{t+1}) - F(φ(G_t), encode(CE_t))||²

where:
    φ(G) = W_phi @ f(G) + b_phi        (linear projection, d_features → d_latent)
    F(z, c) = W_F @ [z; c] + b_F      (linear transition, d_latent+d_ce → d_latent)

Rank Regularization
-------------------
A nuclear-norm-based regularizer is added to the gradient of W_phi on every step.
This actively prevents rank collapse during optimization rather than reverting after:

    L_total = L_pred + lambda * penalty_rank(W_phi, d_latent)

    penalty_rank(W, d) = 1 / (1 + max(0, rank(W) - d//2))   [monitoring scalar]
    gradient contribution: -lambda * U @ Vt                   [nuclear norm proxy]

The gradient pushes W_phi toward higher nuclear norm (larger singular values),
which prevents the eigenvalue collapse that would cause entanglement failure.
λ is configurable via EMERGO_RANK_PENALTY (default 0.1).

Optimizers
----------
  "sgd"  (default) — vanilla gradient descent with clipping.
  "adam" — Adam (Kingma & Ba 2015) with β₁=0.9, β₂=0.999, ε=1e-8.

Early stopping
--------------
When `early_stop_patience > 0`, training stops when loss improvement over
`early_stop_patience` consecutive steps is < `early_stop_delta`.

Gradient monitoring
-------------------
Logs WARNING when any gradient norm exceeds 10x grad_clip.

Returns:
    phi_next     — updated PhiMap (rank-regularized W_phi prevents collapse)
    fitting_loss — final mean loss including rank penalty
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import os

import numpy as np

from emergo.features import encode_ce, extract_graph_features
from emergo.types import CoordinationEvent, Errors, Graph, PhiMap

logger = logging.getLogger(__name__)

_LEARNING_RATE: float = 1e-3
_GRAD_CLIP: float = 1.0
_GRAD_EXPLODE_WARN_FACTOR: float = 10.0
_RANK_LAMBDA: float = float(os.getenv("EMERGO_RANK_PENALTY", "0.1"))


def phi_update(
    phi_t: PhiMap,
    g_history: list[Graph],
    ce_history: list[CoordinationEvent],
    all_errors: list[Errors],
    n_steps: int = 20,
    lr: float = _LEARNING_RATE,
    grad_clip: float = _GRAD_CLIP,
    optimizer: str = "sgd",
    early_stop_patience: int = 5,
    early_stop_delta: float = 1e-6,
    rank_lambda: float = _RANK_LAMBDA,
    force_adapt: bool = False,
) -> tuple[PhiMap, float]:
    """Jointly refine φ and F over the full graph/CE history.

    Args:
        phi_t:                Input PhiMap to refine.
        g_history:            List of Graph snapshots G_0 … G_T.
        ce_history:           List of CEs CE_0 … CE_{T-1}.
        all_errors:           Accumulated per-agent errors (reserved for future use).
        n_steps:              Maximum number of gradient steps.
        lr:                   Learning rate.
        grad_clip:            Gradient clipping magnitude (element-wise).
        optimizer:            "sgd" or "adam".
        early_stop_patience:  Steps with improvement < early_stop_delta before early stop.
                              0 = disabled.
        early_stop_delta:     Minimum loss improvement for early stopping.
        rank_lambda:          Rank-regularization strength (EMERGO_RANK_PENALTY env var).
                              0.0 = disabled.  Default 0.1.
        force_adapt:          If True, bypass early stopping for this call (INV-17).
                              Used by kernel when phi_force_adapt_interval fires.

    Returns:
        (phi_next, fitting_loss) — phi is rank-regularized to prevent entanglement collapse.
    """
    T = len(g_history) - 1
    if T < 1 or len(ce_history) < T:
        return phi_t, float("inf")

    phi_candidate = phi_t.copy()

    features = np.array([extract_graph_features(G, phi_t.d_features) for G in g_history])
    ce_encs = np.array([encode_ce(ce, phi_t.d_ce) for ce in ce_history[:T]])

    if optimizer == "adam":
        state = _make_adam_state(phi_candidate)
    else:
        state = None

    prev_loss: float = float("inf")
    no_improve_count: int = 0

    for step in range(n_steps):
        loss, grads = _loss_and_grads(phi_candidate, features, ce_encs, T)

        # Rank regularization: augment W_phi gradient to prevent rank collapse
        if rank_lambda > 0.0:
            pen, pen_grad = _rank_penalty_and_grad(phi_candidate.W_phi, phi_candidate.d_latent)
            loss = loss + rank_lambda * pen
            grads["W_phi"] = grads["W_phi"] + rank_lambda * pen_grad

        _check_grad_norms(grads, grad_clip, step)

        if optimizer == "adam":
            assert state is not None
            _apply_adam(phi_candidate, grads, state, step + 1, lr, grad_clip)
        else:
            _apply_sgd(phi_candidate, grads, lr, grad_clip)

        if early_stop_patience > 0 and not force_adapt:  # INV-17
            improvement = prev_loss - loss
            if improvement < early_stop_delta:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    logger.debug(
                        "phi_update: early stop at step %d (no improvement for %d steps)",
                        step,
                        early_stop_patience,
                    )
                    break
            else:
                no_improve_count = 0
            prev_loss = loss

    final_loss, _ = _loss_and_grads(phi_candidate, features, ce_encs, T)
    if rank_lambda > 0.0:
        pen, _ = _rank_penalty_and_grad(phi_candidate.W_phi, phi_candidate.d_latent)
        final_loss = final_loss + rank_lambda * pen

    # Safety monitoring — regularization should prevent this; log if it fires
    rank = np.linalg.matrix_rank(phi_candidate.W_phi, tol=1e-6)
    min_rank = max(phi_candidate.d_latent // 2, 1)
    if rank < min_rank:
        logger.warning(
            "phi_update: W_phi rank=%d < %d after %d steps with rank_lambda=%.4f "
            "(consider increasing rank_lambda or lr)",
            rank,
            min_rank,
            n_steps,
            rank_lambda,
        )

    return phi_candidate, final_loss


# ---------------------------------------------------------------------------
# Rank regularization
# ---------------------------------------------------------------------------


def _rank_penalty_and_grad(W: np.ndarray, d_latent: int) -> tuple[float, np.ndarray]:
    """Compute rank penalty scalar and nuclear-norm gradient for W.

    Scalar (discrete rank formula -- monitoring):
        penalty = 1 / (1 + max(0, rank(W) - d//2))

    Gradient (smooth nuclear-norm proxy):
        d(-||W||_nuc)/dW = -U @ Vt   (where W = U S Vt)
        Adding rank_lambda * (-U@Vt) to dW/phi pushes W toward higher nuclear norm,
        which prevents singular values from collapsing to zero (rank preservation).
    """
    rank = np.linalg.matrix_rank(W, tol=1e-6)
    rank_gap = max(0, rank - d_latent // 2)
    penalty = 1.0 / (1.0 + float(rank_gap))

    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    grad = -(U @ Vt)  # gradient of -||W||_nuc w.r.t. W

    return penalty, grad


# ---------------------------------------------------------------------------
# Gradient application
# ---------------------------------------------------------------------------


def _apply_sgd(phi: PhiMap, grads: dict, lr: float, grad_clip: float) -> None:
    phi.W_phi -= lr * np.clip(grads["W_phi"], -grad_clip, grad_clip)
    phi.b_phi -= lr * np.clip(grads["b_phi"], -grad_clip, grad_clip)
    phi.W_F -= lr * np.clip(grads["W_F"], -grad_clip, grad_clip)
    phi.b_F -= lr * np.clip(grads["b_F"], -grad_clip, grad_clip)


def _make_adam_state(phi: PhiMap) -> dict:
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
    m, v = state["m"], state["v"]
    for key in ("W_phi", "b_phi", "W_F", "b_F"):
        g = np.clip(grads[key], -grad_clip, grad_clip)
        m[key] = beta1 * m[key] + (1 - beta1) * g
        v[key] = beta2 * v[key] + (1 - beta2) * g**2
        m_hat = m[key] / (1 - beta1**step)
        v_hat = v[key] / (1 - beta2**step)
        getattr(phi, key)[...] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _phi_params(phi: PhiMap) -> Iterator[tuple[str, np.ndarray]]:
    yield "W_phi", phi.W_phi
    yield "b_phi", phi.b_phi
    yield "W_F", phi.W_F
    yield "b_F", phi.b_F


# ---------------------------------------------------------------------------
# Gradient monitoring
# ---------------------------------------------------------------------------


def _check_grad_norms(grads: dict, grad_clip: float, step: int) -> None:
    threshold = _GRAD_EXPLODE_WARN_FACTOR * grad_clip
    for key, g in grads.items():
        norm = float(np.linalg.norm(g))
        if norm > threshold:
            logger.warning(
                "phi_update: large gradient at step %d -- %s norm=%.3f > %.1fx clip",
                step,
                key,
                norm,
                _GRAD_EXPLODE_WARN_FACTOR,
            )


# ---------------------------------------------------------------------------
# Loss & exact gradients
# ---------------------------------------------------------------------------


def _loss_and_grads(
    phi: PhiMap,
    features: np.ndarray,
    ce_encs: np.ndarray,
    T: int,
) -> tuple[float, dict]:
    """Compute mean prediction loss and exact gradients over T transitions."""
    d = phi.d_latent

    total_loss = 0.0
    dW_phi = np.zeros_like(phi.W_phi)
    db_phi = np.zeros_like(phi.b_phi)
    dW_F = np.zeros_like(phi.W_F)
    db_F = np.zeros_like(phi.b_F)

    for t in range(T):
        f_t = features[t]
        f_next = features[t + 1]
        c_t = ce_encs[t]

        z_t = phi.W_phi @ f_t + phi.b_phi
        z_next = phi.W_phi @ f_next + phi.b_phi
        zc_t = np.concatenate([z_t, c_t])
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
        "W_F": dW_F * inv_T,
        "b_F": db_F * inv_T,
    }
