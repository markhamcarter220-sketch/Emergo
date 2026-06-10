"""Fixed-point iterator — the Emergo Kernel.

emergo_kernel(initial_state, ...) drives the four atomic operations in sequence
until φ-prediction error converges (plateaus across consecutive windows).

State = (G_t, φ_t, A_t, E_t) is the only mutable state; each iteration produces
a new immutable tuple.  The loop is strictly sequential — no parallelism, no
shared mutable state, no race conditions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from emergo.authority_update import authority_update
from emergo.ce_execution import ce_execute
from emergo.error_computation import error_computation
from emergo.lux import Lux
from emergo.observer import fire_observers
from emergo.phi_update import phi_update
from emergo.proposal import DefaultProposalGenerator, ProposalGenerator
from emergo.types import Authority, CoordinationEvent, EligibilityTraces, Errors, ErrorScales, Graph, PhiMap, State

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from emergo.diagnostics import KernelDiagnostics

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_initial_phi(
    d_latent: int = 8,
    d_features: int = 16,
    d_ce: int = 4,
    seed: int = 42,
) -> PhiMap:
    """Construct a PhiMap with small random weights (near-zero initialization)."""
    rng = np.random.default_rng(seed)
    scale = 0.01
    return PhiMap(
        W_phi=rng.normal(0.0, scale, (d_latent, d_features)),
        b_phi=np.zeros(d_latent),
        W_F=rng.normal(0.0, scale, (d_latent, d_latent + d_ce)),
        b_F=np.zeros(d_latent),
        d_latent=d_latent,
        d_features=d_features,
        d_ce=d_ce,
    )


def make_initial_authority(agent_ids: tuple, baseline: float = 0.5) -> Authority:
    """Construct an Authority with all agents at baseline."""
    return Authority(scores={a: baseline for a in agent_ids}, baseline=baseline)


def _topology_entropy_from_graph(G: Graph) -> float:
    """Compute topology entropy for a given graph."""
    from emergo.diagnostics import topology_entropy

    return topology_entropy(G.adjacency)


def emergo_kernel(
    initial_state: State,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-4,
    lux: Lux | None = None,
    rng: np.random.Generator | None = None,
    phi_update_interval: int = 10,
    collect_diagnostics: bool = False,
    observers: list | None = None,
    proposal_generator: ProposalGenerator | None = None,
    phi_optimizer: str = "sgd",
    phi_lr: float | None = None,
    phi_grad_clip: float = 1.0,
    phi_early_stop_patience: int = 5,
    phi_force_adapt_interval: int = 1000,
    phi_window: int = 200,
    trace_decay: float = 0.0,
) -> tuple[State, str] | tuple[State, str, KernelDiagnostics]:
    """Run the fixed-point loop until convergence or max_iterations.

    Each iteration (Axiom — sequential, atomic):
      1. Sample CE via proposal_generator (default: softmax authority edge-flip)
      2. CE_Execution  → G_{t+1}
      3. ErrorComputation → per-agent errors
      4. AuthorityUpdate → A_{t+1}
      5. PhiUpdate (every phi_update_interval steps) → φ_{t+1}

    Args:
      observers:              Optional list of KernelObserver objects (INV-10: read-only).
      proposal_generator:     CE sampling strategy.  Defaults to DefaultProposalGenerator
                              (softmax authority-weighted edge flip).  Pass any object
                              implementing the ProposalGenerator protocol to customise
                              CE proposals — e.g., SequenceProposalGenerator for replay
                              or WeightedMixGenerator for blended strategies.
      phi_optimizer:          "sgd" (default) or "adam".
      phi_lr:                 Learning rate for phi_update.  None = use module default.
      phi_grad_clip:          Gradient clipping magnitude for phi_update.
      phi_early_stop_patience: Early-stopping patience for phi_update (0 = disabled).
      phi_force_adapt_interval: Every this many iterations, bypass early stopping to
                                prevent stalled φ adaptation (INV-17).  Default 1000.
      phi_window:               Sliding window passed to phi_update: only the most recent
                                phi_window transitions are used for training.  0 = no limit.
                                Default 200 (bounds O(T) training cost).
      trace_decay:              Eligibility trace decay factor ∈ (0, 1).  0.0 (default)
                                disables traces entirely — identical to pre-trace behavior.
                                Recommended: 0.8 for multi-step credit assignment.

    Returns:
      - (final_state, reason) when collect_diagnostics=False (default)
      - (final_state, reason, KernelDiagnostics) when collect_diagnostics=True

    reason ∈ {"Converged", "Max iterations reached"}.
    """
    if lux is None:
        lux = Lux()
    if rng is None:
        rng = np.random.default_rng(seed=0)
    if proposal_generator is None:
        proposal_generator = DefaultProposalGenerator()
    _observers: list = list(observers) if observers else []

    from emergo.phi_update import _LEARNING_RATE

    _phi_lr = phi_lr if phi_lr is not None else _LEARNING_RATE

    G_t, phi_t, A_t, E_history = initial_state

    g_history: list[Graph] = [G_t]
    ce_history: list[CoordinationEvent] = []

    state = initial_state

    # Dual-channel EMA scales for authority normalization.
    # global_scale tracks proposer (phi-prediction) errors;
    # local_scale tracks participant (structural delta) errors.
    err_scales = ErrorScales()

    # Per-agent eligibility traces for multi-step credit assignment.
    # trace_decay=0.0 means traces are disabled (uniform weight = pre-trace behavior).
    eligibility_traces: EligibilityTraces | None = (
        EligibilityTraces.uniform(G_t.agent_ids, decay=trace_decay)
        if trace_decay > 0.0
        else None
    )

    # Phi-loss history for convergence signal 2 (Critical 1.3).
    _phi_losses: list[float] = []

    # Diagnostics setup
    if collect_diagnostics:
        from emergo.diagnostics import IterationRecord, KernelDiagnostics

        diag_records: list[IterationRecord] = []

    for t in range(max_iterations):
        # INV-10: fire observers before CE sampling (read-only snapshot)
        fire_observers(_observers, "on_iteration_start", t, state)

        CE_t = proposal_generator.propose(A_t, G_t, rng)
        if CE_t is None:
            continue  # degenerate graph or exhausted sequence; keep waiting

        # Step 1: CE_Execution — atomic, all-or-nothing
        G_next, success, _ = ce_execute(G_t, CE_t, lux, A_t)
        if not success:
            # CE rejected; state unchanged.  Decay traces even on rejection (TD(λ) convention).
            if eligibility_traces is not None:
                eligibility_traces = eligibility_traces.step()
            fire_observers(_observers, "on_ce_result", t, CE_t, False, None)
            if collect_diagnostics:
                edge_count = int(np.sum(G_t.adjacency > 0))
                diag_records.append(
                    IterationRecord(
                        iteration=t,
                        ce_attempted=True,
                        ce_accepted=False,
                        ce_type=CE_t.event_type,
                        ce_proposer=CE_t.participants[0] if CE_t.participants else None,
                        ce_participants=tuple(CE_t.participants),
                        authority_scores={aid: A_t.get(aid) for aid in G_t.agent_ids},
                        authority_delta={},
                        topology_entropy=_topology_entropy_from_graph(G_t),
                        edge_count=edge_count,
                        error_mean=None,
                        phi_loss=None,
                    )
                )
            continue

        # Step 2: ErrorComputation — pure function over current φ
        errors = error_computation(G_t, G_next, phi_t, CE_t)

        # Step 3: AuthorityUpdate — deterministic from errors, per-channel scales, and traces
        A_next = authority_update(A_t, errors, err_scales, eligibility_traces)
        if eligibility_traces is not None:
            eligibility_traces = eligibility_traces.step(
                accepted_participants=tuple(CE_t.participants)
            )

        # Update per-channel EMA scales after authority_update (prevents look-ahead bias).
        # Multiply by 2 so scale ≈ 2*mean_error: at typical performance,
        # correctness = 1 - err/(2*mean_err) = 0.5 = baseline → neutral delta.
        # Without this, scale → mean_error and everyone always gets correctness=0.
        _scale_factor = 2.0
        global_errs = (
            [errors.per_agent[errors.proposer_id] * _scale_factor]
            if errors.proposer_id and errors.proposer_id in errors.per_agent
            else []
        )
        local_errs = [
            v * _scale_factor for aid, v in errors.per_agent.items() if aid != errors.proposer_id
        ]
        err_scales = err_scales.update(global_errs, local_errs)
        logger.debug(
            "iter %d: err_scales global=%.4f local=%.4f",
            t,
            err_scales.global_scale,
            err_scales.local_scale,
        )

        # Accumulate history for φ fitting
        g_history.append(G_next)
        ce_history.append(CE_t)
        E_next = [*E_history, errors]

        # Step 4: PhiUpdate — joint optimization of φ and F
        phi_loss_value: float | None = None
        force_adapt = (t > 0) and (t % phi_force_adapt_interval == 0)  # INV-17
        if (t % phi_update_interval == 0) and len(g_history) >= 2:
            phi_next, phi_loss_value = phi_update(
                phi_t,
                g_history,
                ce_history,
                E_next,
                lr=_phi_lr,
                grad_clip=phi_grad_clip,
                optimizer=phi_optimizer,
                early_stop_patience=phi_early_stop_patience if not force_adapt else 0,
                force_adapt=force_adapt,
                phi_window=phi_window,
            )
            if phi_loss_value is not None:
                _phi_losses.append(phi_loss_value)
                fire_observers(_observers, "on_phi_updated", t, phi_loss_value)
        else:
            phi_next = phi_t

        # Record diagnostics for accepted CE
        if collect_diagnostics:
            edge_count = int(np.sum(G_next.adjacency > 0))
            authority_delta = {
                aid: A_next.get(aid) - A_t.get(aid)
                for aid in CE_t.participants
                if aid in G_t.agent_ids
            }
            diag_records.append(
                IterationRecord(
                    iteration=t,
                    ce_attempted=True,
                    ce_accepted=True,
                    ce_type=CE_t.event_type,
                    ce_proposer=CE_t.participants[0] if CE_t.participants else None,
                    ce_participants=tuple(CE_t.participants),
                    authority_scores={aid: A_next.get(aid) for aid in G_next.agent_ids},
                    authority_delta=authority_delta,
                    topology_entropy=_topology_entropy_from_graph(G_next),
                    edge_count=edge_count,
                    error_mean=errors.mean_error(),
                    phi_loss=phi_loss_value,
                )
            )

        # Fire observer after CE accepted (INV-10: errors is a value object, not mutable)
        fire_observers(_observers, "on_ce_result", t, CE_t, True, errors)

        state = (G_next, phi_next, A_next, E_next)
        G_t, phi_t, A_t, E_history = state

        if _converged(E_history, threshold=convergence_threshold, phi_losses=_phi_losses):
            fire_observers(_observers, "on_kernel_done", "Converged", state, t + 1)
            if collect_diagnostics:
                return (
                    state,
                    "Converged",
                    KernelDiagnostics(
                        records=diag_records,
                        agent_ids=G_t.agent_ids,
                    ),
                )
            return state, "Converged"

    fire_observers(_observers, "on_kernel_done", "Max iterations reached", state, max_iterations)
    if collect_diagnostics:
        return (
            state,
            "Max iterations reached",
            KernelDiagnostics(
                records=diag_records,
                agent_ids=G_t.agent_ids,
            ),
        )
    return state, "Max iterations reached"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _converged(
    error_history: list[Errors],
    threshold: float,
    window: int = 5,
    phi_losses: list[float] | None = None,
    phi_loss_patience: int = 10,
) -> bool:
    """Return True when either convergence signal fires (Critical 1.3).

    Signal 1 — structural error plateau: mean error across two consecutive
    windows of `window` steps differs by less than `threshold`.

    Signal 2 — φ-loss plateau: the range of the last `phi_loss_patience`
    phi-loss values is below `threshold` (φ has stopped learning).
    """
    # Signal 1: structural error plateau
    if len(error_history) >= window * 2:
        recent = np.mean([e.mean_error() for e in error_history[-window:]])
        prev = np.mean([e.mean_error() for e in error_history[-2 * window : -window]])
        if float(abs(recent - prev)) < threshold:
            return True

    # Signal 2: phi-loss plateau
    if phi_losses and len(phi_losses) >= phi_loss_patience:
        recent_losses = phi_losses[-phi_loss_patience:]
        if (max(recent_losses) - min(recent_losses)) < threshold:
            return True

    return False
