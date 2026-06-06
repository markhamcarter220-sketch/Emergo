"""Fixed-point iterator — the Emergo Kernel.

emergo_kernel(initial_state, ...) drives the four atomic operations in sequence
until φ-prediction error converges (plateaus across consecutive windows).

State = (G_t, φ_t, A_t, E_t) is the only mutable state; each iteration produces
a new immutable tuple.  The loop is strictly sequential — no parallelism, no
shared mutable state, no race conditions.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from emergo.authority_update import authority_update
from emergo.ce_execution import ce_execute
from emergo.error_computation import error_computation
from emergo.lux import Lux
from emergo.observer import fire_observers
from emergo.phi_update import phi_update
from emergo.types import Authority, CoordinationEvent, Errors, Graph, PhiMap, State


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


def _topology_entropy_from_G(G: Graph) -> float:
    """Compute topology entropy for a given graph."""
    from emergo.diagnostics import topology_entropy
    return topology_entropy(G.adjacency)


def emergo_kernel(
    initial_state: State,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-4,
    lux: Optional[Lux] = None,
    rng: Optional[np.random.Generator] = None,
    phi_update_interval: int = 10,
    collect_diagnostics: bool = False,
    observers: Optional[List] = None,
) -> Union[Tuple[State, str], Tuple[State, str, "KernelDiagnostics"]]:
    """Run the fixed-point loop until convergence or max_iterations.

    Each iteration (Axiom — sequential, atomic):
      1. Sample CE from authority-weighted distribution
      2. CE_Execution  → G_{t+1}
      3. ErrorComputation → per-agent errors
      4. AuthorityUpdate → A_{t+1}
      5. PhiUpdate (every phi_update_interval steps) → φ_{t+1}

    Args:
      observers: Optional list of KernelObserver objects (INV-10: read-only).
                 Observer exceptions are caught and logged, never propagated.

    Returns:
      - (final_state, reason) when collect_diagnostics=False (default)
      - (final_state, reason, KernelDiagnostics) when collect_diagnostics=True

    reason ∈ {"Converged", "Max iterations reached"}.
    """
    if lux is None:
        lux = Lux()
    if rng is None:
        rng = np.random.default_rng(seed=0)
    _observers: List = list(observers) if observers else []

    G_t, phi_t, A_t, E_history = initial_state

    g_history: List[Graph] = [G_t]
    ce_history: List[CoordinationEvent] = []

    state = initial_state

    # Diagnostics setup
    if collect_diagnostics:
        from emergo.diagnostics import IterationRecord, KernelDiagnostics
        diag_records: List[IterationRecord] = []

    for t in range(max_iterations):
        # INV-10: fire observers before CE sampling (read-only snapshot)
        fire_observers(_observers, "on_iteration_start", t, state)

        CE_t = _sample_next_ce(A_t, G_t, rng)
        if CE_t is None:
            continue  # degenerate graph; keep waiting

        # Step 1: CE_Execution — atomic, all-or-nothing
        G_next, success, _ = ce_execute(G_t, CE_t, lux, A_t)
        if not success:
            # CE rejected; state unchanged
            fire_observers(_observers, "on_ce_result", t, CE_t, False, None)
            if collect_diagnostics:
                edge_count = int(np.sum(G_t.adjacency > 0))
                diag_records.append(IterationRecord(
                    iteration=t,
                    ce_attempted=True,
                    ce_accepted=False,
                    ce_type=CE_t.event_type,
                    ce_proposer=CE_t.participants[0] if CE_t.participants else None,
                    ce_participants=tuple(CE_t.participants),
                    authority_scores={aid: A_t.get(aid) for aid in G_t.agent_ids},
                    authority_delta={},
                    topology_entropy=_topology_entropy_from_G(G_t),
                    edge_count=edge_count,
                    error_mean=None,
                    phi_loss=None,
                ))
            continue

        # Step 2: ErrorComputation — pure function over current φ
        errors = error_computation(G_t, G_next, phi_t, CE_t)

        # Step 3: AuthorityUpdate — deterministic from errors
        A_next = authority_update(A_t, errors)

        # Accumulate history for φ fitting
        g_history.append(G_next)
        ce_history.append(CE_t)
        E_next = E_history + [errors]

        # Step 4: PhiUpdate — joint optimization of φ and F
        phi_loss_value: Optional[float] = None
        if (t % phi_update_interval == 0) and len(g_history) >= 2:
            phi_next, phi_loss_value = phi_update(phi_t, g_history, ce_history, E_next)
            if phi_loss_value is not None:
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
            diag_records.append(IterationRecord(
                iteration=t,
                ce_attempted=True,
                ce_accepted=True,
                ce_type=CE_t.event_type,
                ce_proposer=CE_t.participants[0] if CE_t.participants else None,
                ce_participants=tuple(CE_t.participants),
                authority_scores={aid: A_next.get(aid) for aid in G_next.agent_ids},
                authority_delta=authority_delta,
                topology_entropy=_topology_entropy_from_G(G_next),
                edge_count=edge_count,
                error_mean=errors.mean_error(),
                phi_loss=phi_loss_value,
            ))

        # Fire observer after CE accepted (INV-10: errors is a value object, not mutable)
        fire_observers(_observers, "on_ce_result", t, CE_t, True, errors)

        state = (G_next, phi_next, A_next, E_next)
        G_t, phi_t, A_t, E_history = state

        if _converged(E_history, threshold=convergence_threshold):
            fire_observers(_observers, "on_kernel_done", "Converged", state, t + 1)
            if collect_diagnostics:
                return state, "Converged", KernelDiagnostics(
                    records=diag_records,
                    agent_ids=G_t.agent_ids,
                )
            return state, "Converged"

    fire_observers(_observers, "on_kernel_done", "Max iterations reached", state, max_iterations)
    if collect_diagnostics:
        return state, "Max iterations reached", KernelDiagnostics(
            records=diag_records,
            agent_ids=G_t.agent_ids,
        )
    return state, "Max iterations reached"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_next_ce(
    A_t: Authority,
    G_t: Graph,
    rng: np.random.Generator,
) -> Optional[CoordinationEvent]:
    """Propose a CE biased toward high-authority agents (softmax sampling).

    High authority → more likely to be the initiating agent.
    CE alternates add_edge / remove_edge based on current adjacency state.
    """
    if G_t.n_agents < 2:
        return None

    agents = list(G_t.agent_ids)
    auth_vec = np.array([A_t.get(a) for a in agents])

    # Softmax over authority scores for initiator selection
    shifted = auth_vec - auth_vec.max()
    weights = np.exp(shifted)
    weights /= weights.sum()

    from_idx = int(rng.choice(len(agents), p=weights))
    candidates = [i for i in range(len(agents)) if i != from_idx]
    to_idx = int(rng.choice(candidates))

    from_agent = agents[from_idx]
    to_agent = agents[to_idx]

    i, j = G_t.agent_index(from_agent), G_t.agent_index(to_agent)

    if G_t.adjacency[i, j] == 0.0:
        return CoordinationEvent(
            event_type="add_edge",
            participants=(from_agent, to_agent),
            params=frozenset([("weight", float(auth_vec[from_idx]))]),
        )
    else:
        return CoordinationEvent(
            event_type="remove_edge",
            participants=(from_agent, to_agent),
            params=frozenset(),
        )


def _converged(
    error_history: List[Errors],
    threshold: float,
    window: int = 5,
) -> bool:
    """Return True when mean φ-error has plateaued across two consecutive windows."""
    if len(error_history) < window * 2:
        return False
    recent = np.mean([e.mean_error() for e in error_history[-window:]])
    prev = np.mean([e.mean_error() for e in error_history[-2 * window: -window]])
    return float(abs(recent - prev)) < threshold
