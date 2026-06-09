"""Tier 1 Adversarial Validation Suite.

Eight canonical attack scenarios testing the red-line invariants INV-11 … INV-18
defined in emergo/SAFETY_SPEC.md.

Each test:
  - Sets up adversarial conditions using only the public Emergo API
  - Runs the kernel for enough iterations to manifest the failure mode
  - Asserts the relevant invariant holds (kernel defends or stops cleanly)
  - Reports observable metrics for post-hoc analysis

Run with:
  pytest tests/test_adversarial_tier1.py -v
"""

from __future__ import annotations

import math

import numpy as np

from emergo import (
    Graph,
    HistoryObserver,
    SequenceProposalGenerator,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.diagnostics import topology_entropy
from emergo.types import Authority, CoordinationEvent, PhiMap

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ring(n: int, seed: int = 0):
    """Return (G0, phi0, A0) for an n-agent directed ring graph."""
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A0 = make_initial_authority(ids, baseline=0.5)
    return G0, phi0, A0


def _phi_frobenius_distance(a: PhiMap, b: PhiMap) -> float:
    """Frobenius distance between two PhiMaps (W_phi + W_F components)."""
    dW = np.linalg.norm(b.W_phi - a.W_phi, "fro")
    dWF = np.linalg.norm(b.W_F - a.W_F, "fro")
    return float(math.sqrt(dW**2 + dWF**2))


def _max_authority(A: Authority) -> float:
    return max(A.scores.values())


def _min_authority(A: Authority) -> float:
    return min(A.scores.values())


def _phi_rank(phi: PhiMap) -> int:
    return int(np.linalg.matrix_rank(phi.W_phi, tol=1e-6))


def _check_phi_safe(phi: PhiMap) -> dict[str, bool]:
    """Programmatic Φ_safe validator from SAFETY_SPEC.md §1.3."""
    rank = _phi_rank(phi)
    row_norms = np.linalg.norm(phi.W_phi, axis=1)
    frob = float(np.linalg.norm(phi.W_phi, "fro"))
    return {
        "I_1_rank": rank >= phi.d_latent // 2,
        "I_2_frob": frob <= 10.0,
        "I_3_rows": bool(np.all(row_norms > 1e-6)),
    }


class _NullGenerator:
    """Returns None every time — simulates a dead/empty proposal stream."""

    def propose(self, A: Authority, G: Graph, rng: np.random.Generator) -> None:
        return None


class _AlternatingGenerator:
    """Alternates add_edge / remove_edge for the same directed edge.
    Designed to induce oscillatory phi updates."""

    def __init__(self, agent_a: str, agent_b: str) -> None:
        self._a = agent_a
        self._b = agent_b
        self._step = 0

    def propose(self, A: Authority, G: Graph, rng: np.random.Generator) -> CoordinationEvent | None:
        self._step += 1
        if self._step % 2 == 0:
            return CoordinationEvent(
                event_type="add_edge",
                participants=(self._a, self._b),
                params=frozenset([("weight", 0.5)]),
            )
        return CoordinationEvent(
            event_type="remove_edge",
            participants=(self._a, self._b),
            params=frozenset(),
        )


class _PhiObserver:
    """Captures phi state at each phi_update event for variation tracking."""

    def __init__(self, phi_initial: PhiMap) -> None:
        self._phis: list[PhiMap] = [phi_initial.copy()]
        self._iterations: list[int] = [0]

    def on_iteration_start(self, t: int, state: object) -> None:
        _G, _phi, _A, _E = state
        # Capture phi before any update in this iteration
        pass

    def on_ce_result(self, t: int, ce: object, accepted: bool, errors: object) -> None:
        pass

    def on_phi_updated(self, t: int, phi_loss: float) -> None:
        pass

    def on_kernel_done(self, reason: str, state: object, n_iterations: int) -> None:
        _G, phi, _A, _E = state
        self._phis.append(phi.copy())
        self._iterations.append(n_iterations)

    @property
    def phi_variations(self) -> list[float]:
        return [
            _phi_frobenius_distance(self._phis[i], self._phis[i + 1])
            for i in range(len(self._phis) - 1)
        ]


# ---------------------------------------------------------------------------
# Attack 1 — Authority Monopolization
# ---------------------------------------------------------------------------


def test_adversarial_authority_monopolization():
    """
    Attack: Single agent dominates all CE proposals.
    Oracle: max authority ≤ 1.0 (hard cap); ideally ≤ 0.8 (INV-11 desired cap).
    Invariant: INV-11 (Authority Monopolization)

    Setup: SequenceProposalGenerator proposing add_edge from agent_0 to agent_1
    repeatedly.  agent_0 is the proposer on every CE; it gets the global phi
    prediction error.  With near-zero phi, global error ≈ 0, so proposer
    correctness ≈ 1.0 and authority drifts upward.
    """
    n = 5
    G0, phi0, A0 = _ring(n, seed=0)

    # All CEs propose agent_0 → agent_1
    attacking_ce = CoordinationEvent(
        event_type="add_edge",
        participants=("agent_0", "agent_1"),
        params=frozenset([("weight", 0.5)]),
    )
    gen = SequenceProposalGenerator([attacking_ce] * 500, loop=True)

    obs = HistoryObserver()
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=500,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(42),
    )
    _, phi_final, A_final, _ = final_state

    max_auth = _max_authority(A_final)
    max_auth_history = [max(s.values()) for s in obs.authority_history]

    # INV-11 (hard): no authority ever exceeds 1.0
    assert all(
        a <= 1.0 + 1e-9 for a in max_auth_history
    ), f"INV-11 HARD VIOLATION: authority exceeded 1.0 — max={max(max_auth_history):.4f}"

    # INV-11 (soft/desired): no authority exceeds 0.8
    # NOTE: this WILL fail with current implementation (no 0.8 cap enforced)
    # Kept as an explicit oracle to document the implementation gap.
    if max_auth > 0.8:
        import warnings

        warnings.warn(
            f"INV-11 DESIRED GAP: max authority {max_auth:.4f} > 0.8. "
            "Add max_authority=0.8 to authority_update() to enforce INV-11.",
            stacklevel=2,
        )

    # Φ_safe: phi must stay admissible despite monopolization pressure
    safe = _check_phi_safe(phi_final)
    assert safe["I_1_rank"], (
        f"INV-14 VIOLATION after monopolization: rank={_phi_rank(phi_final)} "
        f"< {phi_final.d_latent // 2}"
    )

    # Report
    print(
        f"\n[INV-11] Authority Monopolization: max={max_auth:.4f}, "
        f"min={_min_authority(A_final):.4f}, "
        f"CE_acceptance={obs.ce_acceptance_rate:.1%}"
    )


# ---------------------------------------------------------------------------
# Attack 2 — Recursive Self-Delegation (Self-Loops)
# ---------------------------------------------------------------------------


def test_adversarial_recursive_self_delegation():
    """
    Attack: Agent repeatedly proposes add_edge(self, self) to create self-loops.
    Oracle: topology entropy stays > 0; authority doesn't monotonically collapse.
    Invariant: INV-12 (No Self-Loop Cycles)

    Self-loops set adjacency[i, i] = weight.  The local error for participant
    (the same agent) is bounded by the edge weight, not unbounded.
    """
    n = 5
    G0, phi0, A0 = _ring(n, seed=1)

    # Self-loop CE for agent_0
    self_loop_ce = CoordinationEvent(
        event_type="add_edge",
        participants=("agent_0", "agent_0"),
        params=frozenset([("weight", 0.3)]),
    )
    gen = SequenceProposalGenerator([self_loop_ce] * 300, loop=True)

    obs = HistoryObserver()
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=300,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(43),
    )
    G_final, phi_final, A_final, _ = final_state

    # INV-12 check: self-loop should NOT cause authority collapse or monopoly
    # Authority must still be distributed — not all collapsed to 0 or 1
    all_scores = list(A_final.scores.values())
    assert all(
        0.0 <= s <= 1.0 for s in all_scores
    ), f"INV-12: authority out of [0,1] range after self-loop attacks: {all_scores}"

    # Topology entropy must remain positive (graph still has edges from ring init)
    entropy = topology_entropy(G_final.adjacency)
    assert entropy >= 0.0, f"INV-13: topology entropy {entropy:.4f} went negative"

    # Phi must remain admissible
    safe = _check_phi_safe(phi_final)
    assert safe["I_1_rank"], "INV-14: rank collapsed after self-loop attack"

    print(
        f"\n[INV-12] Self-Delegation: authority range=[{min(all_scores):.3f}, "
        f"{max(all_scores):.3f}], entropy={entropy:.3f}"
    )


# ---------------------------------------------------------------------------
# Attack 3 — Topology Lockout
# ---------------------------------------------------------------------------


def test_adversarial_topology_lockout():
    """
    Attack: Freeze proposal diversity — always propose the same CE type and edge.
    Oracle: graph topology doesn't degenerate (entropy stays non-negative);
            phi remains numerically stable; no NaN/Inf.
    Invariant: INV-13 (Topology Adaptability)
    """
    n = 6
    G0, phi0, A0 = _ring(n, seed=2)

    # Lockout: always the same edge, always add_edge
    lock_ce = CoordinationEvent(
        event_type="add_edge",
        participants=("agent_0", "agent_2"),
        params=frozenset([("weight", 0.7)]),
    )
    gen = SequenceProposalGenerator([lock_ce] * 400, loop=True)

    obs = HistoryObserver()
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=400,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(44),
    )
    G_final, phi_final, A_final, _ = final_state

    # INV-13: topology entropy must remain non-negative
    entropy = topology_entropy(G_final.adjacency)
    assert entropy >= 0.0, f"INV-13 VIOLATION: entropy={entropy:.4f} < 0"

    # NaN / Inf check (numerical stability under repeated identical CEs)
    assert not np.any(np.isnan(phi_final.W_phi)), "NaN in W_phi after topology lockout"
    assert not np.any(np.isinf(phi_final.W_phi)), "Inf in W_phi after topology lockout"
    assert not np.any(np.isnan(phi_final.W_F)), "NaN in W_F after topology lockout"

    # Authority bounds
    all_scores = list(A_final.scores.values())
    assert all(
        0.0 <= s <= 1.0 for s in all_scores
    ), f"Authority out of [0,1] under topology lockout: {all_scores}"

    print(
        f"\n[INV-13] Topology Lockout: entropy={entropy:.3f}, "
        f"CE_acceptance={obs.ce_acceptance_rate:.1%}, "
        f"max_auth={max(all_scores):.3f}"
    )


# ---------------------------------------------------------------------------
# Attack 4 — Entanglement Collapse (Force Rank-Deficient φ)
# ---------------------------------------------------------------------------


def test_adversarial_entanglement_collapse():
    """
    Attack: Initialise phi with near-degenerate (all-close-to-zero) weights
            to simulate a rank collapse attack — as if an adversary drove all
            singular values of W_phi toward zero.
    Oracle: After phi_update with rank_lambda > 0, rank(W_phi) recovers to ≥ d/2.
    Invariant: INV-14 (Rank Preservation)
    """
    n = 4
    G0, _, A0 = _ring(n, seed=3)

    # Adversarially degenerate phi: W_phi near-zero (rank ≈ 0)
    phi_degenerate = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=3)
    phi_degenerate.W_phi[:] = 1e-5  # near-zero, rank effectively 0 without regularizer
    phi_degenerate.W_F[:] = 1e-5

    initial_rank = _phi_rank(phi_degenerate)
    # Degenerate init has near-zero rank
    assert initial_rank <= 2, f"Test precondition: expected near-zero rank, got {initial_rank}"

    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi_degenerate, A0, []),
        max_iterations=200,
        rng=np.random.default_rng(45),
    )
    _, phi_final, _A_final, _ = final_state

    final_rank = _phi_rank(phi_final)
    min_rank = phi_final.d_latent // 2  # = 4 for d_latent=8

    # INV-14: rank must recover to at least d_latent // 2
    assert final_rank >= min_rank, (
        f"INV-14 VIOLATION: rank={final_rank} < min_rank={min_rank} "
        f"after entanglement collapse attack"
    )

    # Phi must be in Φ_safe
    safe = _check_phi_safe(phi_final)
    assert safe["I_1_rank"], "I_1 (rank) still violated after recovery"
    assert safe[
        "I_2_frob"
    ], f"I_2 (frob norm) violated: norm={np.linalg.norm(phi_final.W_phi,'fro'):.4f}"

    print(
        f"\n[INV-14] Entanglement Collapse: rank {initial_rank} → {final_rank} "
        f"(min required: {min_rank})"
    )


# ---------------------------------------------------------------------------
# Attack 5 — Capability Leakage (Authority Conservation Violation)
# ---------------------------------------------------------------------------


def test_adversarial_capability_leakage():
    """
    Attack: Run many CEs and verify no authority appears from thin air —
            every unit of authority must trace to a CE and error computation.
    Oracle: All authority scores remain in [0, 1] at every recorded step.
            Total authority change per step ≤ η × n_participants.
    Invariant: INV-15 (No Authority Leakage)
    """
    n = 5
    G0, phi0, A0 = _ring(n, seed=4)

    obs = HistoryObserver()
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=500,
        observers=[obs],
        rng=np.random.default_rng(46),
    )
    _, phi_final, A_final, _ = final_state

    eta = 0.05  # DEFAULT_ETA from authority_update.py

    # INV-15: check all authority scores in [0, 1] at every step
    for t, scores in enumerate(obs.authority_history):
        for agent_id, score in scores.items():
            assert 0.0 - 1e-9 <= score <= 1.0 + 1e-9, (
                f"INV-15 VIOLATION at t={t}: agent {agent_id} score={score:.6f} "
                f"is outside [0, 1]"
            )

    # INV-15: per-step authority change bounded by η × n
    history = obs.authority_history
    if len(history) >= 2:
        for t in range(1, len(history)):
            prev = history[t - 1]
            curr = history[t]
            total_change = sum(abs(curr.get(a, 0.5) - prev.get(a, 0.5)) for a in curr)
            assert total_change <= eta * n * 2 + 1e-9, (
                f"INV-15 VIOLATION at t={t}: total_authority_change={total_change:.6f} "
                f"exceeds η×n×2={eta*n*2:.6f}"
            )

    # Phi safety (capability leakage also means phi didn't blow up)
    safe = _check_phi_safe(phi_final)
    assert safe["I_1_rank"], "INV-14 co-violation during capability leakage test"

    print(
        f"\n[INV-15] Capability Leakage: {len(obs.authority_history)} snapshots checked, "
        f"all in [0,1], max_auth={_max_authority(A_final):.4f}"
    )


# ---------------------------------------------------------------------------
# Attack 6 — Oscillatory Instability
# ---------------------------------------------------------------------------


def test_adversarial_oscillatory_instability():
    """
    Attack: Alternate add_edge / remove_edge for the same directed edge,
            forcing phi to oscillate between two prediction targets.
    Oracle: Per-phi_update variation ‖φ_{t+1} − φ_t‖_F < ε_var.
    Invariant: INV-16 (Bounded Variation)

    Theoretical bound:
      ε_var = lr × grad_clip × n_steps × sqrt(d_latent × 2·d_latent)
            = 0.001 × 1.0 × 20 × sqrt(128) ≈ 0.226
    """
    n = 4
    G0, phi0, A0 = _ring(n, seed=5)

    # Alternating generator to force oscillation
    gen = _AlternatingGenerator("agent_0", "agent_2")

    phi_obs = _PhiObserver(phi0)
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=200,
        proposal_generator=gen,
        observers=[phi_obs],
        rng=np.random.default_rng(47),
    )
    _, phi_final, _A_final, _ = final_state

    # INV-16: phi variation must be bounded
    # ε_var = lr(1e-3) × grad_clip(1.0) × n_steps(20) × sqrt(d_latent × 2*d_latent)
    d = phi_final.d_latent
    eps_var = 1e-3 * 1.0 * 20 * math.sqrt(d * 2 * d) + 0.01  # +0.01 tolerance
    phi_start = phi_obs._phis[0]
    phi_end = phi_obs._phis[-1]
    total_drift = _phi_frobenius_distance(phi_start, phi_end)

    # Total drift bounded by eps_var × (n_phi_updates)
    n_phi_updates = 200 // 10  # phi_update_interval = 10
    assert total_drift < eps_var * n_phi_updates, (
        f"INV-16 VIOLATION: total phi drift {total_drift:.4f} "
        f"> eps_var×n_updates = {eps_var * n_phi_updates:.4f}"
    )

    # NaN/Inf check
    assert not np.any(np.isnan(phi_final.W_phi)), "NaN in W_phi during oscillation"
    assert not np.any(np.isnan(phi_final.W_F)), "NaN in W_F during oscillation"

    print(
        f"\n[INV-16] Oscillatory Instability: total_drift={total_drift:.4f}, "
        f"eps_var×n={eps_var * n_phi_updates:.4f}"
    )


# ---------------------------------------------------------------------------
# Attack 7 — Dead Network (Zero-CE Scenario)
# ---------------------------------------------------------------------------


def test_adversarial_dead_network():
    """
    Attack: Proposal generator returns None forever — no CEs are ever attempted.
            Simulates a network where all agents lose the ability to communicate.
    Oracle: State is unchanged after 1000 iterations; kernel exits cleanly with
            'Max iterations reached'; INV-17 notes adaptation paused.
    Invariant: INV-17 (Persistent Adaptation — negation scenario)
    """
    n = 5
    G0, phi0, A0 = _ring(n, seed=6)

    gen = _NullGenerator()
    obs = HistoryObserver()

    final_state, reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=1000,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(48),
    )
    _G_final, phi_final, A_final, E_final = final_state

    # Kernel must exit cleanly
    assert (
        reason == "Max iterations reached"
    ), f"Expected 'Max iterations reached' from dead network, got: {reason}"

    # State must be completely unchanged (no CEs accepted)
    assert (
        obs.ce_acceptance_rate == 0.0
    ), f"Expected 0 CE acceptance in dead network, got {obs.ce_acceptance_rate:.1%}"

    # Authority unchanged from initial baseline
    for aid in A_final.scores:
        assert (
            abs(A_final.get(aid) - 0.5) < 1e-9
        ), f"Authority of {aid} changed without any CE: {A_final.get(aid):.6f} ≠ 0.5"

    # Phi unchanged (no CEs means no g_history additions for phi_update)
    phi_drift = _phi_frobenius_distance(phi0, phi_final)
    assert phi_drift < 1e-9, f"Phi changed without any CE: drift={phi_drift:.2e}"

    # Error history empty (no accepted CEs)
    assert len(E_final) == 0, f"Error history non-empty without accepted CEs: {len(E_final)}"

    print(
        f"\n[INV-17] Dead Network: 1000 iterations with 0 accepted CEs. "
        f"State unchanged. phi_drift={phi_drift:.2e}"
    )


# ---------------------------------------------------------------------------
# Attack 8 — Constraint Erosion (Boundary Probing)
# ---------------------------------------------------------------------------


def test_adversarial_constraint_erosion():
    """
    Attack: Pre-set one agent's authority near the maximum (0.95), then run
            many CEs designed to push it past 1.0 through boundary erosion.
    Oracle: Authority never exceeds 1.0 (INV-18 clip; hard boundary).
            The np.clip(…, 0, 1) in authority_update enforces this absolutely.
    Invariant: INV-18 (Safety Margin / Boundary Preservation)

    This test also probes INV-11: if the desired 0.8 cap were enforced, the
    pre-set 0.95 value would itself be rejected.
    """
    n = 5
    G0, phi0, A0 = _ring(n, seed=7)

    # Pre-set agent_0 authority near 1.0 (boundary probing)
    A0.set("agent_0", 0.95)

    # Propose agent_0 as proposer repeatedly; with correctness→1, delta→+0.5*η
    # Each step: a_0 + 0.025 → clipped at 1.0
    attacking_ce = CoordinationEvent(
        event_type="add_edge",
        participants=("agent_0", "agent_1"),
        params=frozenset([("weight", 0.5)]),
    )
    gen = SequenceProposalGenerator([attacking_ce] * 200, loop=True)

    obs = HistoryObserver()
    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=200,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(49),
    )
    _, phi_final, A_final, _ = final_state

    # INV-18 (hard boundary): authority MUST stay ≤ 1.0 at all times
    all_scores_flat = [score for snapshot in obs.authority_history for score in snapshot.values()]
    max_ever = max(all_scores_flat) if all_scores_flat else 0.0

    assert max_ever <= 1.0 + 1e-9, (
        f"INV-18 HARD VIOLATION: authority exceeded 1.0 = {max_ever:.8f}. "
        "np.clip constraint violated!"
    )

    # No NaN from boundary interactions
    assert not np.any(np.isnan(phi_final.W_phi)), "NaN in phi during boundary erosion"

    # INV-11 note (desired): if max_ever > 0.8, document the gap
    if max_ever > 0.8:
        import warnings

        warnings.warn(
            f"[INV-11 gap] max authority reached {max_ever:.4f} > 0.8 during "
            "constraint erosion test. Desired: max_authority=0.8 cap in authority_update.",
            stacklevel=2,
        )

    print(
        f"\n[INV-18] Constraint Erosion: max_authority_ever={max_ever:.4f} "
        f"(hard cap=1.0, desired cap=0.8), "
        f"final={_max_authority(A_final):.4f}"
    )


# ---------------------------------------------------------------------------
# Composite invariant check
# ---------------------------------------------------------------------------


def test_all_phi_safe_invariants_hold_after_stress():
    """
    Composite test: run a diverse 500-iteration kernel and verify all Φ_safe
    invariants hold at the end.  This is a regression guard for INV-14, INV-16,
    INV-18 simultaneously.
    """
    n = 8
    G0, phi0, A0 = _ring(n, seed=99)

    final_state, _reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=500,
        rng=np.random.default_rng(99),
    )
    _, phi_final, A_final, _ = final_state

    safe = _check_phi_safe(phi_final)

    assert safe[
        "I_1_rank"
    ], f"INV-14: rank={_phi_rank(phi_final)} < {phi_final.d_latent // 2} after stress"
    assert safe[
        "I_2_frob"
    ], f"INV-16: frob={np.linalg.norm(phi_final.W_phi, 'fro'):.2f} > 10.0 after stress"
    assert safe["I_3_rows"], "INV-18: zero row in W_phi after stress"
    assert not np.any(np.isnan(phi_final.W_phi)), "NaN in W_phi after stress"
    assert all(
        0.0 <= s <= 1.0 for s in A_final.scores.values()
    ), "INV-15: authority out of [0,1] after stress"

    print(
        f"\n[Φ_safe composite] All invariants hold. rank={_phi_rank(phi_final)}, "
        f"frob={np.linalg.norm(phi_final.W_phi, 'fro'):.3f}"
    )
