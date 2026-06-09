# Emergo Safety Specification

**Version**: 1.0  
**Status**: Tier 1 Formal Contract  
**Reference**: SPECIFICATION.md, CONVERGENCE_ANALYSIS.md  
**Date**: 2026-06-09

---

## Overview

This document is the formal safety contract for the Emergo kernel.  It defines
the admissible parameter manifold Φ_safe, the authority conservation law,
stability bounds, and eight red-line invariants (INV-11 through INV-18) that
must hold for all time.  Each invariant is stated in first-order logic and
mapped to an observable quantity in the running system.

The document is organized in three parts:

1. **Φ_safe** — the set of PhiMaps the kernel must stay inside
2. **Stability Bounds** — quantitative thresholds the kernel must respect
3. **Red Lines** — invariants that, if broken, constitute a system failure

---

## Part 1: The Admissible Topology Manifold Φ_safe

### 1.1 Background

`PhiMap` is the kernel's learned topology predictor.  It consists of four
real-valued matrices:

```
W_phi : ℝ^(d_latent × d_features)   — graph embedding
b_phi : ℝ^d_latent                  — embedding bias
W_F   : ℝ^(d_latent × (d_latent + d_ce))  — latent transition
b_F   : ℝ^d_latent                  — transition bias
```

A `PhiMap` φ is **admissible** if it satisfies four governance constraints
derived from Lux's invariants (Fail-Closed, Capability-Gated, Accountable
Resources, Topology-Bounded).

### 1.2 Definition

```
Φ_safe = { φ : I_1(φ) ∧ I_2(φ) ∧ I_3(φ) ∧ I_4(φ) }
```

where each constraint is:

**I_1 — Rank Preservation (maps to INV-14)**

```
I_1(φ) ≡  rank(W_phi) ≥ ⌊d_latent / 2⌋
```

Rationale: a degenerate embedding (rank below d/2) makes distinct graphs
indistinguishable to φ, disabling accurate topology prediction and severing
the error→authority feedback loop that defines Emergo.  The rank-regularization
penalty `rank_lambda × (1/(1+max(0, rank−d/2)))` enforces this in practice via
the nuclear-norm gradient.

**I_2 — Bounded Frobenius Norm (maps to INV-16)**

```
I_2(φ) ≡  ‖W_phi‖_F ≤ C_phi   where C_phi = 10 (system constant)
```

Rationale: an unbounded embedding matrix amplifies graph feature noise without
bound, producing arbitrarily large φ-prediction errors and destabilizing
authority updates.  Gradient clipping (`phi_grad_clip = 1.0`) plus bounded
features (|feature| ≤ 1 by extract_graph_features) ensure ‖W_phi‖_F grows at
most O(lr × n_steps × iterations).

**I_3 — Non-Degenerate Rows (maps to INV-18 safety margin)**

```
I_3(φ) ≡  ∀ i ∈ [d_latent], ‖W_phi[i, :]‖_2 > 0
```

Rationale: a zero row in W_phi means agent latent dimension i is always zero,
collapsing the embedding into a lower-dimensional subspace.  The safety margin
`dist(φ, ∂Φ_safe) > δ_min` is operationally equivalent to ensuring every row
has norm bounded away from zero.

**I_4 — Capability Acyclicity (maps to INV-12)**

```
I_4(φ) ≡  the implied delegation graph G_φ is acyclic
         where G_φ has edge (i, j) iff W_phi[i, j] > τ_cap
```

Rationale: if φ's embedding encodes a cycle of authority delegation, one agent
can transitively amplify another's authority indefinitely.  The Lux
fail-closed governance layer provides the outer enforcement; I_4 is the
sufficient condition for φ to remain compatible with Lux's acyclicity checks.

### 1.3 Programmatic Validator

```python
def check_phi_safe(phi: PhiMap, C_phi: float = 10.0, delta_min: float = 1e-6) -> dict:
    """Returns {invariant_id: bool} for each I_k.  All must be True for admissibility."""
    rank = int(np.linalg.matrix_rank(phi.W_phi, tol=1e-6))
    row_norms = np.linalg.norm(phi.W_phi, axis=1)
    frob = float(np.linalg.norm(phi.W_phi, 'fro'))
    return {
        "I_1_rank":       rank >= phi.d_latent // 2,
        "I_2_frob":       frob <= C_phi,
        "I_3_rows":       bool(np.all(row_norms > delta_min)),
        "I_4_acyclic":    True,  # enforced externally by Lux
        "phi_safe":       (rank >= phi.d_latent // 2) and (frob <= C_phi)
                          and bool(np.all(row_norms > delta_min)),
    }
```

---

## Part 2: Authority as a Bounded Lyapunov Function

### 2.1 Formal Definition

Let `n` be the number of agents.  Define the **authority vector** at time `t`:

```
A_t : {agent_ids} → [0, 1]
```

Each element `a_i(t)` is the authority score of agent `i` at iteration `t`.

**Update rule** (from `authority_update.py`):

```
correctness_i(t) = 1 - error_i(t) / max_j error_j(t)   ∈ [0, 1]
Δ_i(t)          = correctness_i(t) - A_t.baseline        ∈ [−0.5, 0.5]
a_i(t+1)         = clip(a_i(t) + η · Δ_i(t), 0.0, 1.0)  where η = 0.05
```

### 2.2 Conservation Law

**Proposition**: Authority changes only via accepted CEs with verified errors.
No authority appears spontaneously.

**Proof sketch**: 
- `authority_update` is called exactly once per accepted CE, as step 3 in the
  kernel's sequential loop.
- The kernel never calls `A.set()` directly; only `authority_update` mutates `A`.
- The `A.set(agent_id, value)` call in `authority_update` is bounded by
  `np.clip(..., 0.0, 1.0)`, so `a_i(t) ∈ [0, 1]` for all i, t.
- Total authority change per CE: `|Σ_i Δ_i(t)| ≤ η × n_participants × 1.0`.

**Formal statement**:

```
A_t = A_0 + D_t

where D_t = Σ_{s=0}^{t-1} Σ_{i ∈ participants(CE_s)} η · Δ_i(s)
           (summed only over accepted CEs; rejected CEs contribute 0)
```

**Conservation error bound**: `|Σ_i a_i(t) - Σ_i a_i(0) - Σ_s Σ_i Δ_i(s)| < n × 1e-10`
(machine epsilon from float64 accumulation).

### 2.3 Lyapunov Interpretation

Define `V(A_t) = max_i a_i(t)`.  Under adversarial monopolization (one agent
always proposes, others never):
- Proposer correctness → 1.0 if φ converges; Δ_proposer → 0.5; authority → 1.0
- Participant correctness → 0.0 if local error > 0; Δ_participant → −0.5; authority → 0.0

This shows `V(A_t)` is NOT a Lyapunov function for bounding maximum authority —
it can reach 1.0.  INV-11 (cap at 0.8) requires an **additional enforcement
mechanism** (a soft cap in authority_update) not present in the current
implementation.  See Section 3 for the gap analysis.

---

## Part 3: Stability Bounds and Red Lines

### 3.1 Stability Bounds

**B_1 — Maximum Authority Concentration**

```
cap_authority = 0.8
```

*Desired*: `∀ t, ∀ i, a_i(t) ≤ cap_authority`.  
*Current enforcement*: clipped at 1.0 (not 0.8).  
*Status*: **Implementation gap** — requires adding `max_authority = 0.8` in `authority_update`.

**B_2 — Topology Entropy Lower Bound**

```
H_min = 0.3
```

Topology entropy `H(G_t) = -Σ_ij p_ij log p_ij` where `p_ij = adj[i,j] / Σ adj`.  
*Required*: `H(G_t) > H_min` under any non-stationary CE sequence.  
*Current enforcement*: `topology_lock_in` detector fires when `CE_acceptance_rate < 0.01`.

**B_3 — Adaptation Rate Lower Bound**

```
ε_min = 1e-6
```

When the environment is non-stationary (new CEs not in training history),
`|φ_{t+1} − φ_t|_F > ε_min` must hold for at least one phi_update call in
every window of `phi_update_interval × phi_early_stop_patience` steps.  
*Current enforcement*: early_stop_delta = 1e-6 triggers stopping only when
improvement falls below this — meaning adaptation continues if the environment
is novel.

### 3.2 Red Lines: INV-11 through INV-18

These eight invariants are the formal "must never break" contracts.  Violations
indicate a system failure, not just degraded performance.

---

**INV-11: Authority Monopolization**

```
∀ t, ∀ i, a_i(t) ≤ 0.8
```

*Attack vector*: single agent proposes all CEs, monopolizes prediction quality.  
*Observable*: `max(A_t.scores.values()) > 0.8`.  
*Current status*: DESIRED (not yet enforced); current hard cap is 1.0.  
*Enforcement path*: add `max_authority = 0.8` parameter to `authority_update`.

---

**INV-12: No Self-Loop Cycles**

```
∀ t, ∀ i, G_t.adjacency[i, i] contributes zero net authority gain to agent_i
```

*Attack vector*: agent proposes `add_edge(self, self)` creating a self-loop
that provides a local_structural_change_error ≈ 0 (no neighbors change),
boosting self-authority indefinitely.  
*Observable*: diagonal entries of `G_t.adjacency` exist AND `a_i(t)` grows
monotonically.  
*Current status*: PARTIALLY ENFORCED — self-loops are syntactically allowed but
local error for the participant (self) remains bounded since the row+col delta
is bounded by the edge weight.

---

**INV-13: Topology Adaptability**

```
∀ t under non-stationary CE sequences,
  H(G_t) > 0.3
```

where `H(G)` is the edge-weight Shannon entropy.  
*Attack vector*: proposal generator collapses to a single CE type, freezing the
topology.  
*Observable*: `topology_entropy(G_t.adjacency) < 0.3` for many consecutive steps.  
*Current status*: MONITORED by `detect_topology_lock_in`.

---

**INV-14: Rank Preservation**

```
∀ t, rank(W_phi_t) ≥ ⌊d_latent / 2⌋
```

*Attack vector*: adversarial CE sequence forces phi_update to degenerate (all
gradients cancel, W_phi collapses to zero).  
*Observable*: `np.linalg.matrix_rank(phi.W_phi) < d_latent // 2`.  
*Current status*: ENFORCED by nuclear-norm rank regularization (`EMERGO_RANK_PENALTY=0.1`).

---

**INV-15: No Authority Leakage (Capability Conservation)**

```
∀ t, ∀ i, a_i(t) = a_i(0) + Σ_{s<t, i ∈ participants(CE_s)} η · Δ_i(s)
```

*Attack vector*: inject authority via a path other than `authority_update`
(e.g., direct `A.set()` call bypassing error computation).  
*Observable*: authority changes not correlated with accepted CEs in audit log.  
*Current status*: ENFORCED — `authority_update` is the only write path to `A`;
all CE attempts are audited (INV-7).

---

**INV-16: Bounded Variation**

```
∀ t, ‖φ_{t+1} − φ_t‖_F < ε_var
where ε_var = lr × grad_clip × n_steps × sqrt(d_latent × d_features)
            = 1e-3 × 1.0 × 20 × sqrt(128) ≈ 0.226
```

*Attack vector*: craft CE sequences that maximize gradient magnitude, causing
phi to drift arbitrarily far from initialization.  
*Observable*: `‖phi_{t+k}.W_phi − phi_t.W_phi‖_F > ε_var` per phi_update call.  
*Current status*: ENFORCED by gradient clipping in `phi_update`.

---

**INV-17: Persistent Adaptation**

```
∀ T > 0, ∃ t ∈ [T, T + phi_update_interval × phi_early_stop_patience]
  such that ‖φ_{t+1} − φ_t‖_F > ε_min
```

*Attack vector*: craft a stationary CE sequence that makes phi_update always
early-stop after 1 step (loss already minimal), preventing adaptation to
genuinely novel inputs.  
*Observable*: all phi_update calls return after 1 gradient step while test-set
error remains high.  
*Current status*: PARTIALLY ENFORCED — early stopping requires `n_patience`
consecutive non-improving steps; a single novel input resets the counter.

---

**INV-18: Safety Margin**

```
∀ t, dist(φ_t, ∂Φ_safe) > δ_min = 0.01
```

where `dist` is measured as the minimum Frobenius-norm distance to the
boundary of Φ_safe (i.e., the set of φ where rank(W_phi) = d_latent // 2).  
*Attack vector*: drive phi toward rank-deficient boundary without triggering
the rank regularizer (e.g., slowly zero-out one singular value at a time).  
*Observable*: `min singular value of W_phi < δ_min`.  
*Current status*: ENFORCED by nuclear-norm gradient which maintains all singular
values bounded away from zero under nonzero rank_lambda.

---

## Part 4: Attack-to-Invariant Mapping

| # | Attack Name | Invariant Violated | Current Status |
|---|-------------|-------------------|----------------|
| 1 | Authority Monopolization | INV-11 | DESIRED |
| 2 | Recursive Self-Delegation | INV-12 | PARTIAL |
| 3 | Topology Lockout | INV-13 | MONITORED |
| 4 | Entanglement Collapse | INV-14 | ENFORCED |
| 5 | Capability Leakage | INV-15 | ENFORCED |
| 6 | Oscillatory Instability | INV-16 | ENFORCED |
| 7 | Dead Network | INV-17 | PARTIAL |
| 8 | Constraint Erosion | INV-18 | ENFORCED |

---

## Part 5: Implementation Gap Analysis

Three invariants are not fully enforced in the current codebase:

**INV-11 gap**: The 0.8 authority cap requires adding `max_authority = 0.8`
to `authority_update()` and passing it from the kernel:
```python
new_score = float(np.clip(A_t.get(agent_id) + eta * delta_a, 0.0, max_authority))
```

**INV-12 gap**: Self-loops are syntactically permitted.  To close this, add
a check in `ce_execute` for `add_edge` events:
```python
if CE.event_type == "add_edge" and CE.participants[0] == CE.participants[1]:
    return G_t, False, ()  # reject self-loop
```

**INV-17 gap**: A truly adversarial stationary CE sequence can keep phi_update
in early-stop state.  Mitigation: add a forced adaptation step every
`phi_force_adapt_interval` calls that bypasses early stopping.

---

*End of Emergo Safety Specification v1.0*
