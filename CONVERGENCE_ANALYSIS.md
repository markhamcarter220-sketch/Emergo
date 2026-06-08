# Convergence Analysis — Emergo Kernel

*Proof sketch and auditor checklist for the Emergo fixed-point iterator.*

---

## 1. The Fixed-Point Problem

The Emergo kernel iterates the state tuple

    S_t = (G_t, φ_t, A_t, E_t)

through four sequential atomic operations per accepted CE:

| Step | Operation           | Output          |
|------|---------------------|-----------------|
| 1    | CE_Execution        | G_{t+1}         |
| 2    | ErrorComputation    | e_t             |
| 3    | AuthorityUpdate     | A_{t+1}         |
| 4    | PhiUpdate (every k) | φ_{t+1}         |

We seek conditions under which `S_t → S*` as `t → ∞`.

---

## 2. Theorem C1 — Determinism (Well-Definedness)

**Statement.**  For any initial state `S_0` and any fixed sequence of accepted CEs
`{CE_t}`, the state trajectory `{S_t}` is uniquely determined.  No step depends on
mutable global state, hidden randomness after CE sampling, or agent-ordering.

**Proof sketch.**

*Step 1 — CE_Execution.*  `ce_execute(G_t, CE_t, lux, A_t)` is a deterministic
function: given the same inputs it returns the same `(G_{t+1}, success, _)`.  The
Lux bridge (`SimulatedLuxBridge`) operates under a mutex; resource checks are
pure comparisons; capability grants are idempotent on the same input.  No
side-channel state leaks between calls.

*Step 2 — ErrorComputation.*  `error_computation(G_t, G_{t+1}, φ_t, CE_t)`
computes:

    global_error = ||φ_t.embed(G_t+1) - φ_t.transition(φ_t.embed(G_t), encode(CE_t))||

and local adjacency delta norms for non-proposing participants.  Both are
closed-form linear operations on fixed matrices; the result is a pure function
of its arguments.

*Step 3 — AuthorityUpdate.*  `authority_update(A_t, e_t)` updates each
participant's score by `η × (correctness − baseline)`.  This is a bounded
linear map `A_{t+1} = clip(A_t + Δ, min_auth, max_auth)`.  Deterministic.

*Step 4 — PhiUpdate.*  `phi_update(φ_t, G-history, CE-history, ...)` runs
gradient descent on a quadratic loss (plus a convex nuclear-norm regularizer).
Given fixed inputs, the gradient sequence is uniquely determined.  ∎

**Corollary (Reproducibility).**  Fixing the RNG seed at kernel entry guarantees
identical trajectories across runs.

---

## 3. Theorem C2 — Stability (Bounded Authority)

**Statement.**  For all `t ≥ 0`,

    min_authority ≤ A_t(i) ≤ max_authority    ∀ agent i.

Furthermore, `|A_{t+1}(i) − A_t(i)| ≤ η` for all `i`.

**Proof sketch.**

The authority update rule (with `η > 0`, `baseline ∈ (0,1)`, `correctness ∈ [0,1]`) is:

    delta_i = η × (correctness_i − baseline)
    A_{t+1}(i) = clip(A_t(i) + delta_i, min_auth, max_auth)

*Boundedness.*  The `clip` operation is applied unconditionally after every
update, so `A_t(i) ∈ [min_auth, max_auth]` for all `t` by induction.

*Lipschitz step bound.*  `|delta_i| = η × |correctness_i − baseline| ≤ η`,
because `correctness_i ∈ [0,1]` and `baseline ∈ [0,1]`.  Thus each step
moves at most `η` in authority space.  ∎

**Implication for topology.**  Because `A_t` is bounded away from zero by
`min_authority > 0` (default `0.1`), the proposal distribution (softmax over
authority-weighted edges) always has full support over connected edges.  Topology
exploration cannot permanently freeze.

---

## 4. Theorem C3 — Identifiability (φ Converges to True Topology Structure)

**Statement.**  Let `φ*` be a PhiMap such that `F(φ*(G_t), encode(CE_t)) ≈ φ*(G_{t+1})`
for all transitions in the history.  Under the gradient-descent update in
`phi_update`, the sequence `{φ_t}` converges to a neighborhood of `φ*` when:

1. The training loss `L(φ)` is strongly convex in a neighborhood of `φ*`
   (satisfied when the graph-feature matrix has full column rank — guaranteed
   by INV-7 entanglement invariant).
2. The learning rate `lr ≤ 2 / (L + μ)` (standard SGD convergence condition),
   where `L` is the Lipschitz constant of ∇L and `μ` is the strong-convexity
   modulus.

**Proof sketch.**

The prediction loss is

    L(W_phi, b_phi, W_F, b_F) = (1/T) Σ_{t=0}^{T-1}
        (1/2) ||W_phi f_{t+1} + b_phi − W_F [W_phi f_t + b_phi; c_t] − b_F||²

This is a sum of squared norms of affine functions of the parameters — i.e.,
a quadratic form.  It is convex; it is strongly convex when the composite
feature-transition matrix (formed by `[f_t; c_t]` stacked over all `t`) has
full column rank.

*Rank guarantee.*  `INV-7` (entanglement invariant) asserts `rank(W_phi) ≥ d_latent//2`.
The rank-regularization in `phi_update` enforces this during optimization via the
nuclear-norm gradient `−λ U Vᵀ`, which maximizes the sum of singular values and
prevents any singular value from collapsing to zero.  A matrix with all singular
values > 0 maps to a full-rank latent space, ensuring the feature-transition
matrix has the required rank.

*Convergence.*  Under strong convexity with constant `μ > 0` and Lipschitz
gradient with constant `L`:

    L(φ_{t+1}) − L(φ*) ≤ (1 − 2μ × lr / (1 + μ/L)) × (L(φ_t) − L(φ*))

giving linear (geometric) convergence rate.  With rank regularization `λ > 0`,
the regularized loss remains strongly convex (nuclear norm is convex, and a
convex perturbation preserves strong convexity modulus within a sublevel set).

Early stopping halts optimization once improvement falls below `early_stop_delta`,
yielding a final iterate within `O(early_stop_delta)` of the optimum.  ∎

**Practical note on identifiability.**  True identifiability (unique `φ*`) requires
distinct graph states and CE encodings to produce distinct latent trajectories.
This is monitored by `diagnostics.topology_entropy` — low entropy signals topology
lock-in and may require increased `rank_lambda` or a more diverse proposal generator.

---

## 5. Auditor Checklist

The following invariants can be verified mechanically via the test suite
(`python -m pytest tests/ -v`) or by inspection.

### Structural Invariants (always true)

| ID    | Invariant                                             | Verification                                      |
|-------|-------------------------------------------------------|---------------------------------------------------|
| INV-1 | `A_t(i) ∈ [min_authority, max_authority]` for all `t` | `test_invariants.py::TestInvariant1Authority`     |
| INV-2 | CE accepted ↔ authority influences proposal           | `test_invariants.py::TestInvariant2FeedbackLoop`  |
| INV-3 | Rejected CE leaves state unchanged                    | `test_invariants.py::TestInvariant3BlastRadius`   |
| INV-4 | Error attributed only to CE participants              | `test_invariants.py::TestInvariant4ErrorScope`    |
| INV-5 | φ dimensions preserved across phi_update              | `test_phi_update.py::test_phi_dimensions_preserved`|
| INV-6 | Original φ_t not mutated by phi_update                | `test_phi_update.py::test_original_phi_not_mutated`|
| INV-7 | `rank(W_phi) ≥ d_latent//2` after training            | `test_phi_update.py::test_rank_penalty_prevents_collapse`|
| INV-8 | Decomposition depth bounded by `MAX_DECOMPOSITION_DEPTH` | `test_executor.py`                            |
| INV-9 | Max concurrent proposals bounded                      | `test_coordinator.py`                             |
| INV-10| Observers are read-only; exceptions never propagate   | `test_observer.py::TestInvariant10`               |

### Convergence Properties (empirical)

| Property                  | How to Check                                    | Expected                         |
|---------------------------|-------------------------------------------------|----------------------------------|
| phi_loss decreasing       | `obs.phi_losses[-1] < obs.phi_losses[0]`        | True after ≥ 100 accepted CEs    |
| Authority exploration     | All agents have `A > min_authority` after 50 CEs | True (Theorem C2)               |
| No NaN in features        | `np.all(np.isfinite(features))`                 | True (always, by construction)   |
| Feature shape invariant   | `features.shape == (d_features,)` for any graph | True for all n_agents ≥ 0        |
| Rank preservation         | `rank(W_phi) ≥ d_latent//2`                     | True after phi_update with λ>0   |

### Red-Line Checks (must never occur)

| Red Line                                         | Detector                                         |
|--------------------------------------------------|--------------------------------------------------|
| NaN / Inf in phi_update gradients                | `_check_grad_norms` (logs WARNING if norm > 10×clip) |
| W_phi rank below threshold after training        | `phi_update` safety log (WARNING if rank < d//2) |
| Authority outside `[min, max]` bounds            | `authority_update` `clip` call                   |
| CE mutating state when `success=False`           | `ce_execute` returns original `G_t` unchanged    |
| Observer exception propagating into kernel loop  | `fire_observers` `try/except` wrapping           |

### Environment Variables (configuration audit)

| Variable                  | Default | Effect                                          |
|---------------------------|---------|-------------------------------------------------|
| `EMERGO_RANK_PENALTY`     | `0.1`   | Nuclear-norm regularization λ for phi_update    |
| `EMERGO_PHI_LR`           | `0.001` | Gradient descent learning rate                  |
| `EMERGO_PHI_OPTIMIZER`    | `sgd`   | Optimizer: `sgd` or `adam`                      |
| `EMERGO_PHI_EARLY_STOP`   | `5`     | Early-stopping patience (0 = disabled)          |
| `EMERGO_PHI_GRAD_CLIP`    | `1.0`   | Gradient clipping magnitude                     |
| `EMERGO_PHI_INTERVAL`     | `10`    | PhiUpdate frequency (every N accepted CEs)      |
| `EMERGO_ETA`              | `0.05`  | Authority update learning rate η                |
| `EMERGO_LUX_MODE`         | `simulated` | `simulated` or `real` Lux bridge           |
| `EMERGO_MAX_DEPTH`        | `5`     | Maximum CE decomposition depth (INV-8)          |

---

*End of convergence analysis.*
