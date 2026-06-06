# EMERGO KERNEL — SPECIFICATION

## Overview

A self-referential graph dynamical system where agents learn to predict topology evolution.
Every operation enforces entangled φ-space learning under correctness-weighted authority feedback.

## State Machine

```
State = (G_t, φ_t, A_t, E_t)

Iteration(State):
  1. CE_Execution(G_t, CE_t, Lux) → G_{t+1}
  2. ErrorComputation(G_t, G_{t+1}, φ_t, F_t) → ε_per_agent
  3. AuthorityUpdate(A_t, ε_per_agent) → A_{t+1}
  4. PhiUpdate(all_errors, G_history) → (φ_{t+1}, F_{t+1})

  return (G_{t+1}, φ_{t+1}, A_{t+1}, E_{t+1})
```

Repeat until convergence (φ errors plateau).

## State Components

- **G_t**: Immutable graph. Agents are nodes; directed weighted edges encode relationships.
  Capabilities matrix encodes per-agent attributes.
- **φ_t**: Learned latent map φ: G → R^d. Parameterized by (W_phi, b_phi).
  Must be non-factorizable (entangled): no per-agent decomposition.
- **A_t**: Per-agent authority scores in [0,1]. High authority → broader reachability.
- **E_t**: History of per-agent prediction errors from all past iterations.

## Operations

### 1. CE_Execution

- Check Lux.authorize(CE, G, A) — if false, return failure
- Apply CE to G_t deterministically (all-or-nothing)
- Produce G_{t+1}
- CE types: add_edge, remove_edge, update_capabilities, add_agent, remove_agent

### 2. ErrorComputation

For each agent `a` in CE_t.participants:
- Predicted shift: φ̃_{t+1} = F_t(φ_t(G_t), encode(CE_t))
- Actual shift: φ_{t+1}(G_{t+1})
- error_a = ||φ̃_{t+1} - φ_{t+1}(G_{t+1})|| (L2 norm)

### 3. AuthorityUpdate

For each agent `a`:
- correctness_a = 1 - (error_a / max_possible_error)  [normalized to [0,1]]
- Δ_a = correctness_a - baseline
- A_{t+1}(a) = clamp(A_t(a) + η * Δ_a, 0, 1)  [η ≈ 0.01–0.1]

### 4. PhiUpdate

Minimize: L = Σ_t ||φ(G_{t+1}) - F(φ(G_t), encode(CE_t))||²

- φ(G) = W_phi @ f(G) + b_phi  (linear projection from spectral graph features)
- F(z, c) = W_F @ [z; c] + b_F  (linear transition in latent space)
- Joint optimization over (W_phi, b_phi, W_F, b_F)
- Entanglement constraint: W_phi must have rank ≥ d_latent/2; reject if violated
- Axiom 5.4.3: F is approximately linear in φ-space

## Invariants

1. **State ownership**: (G_t, φ_t, A_t, E_t) is the only mutable state; all ops are pure functions
2. **Feedback/observability**: error → authority → topology; φ updates visible to next iteration
3. **Blast radius**: Failures degrade gracefully; φ rolls back if entanglement violated
4. **Timing**: Fixed-point loop is sequential; all operations are atomic; no race conditions

## Convergence

System converges when mean φ-prediction error plateaus across consecutive windows.
This signals a stable latent geometry of coordination dynamics.
