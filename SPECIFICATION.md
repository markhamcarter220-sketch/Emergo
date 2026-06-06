# EMERGO KERNEL — SPECIFICATION

## Overview

Emergo is the adaptive interaction layer that sits on top of Lux (the governance kernel).

```
┌──────────────────────────────────────────┐
│               Emergo                     │
│  Planner → Executor → [4 operations]    │
│  φ-learning, authority, error feedback  │
└──────────────┬───────────────────────────┘
               │  LuxBridge (authorize, deduct, audit)
┌──────────────▼───────────────────────────┐
│                Lux                       │
│  Capabilities, Ledger, Policy, Audit    │
└──────────────────────────────────────────┘
```

- **Lux**: stable structural layer — capabilities, resource ledger, topology enforcement, fail-closed.
- **Emergo**: adaptive interaction layer — planning, decomposition, execution, learning.

Emergo never grants capabilities, never directly modifies ledger balances, and never
bypasses Lux. Every coordination event goes through the LuxBridge.

---

## State Machine

```
State = (G_t, φ_t, A_t, E_t)

Iteration(State):
  1. CE_Execution(G_t, CE_t, Lux)           → G_{t+1}
  2. ErrorComputation(G_t, G_{t+1}, φ_t)    → ε_per_agent
  3. AuthorityUpdate(A_t, ε_per_agent)       → A_{t+1}
  4. PhiUpdate(all_errors, G_history)        → (φ_{t+1}, F_{t+1})

  return (G_{t+1}, φ_{t+1}, A_{t+1}, E_{t+1})
```

Repeat until convergence (φ errors plateau).

### Execution Runtime (Executor)

```
Goal → Planner.decompose() → [Task_1, Task_2, ...]

For each Task:
  INV-8 guards (depth + pending quota)
  lux.authorize_full(execute_task_CE)    ← capability + resource pre-deduction
  TaskRunner(task) → TaskOutcome
  [on failure: lux.refund_resource()]   ← INV-6
  lux.audit(result)                     ← INV-7
  ce_execute(update_capabilities CE)    ← INV-5: only path for graph mutation
  error_computation + authority_update  ← feedback loop
```

---

## State Components

- **G_t**: Immutable graph. Agents are nodes; directed weighted edges encode relationships.
  `adjacency[i,j]` = weight; `capabilities[i,:]` = per-agent feature vector.
  Arrays are locked read-only (`writeable=False`) after construction.

- **φ_t**: Learned latent map φ: G → R^d. Parameterized by (W_phi, b_phi).
  F: R^d × CE_enc → R^d is the stationary transition operator.
  Must be non-factorizable (entangled): rank(W_phi) ≥ d_latent//2 at all times.

- **A_t**: Per-agent authority scores in [0, 1].
  High authority → higher probability of initiating coordination events.
  Updates are continuous (no discrete jumps).

- **E_t**: History of per-agent prediction errors from all past iterations.

- **Task / Goal**: Execution-layer types (see `types.py`).
  Task carries `depth` for INV-8; `required_capability` and `resource_cost` for INV-5/6.

---

## Operations

### 1. CE_Execution

- Call `lux.authorize(CE, G, A)` — if false, return unchanged G_t (no resource charge here)
- Apply CE to G_t deterministically (all-or-nothing)
- Produce G_{t+1}

**Graph-mutation CE types** (handled by ce_execute):
`add_edge`, `remove_edge`, `update_capabilities`, `add_agent`, `remove_agent`

**Execution-layer CE types** (handled by Executor, NOT ce_execute):
`execute_task`, `decompose_goal`, `delegate`
Routing these to ce_execute raises ValueError (explicit boundary enforcement).

### 2. ErrorComputation

For each agent `a` in CE_t.participants:
- Predicted shift: φ̃_{t+1} = F_t(φ_t(G_t), encode(CE_t))
- Actual shift: z_{actual} = φ_t(G_{t+1})
- error_a = ||φ̃_{t+1} − z_{actual}||₂

### 3. AuthorityUpdate

For each agent `a`:
- correctness_a = 1 − (error_a / max_error_in_batch)
- Δ_a = correctness_a − baseline
- A_{t+1}(a) = clamp(A_t(a) + η·Δ_a, 0, 1)   [η ≈ 0.01–0.1]

### 4. PhiUpdate

Minimize: L = (1/T) Σ_t (1/2)||φ(G_{t+1}) − F(φ(G_t), encode(CE_t))||²

- φ(G) = W_phi @ f(G) + b_phi  (spectral + capability features, fixed dim)
- F(z, c) = W_F @ [z; c] + b_F  (linear transition in φ-space)
- Exact gradients; gradient clipping at ±1.0
- Entanglement guard: if rank(W_phi) < d_latent//2 after update → revert to φ_t

### LuxBridge Contract

Emergo interacts with Lux exclusively through `LuxBridge`:

| Method | Called by | Purpose |
|--------|-----------|---------|
| `authorize_ce(CE, G, A, reserve_resources=False)` | `ce_execute` | Check-only auth |
| `authorize_ce(CE, G, A, reserve_resources=True)` | `Executor` | Auth + pre-deduct |
| `refund_resource(agent, resource, amount)` | `Executor` (failure path) | Rollback charge |
| `check_capability(agent, capability)` | `LuxBridge.authorize_ce` | INV-5 |
| `grant_capability(agent, capability)` | System initializer only | Never from CEs |
| `audit(ce_type, agents, success, ...)` | `Executor`, `Lux` | INV-7 |

---

## Invariants

### INV-1: State Ownership
`(G_t, φ_t, A_t, E_t)` is the only mutable state. All operations are pure functions
over this tuple. Graph arrays are read-only (`writeable=False`). No shared mutable
sub-state exists between operations.

### INV-2: Feedback Loop (Observable)
`error → authority → CE sampling probability → topology → next error` is a closed loop.
φ updates are visible to the next iteration. No shortcut paths.

### INV-3: Blast Radius
Failed CEs leave G_t unchanged (all-or-nothing). φ reverts to φ_t if entanglement
violated. Unauthorized CEs are rejected at the gate; no partial resource charges.

### INV-4: Timing (Atomic, Sequential)
The fixed-point loop is strictly sequential. CE execution is atomic. No parallelism,
no shared mutable state, no deadlocks.

### INV-5: Proposal-Only Authority
Emergo may only *propose* capability changes; it cannot mint capabilities or bypass
Lux policy. Capability changes enter the graph exclusively through
`ce_execute(update_capabilities CE)` — never by direct field assignment.
`grant_capability` is a system-init-only operation; no CE type triggers it.

### INV-6: Resource Conservation via Lux Ledger
Every `execute_task` CE pre-deducts resource from the Lux ledger before execution
begins (`lux.authorize_full`). If execution fails for any reason (authorization denied,
TaskRunner exception, task failure), `lux.refund_resource()` is called unconditionally.
No resource leaves an agent's budget without a corresponding successful execution.

### INV-7: Observable + Fail-Closed Propagation
Every CE execution attempt (authorized or not, succeeded or not) produces an audit record
via `lux.audit()`. Audit is written before the state update is considered committed.
Audit records are stored as deep copies — callers cannot retroactively modify them.
`RealLuxBridge.audit()` raises `LuxError` on failure (fail-closed, not silent).

### INV-8: Bounded Speculation
- `task.depth > goal.max_depth` → CE is rejected before authorization, with audit record.
- `pending[agent] >= MAX_PENDING_CES_PER_AGENT` → CE is rejected, with audit record.
- Pending count is decremented after every task (success, failure, or exception).
- Both limits are configurable via env vars (`EMERGO_MAX_DEPTH`, `EMERGO_MAX_PENDING`).

---

## Convergence

The kernel converges when the mean φ-prediction error plateaus across two consecutive
windows of `W` iterations. This signals a stable latent geometry of coordination dynamics:
agents' predictions match the actual topological effects of their coordination events.

---

## Configuration

All runtime parameters are in `emergo/config.py` and can be overridden via env vars:

| Env var | Default | Meaning |
|---------|---------|---------|
| `EMERGO_LUX_MODE` | `simulated` | `simulated` or `real` |
| `EMERGO_MAX_DEPTH` | `5` | Max task decomposition depth (INV-8) |
| `EMERGO_MAX_PENDING` | `10` | Max pending CEs per agent (INV-8) |
| `EMERGO_ETA` | `0.05` | Authority update learning rate |
| `EMERGO_PHI_INTERVAL` | `10` | φ update frequency |
| `EMERGO_INITIAL_BUDGET` | `100.0` | Starting resource balance (simulated) |
| `EMERGO_TASK_COST` | `1.0` | Default task resource cost |
