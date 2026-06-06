# Emergo — Core Theorem Implementation

## The Core Theorem

> **Agents that better predict topology evolution gain greater influence over future topology evolution.**

This document explains how the theorem is formally verified in code, what the three consistency conditions mean, and what remains for future work.

---

## Three Consistency Conditions

The core theorem holds under three structural conditions on the state machine ℱ.

### C1: Well-Definedness

*ℱ produces unique, deterministic successors.*

The state machine is deterministic: given the same `(G_t, φ_t, A_t, CE_t)`, the next state `(G_{t+1}, φ_{t+1}, A_{t+1}, E_{t+1})` is always identical. No randomness is introduced between the input and the output.

Key sub-conditions:

- **CE determinism**: `ce_execute(G, CE, lux, A)` is a pure function. Same inputs → same `G_{t+1}`.
- **Error determinism**: `error_computation(G, G_next, phi, CE)` is pure. Same graph pair → same `Errors`.
- **Entanglement guard**: `phi_update` never returns a `PhiMap` with `rank(W_phi) < d_latent // 2`. If gradient steps collapse rank, the previous `phi_t` is returned unchanged.
- **Error scope**: Only CE participants receive errors from a given event. Non-participants are unaffected.

### C2: Stability

*Authority stays bounded; there is no runaway feedback.*

The authority update formula is:

```
A_{t+1}(a) = clamp(A_t(a) + η · (correctness_a − baseline), 0, 1)
```

where `correctness_a = 1 − (error_a / max_batch_error) ∈ [0, 1]`.

Key stability properties:

- **Bounded interval**: `A_{t+1}(a) ∈ [0, 1]` by construction (explicit clamp).
- **Bounded step size**: `|A_{t+1}(a) − A_t(a)| ≤ η · max(baseline, 1 − baseline)`. For `baseline = 0.5`, this is `η / 2`.
- **Order independence**: Updates read from `A_t`, never from `A_next`. Iteration order over agents doesn't affect the result.
- **No NaN propagation**: All inputs are finite reals; `max_batch_error` is guarded at `1e-8` to prevent division by zero.

### C3: Identifiability

*φ converges toward the real topology structure.*

The latent map `φ: G → R^d` is trained by gradient descent to minimize:

```
L = (1/T) Σ_t (1/2) ||φ(G_{t+1}) − F(φ(G_t), encode(CE_t))||²
```

Identifiability properties:

- **Loss decreases after training**: `phi_update` with `n_steps > 0` reduces prediction loss relative to the untrained phi.
- **Structural discrimination**: Dense and sparse graphs map to different embeddings (different eigenvalues → different spectral features → different `z`).
- **Edge density sensitivity**: Graphs with different edge counts have different spectral signatures.
- **Isomorphism invariance**: The feature extractor uses `eigvalsh` of the symmetric adjacency matrix, which is permutation-invariant. Isomorphic graphs (same structure, different node labels) produce identical embeddings.

---

## Test Coverage

| Condition | Test class | Tests | What is verified |
|-----------|-----------|-------|-----------------|
| C1 | `TestConsistency1WellDefinedness` | 5 | Determinism of CE execution, error computation, entanglement guard, rank preservation, error scope |
| C2 | `TestConsistency2Stability` | 6 | Boundedness, clamping, directional correctness, order independence, long-run stability, step-size bound |
| C3 | `TestConsistency3Identifiability` | 4 | Loss reduction, structural discrimination, density sensitivity, isomorphism invariance |
| Integration | `TestCoreTheoremValidation` | 2 | Core theorem over 30 rounds, full kernel loop end-to-end |
| **Total** | | **17** | |

Run the tests:

```bash
python -m pytest tests/test_core_theorem.py -v
```

---

## Authority Update Fix

The `authority_update` function was updated to make the clamp **explicit** in the formula body:

```python
# Before (implicit clamp via Authority.set())
new_score = A_t.get(agent_id) + eta * delta_a
A_next.set(agent_id, new_score)

# After (explicit clamp — makes the invariant visible in the formula)
new_score = float(np.clip(A_t.get(agent_id) + eta * delta_a, 0.0, 1.0))
A_next.set(agent_id, new_score)
```

The behavior is identical (both paths clamp to `[0, 1]` before storage), but making it explicit:

1. Aligns the code with the formula in `SPECIFICATION.md`.
2. Makes C2 stability immediately readable without needing to trace into `Authority.set()`.
3. Eliminates a subtle reading: previously a reader might wonder if unclamped intermediate values could propagate.

---

## What Remains

The tests in `tests/test_core_theorem.py` constitute **empirical verification** of the theorem's three consistency conditions. The following are not yet formalized:

| Item | Status |
|------|--------|
| Formal proof of convergence (Lyapunov function) | Not implemented |
| Long-horizon experiments (10k+ iterations) | Not implemented |
| Multi-agent differentiation with real LLM planners | Not implemented |
| Proof of identifiability under distribution shift | Not implemented |

These are research-level tasks appropriate for follow-on work once the kernel is stable.
