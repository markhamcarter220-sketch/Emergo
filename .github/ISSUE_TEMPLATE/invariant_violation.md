---
name: Invariant violation
about: Report a case where a system invariant (INV-1 … INV-18) was broken
title: "[INV] "
labels: ["invariant-violation", "critical"]
assignees: ""
---

## Which invariant was violated?

<!-- INV-1 through INV-10 are operational; INV-11 through INV-18 are Tier 1 safety red-lines. -->

**Invariant ID**: INV-___ <!-- e.g. INV-11 -->
**Invariant name**: <!-- e.g. Authority Monopolization -->

## Reproduction

```python
# Minimal code that demonstrates the invariant breach
```

## Observable evidence

What did you observe that indicates the violation?  
For example:
- `max(A.scores.values()) > 0.8` for INV-11
- `np.linalg.matrix_rank(phi.W_phi) < phi.d_latent // 2` for INV-14
- NaN/Inf in phi parameters for INV numerical stability

## Impact

Describe the downstream effect on the system.

## Environment

- Emergo version: 
- Python version: 
- Relevant config (`EMERGO_*` env vars): 

## Reference

See `emergo/SAFETY_SPEC.md` for the formal definition of each invariant.
