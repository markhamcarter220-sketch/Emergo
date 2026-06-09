---
name: Bug report
about: Report a reproducible defect in the Emergo kernel or API
title: "[BUG] "
labels: ["bug", "needs-triage"]
assignees: ""
---

## Describe the bug

A clear and concise description of the defect.

## Minimal reproducible example

```python
# Paste the smallest code that triggers the bug
import numpy as np
from emergo import emergo_kernel, make_initial_phi, make_initial_authority, Graph

n = 5
ids = tuple(f"a{i}" for i in range(n))
G = Graph(agent_ids=ids, adjacency=np.zeros((n, n)), capabilities=np.ones((n, 4)) * 0.5)
phi = make_initial_phi()
A = make_initial_authority(ids)

# ...
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened (include the full traceback if applicable).

```
Traceback (most recent call last):
  ...
```

## Environment

- Emergo version: <!-- e.g. 0.2.0 -->
- Python version: <!-- e.g. 3.11.2 -->
- OS: <!-- e.g. Ubuntu 22.04 / macOS 14 / Windows 11 -->
- Install method: <!-- pip / source -->

## Which invariant / module is affected?

<!-- Optional: reference the relevant invariant (INV-1 … INV-18), module (kernel.py, phi_update.py, etc.), or component (Executor, MultiAgentCoordinator, etc.) -->

## Additional context

Any other information, links to related issues, or screenshots.
