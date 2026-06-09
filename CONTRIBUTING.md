# Contributing to Emergo

Thank you for your interest in contributing. This guide covers everything you
need to go from zero to a passing pull request.

---

## Table of Contents

1. [Development Setup](#1-development-setup)
2. [Running Tests](#2-running-tests)
3. [Code Style and Linting](#3-code-style-and-linting)
4. [Running Examples](#4-running-examples)
5. [How to Add a New Planner](#5-how-to-add-a-new-planner)
6. [How to Add a New Observer](#6-how-to-add-a-new-observer)
7. [How to Add a New Proposal Generator](#7-how-to-add-a-new-proposal-generator)
8. [Invariant Testing Guidelines](#8-invariant-testing-guidelines)
9. [Pull Request Checklist](#9-pull-request-checklist)

---

## 1. Development Setup

### Prerequisites

- Python 3.9 or newer
- `git`

### Clone and install

```bash
git clone https://github.com/markhamcarter220-sketch/Emergo.git
cd Emergo

# Install in editable mode with all dev tools
pip install -e ".[dev]"

# Optional: install visualization extras
pip install -e ".[dev,viz]"
```

`pip install -e .` installs the package in *editable* mode: your local `emergo/`
directory **is** the installed package — no reinstall needed when you edit source
files.

### Verify the installation

```bash
# Run the built-in smoke-test CLI
emergo-demo

# Should print something like:
#   emergo-demo
#   Termination      : Converged
#   CE acceptance    : 68.2%
#   φ loss (first→last): 0.0112 → 0.0089
#   Final authority  :
#     agent_0: 0.623  |████████████        |
#   ...
#   Installation OK.
```

---

## 2. Running Tests

```bash
# Full suite (396 tests, ~30s)
pytest

# Verbose output
pytest -v

# Single test file
pytest tests/test_invariants.py -v

# Run a single test by keyword
pytest -k "test_rank_penalty" -v

# With coverage report
pytest --cov=emergo --cov-report=term-missing

# Property-based tests (hypothesis)
pytest tests/test_property.py -v

# Stress/failure-mode tests
pytest tests/test_stress.py -v -s
```

### Test organization

| File | What it covers |
|---|---|
| `test_invariants.py` | All 10 system invariants (INV-1 … INV-10) |
| `test_core_theorem.py` | C1 Determinism, C2 Stability, C3 Identifiability |
| `test_phi_update.py` | φ gradient descent, rank regularization, Adam |
| `test_features.py` | Feature extractor shape/NaN/bounds guarantees |
| `test_error_computation.py` | Differentiated per-agent error logic |
| `test_observer.py` | Observer isolation (INV-10) |
| `test_planner.py` | SequentialPlanner, DependencyPlanner |
| `test_coordinator.py` | MultiAgentCoordinator serialization (INV-9) |
| `test_lux_bridge_contract.py` | LuxBridge contract (parametrized SimulatedLuxBridge) |
| `test_property.py` | Hypothesis property-based tests |
| `test_benchmark.py` | 10/20/50-agent wall-time regression tests |
| `test_stress.py` | 8 failure-mode detectors |
| `test_integration.py` | End-to-end pipeline tests |

---

## 3. Code Style and Linting

The project uses **ruff** (linter + import sorter), **black** (formatter), and
**mypy** (type checker). Configs live in `pyproject.toml`.

```bash
# Format code (black)
black emergo/ tests/

# Lint + auto-fix safe issues (ruff)
ruff check emergo/ tests/ --fix

# Type checking
mypy emergo/
```

### Style rules in brief

- **Line length**: 100 characters (black enforces this).
- **Type annotations**: All public functions must be annotated. Private helpers
  (`_foo`) should be annotated too, but mypy won't fail if they're not yet.
- **Comments**: Write comments only for **why**, not **what**. Well-named
  functions and variables are self-documenting. No multi-line comment blocks.
- **Docstrings**: One short line for simple functions. For complex functions,
  use the module-level docstring pattern in `phi_update.py` as reference.
- **No emojis** in source files unless explicitly requested.

### Pre-commit (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

Add `.pre-commit-config.yaml` with ruff + black hooks for automatic checks
before every commit.

---

## 4. Running Examples

```bash
# Convergence demo (argparse CLI, ring graph)
python examples/convergence_demo.py
python examples/convergence_demo.py --agents 8 --iterations 1000 --optimizer adam

# Custom proposal generator comparison
python examples/custom_proposal_generator.py

# Long-horizon stress test (no plots if matplotlib not installed)
python examples/long_horizon_stress_test.py --no-plots

# 10k-iteration benchmark
python examples/emergo_10k_stress_test.py --seed 42 --no-plots
```

Or via the installed CLI commands:

```bash
emergo-demo                          # quick 5-agent smoke-test
emergo-kernel --agents 8 --iterations 1000 --optimizer adam
emergo-health --agents 5 --iterations 200
```

---

## 5. How to Add a New Planner

A **Planner** converts a `Goal` into a sequenced list of `Task` objects.

### Step 1 — Implement the `Planner` protocol

```python
# emergo/planner.py  (add alongside SequentialPlanner / DependencyPlanner)

class PriorityPlanner:
    """Planner that orders steps by explicit priority weights."""

    def __init__(self, steps: list[tuple[str, str, float]]) -> None:
        # (step_id, description, priority)
        ...

    def plan(self, goal: Goal, state: State) -> list[Task]:
        """Return tasks sorted by descending priority."""
        ...
```

The `Planner` protocol requires only `.plan(goal, state) -> list[Task]`.  
Check `emergo/planner.py` for the exact protocol definition.

### Step 2 — Export from `__init__.py`

```python
# emergo/__init__.py
from emergo.planner import PriorityPlanner
```

Add `"PriorityPlanner"` to the `__all__` list.

### Step 3 — Write tests

```python
# tests/test_planner.py
class TestPriorityPlanner:
    def test_orders_by_priority(self): ...
    def test_equal_priority_deterministic(self): ...
    def test_empty_steps_returns_empty(self): ...
```

**Checklist for new planners:**
- [ ] `.plan()` is deterministic for the same `(goal, state)` input
- [ ] Returns an empty list (not raises) for empty steps
- [ ] Does not mutate `goal` or `state`
- [ ] Works with `Executor` end-to-end (see `test_integration.py`)

---

## 6. How to Add a New Observer

Observers are read-only listeners attached to the kernel loop.  
They satisfy **INV-10**: exceptions in observers never propagate to the kernel.

### Step 1 — Implement the `KernelObserver` protocol

```python
# emergo/observer.py  (add alongside LoggingObserver / HistoryObserver)

from emergo.observer import _NoOpMixin

class ThrottleObserver(_NoOpMixin):
    """Observer that triggers a callback when φ loss exceeds a threshold."""

    def __init__(self, threshold: float, callback) -> None:
        self._threshold = threshold
        self._callback = callback

    def on_phi_updated(self, iteration: int, phi_loss: float) -> None:
        if phi_loss > self._threshold:
            self._callback(iteration, phi_loss)
```

`_NoOpMixin` provides no-op implementations of every callback so you only
override what you care about. The four callbacks are:

| Method | When it fires | Arguments |
|---|---|---|
| `on_iteration_start` | Before CE sampling | `(iteration, state)` |
| `on_ce_result` | After CE accept/reject | `(iteration, ce, success, errors_or_None)` |
| `on_phi_updated` | After phi_update | `(iteration, phi_loss)` |
| `on_kernel_done` | On termination | `(reason, final_state, n_iterations)` |

### Step 2 — Export and write tests

Same pattern as Planners: add to `__init__.py`, add `test_observer.py` tests.

**Checklist for new observers:**
- [ ] Does **not** modify any argument passed to it (all inputs are read-only)
- [ ] Does **not** raise — any exception must be caught internally
- [ ] Has a `reset()` method if it accumulates state (for reuse between runs)
- [ ] Tested for INV-10 isolation: `fire_observers([bad_observer], ...)` must not raise

---

## 7. How to Add a New Proposal Generator

A **ProposalGenerator** controls how Coordination Events are sampled.

### Step 1 — Implement the `ProposalGenerator` protocol

```python
# emergo/proposal.py  (add alongside DefaultProposalGenerator etc.)

class TopAuthorityProposalGenerator:
    """Always proposes add_edge from the highest-authority agent."""

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> Optional[CoordinationEvent]:
        if G_t.n_agents < 2:
            return None
        best = max(A_t.scores, key=A_t.get)
        ...
```

Return `None` to skip this iteration (kernel's inner loop calls `continue`).

### Step 2 — Export + tests

**Checklist:**
- [ ] Returns `None` gracefully when the graph is empty or degenerate
- [ ] Does not modify `A_t` or `G_t`
- [ ] Tested with `WeightedMixGenerator` to verify composability

---

## 8. Invariant Testing Guidelines

The ten system invariants (INV-1 … INV-10) are the contract that Emergo upholds.
Every PR that touches core components **must** keep all invariant tests green.

### Running invariant tests

```bash
pytest tests/test_invariants.py -v
```

### Adding tests for a new invariant

1. Add a new class `TestInvariantN<Name>` to `tests/test_invariants.py`.
2. Reference the invariant by ID in the class docstring.
3. Write at minimum:
   - A **positive test** (system upholds the invariant under normal operation)
   - A **boundary test** (invariant holds at the edge of valid input)
   - A **stress test** (invariant holds after 50+ kernel iterations)

### Invariant reference

| ID | Invariant | Key test |
|---|---|---|
| INV-1 | State is `(G, φ, A, E)` — pure-function pipeline | `test_state_is_immutable_tuple` |
| INV-2 | `error → authority → topology` feedback closed | `test_authority_influences_ce_sampling_direction` |
| INV-3 | Rejected CE leaves state unchanged | `test_ce_rejection_leaves_state_unchanged` |
| INV-4 | Sequential, deterministic, no race conditions | `test_kernel_is_deterministic` |
| INV-5 | Only Lux grants capabilities | `test_emergo_never_grants_capabilities_directly` |
| INV-6 | Pre-charge + refund on failure | `test_resource_refund_on_failure` |
| INV-7 | Every CE attempt is audited | `test_audit_trail_is_complete` |
| INV-8 | Decomposition depth is bounded | `test_max_decomposition_depth_enforced` |
| INV-9 | Coordinator serializes by authority | `test_coordinator_respects_authority_ordering` |
| INV-10 | Observer exceptions never propagate | `test_observer_exception_does_not_crash_kernel` |

### φ entanglement invariant (additional)

`rank(W_phi) ≥ d_latent // 2` must hold after any `phi_update` call with
`rank_lambda > 0`. This is tested in `test_phi_update.py::test_rank_penalty_prevents_collapse`.

When adding new `phi_update` optimizer paths, add a corresponding rank test.

### Property-based testing

For functions with complex input spaces, use `hypothesis`:

```python
# tests/test_property.py
from hypothesis import given, settings
from hypothesis import strategies as st

@given(st.integers(min_value=2, max_value=20))
def test_feature_extractor_shape_invariant(n_agents: int) -> None:
    ...
```

---

## 9. Pull Request Checklist

Before opening a PR, verify:

- [ ] `pytest` passes with zero failures
- [ ] `ruff check emergo/ tests/` has no errors
- [ ] `mypy emergo/` has no new type errors
- [ ] New public API is exported in `emergo/__init__.py` and `__all__`
- [ ] Tests added for new behaviour (aim for ≥ 3 tests per new function)
- [ ] Invariant tests still pass if any core module was changed
- [ ] `CONVERGENCE_ANALYSIS.md` or `SPECIFICATION.md` updated if behaviour changed
- [ ] Docstring updated for modified functions
- [ ] No secrets or credentials committed

### Commit style

Use the conventional-commits format:

```
feat(proposal): add TopAuthorityProposalGenerator
fix(phi_update): prevent NaN in rank penalty when W_phi is all-zeros
docs(contributing): add planner / observer how-to sections
test(invariants): add INV-11 for new capability scope rule
refactor(features): simplify _structural_metrics helper
```

---

*Questions? Open a GitHub issue or start a Discussion thread.*
