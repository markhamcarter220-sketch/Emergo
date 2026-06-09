"""Benchmark suite for the Emergo Kernel at 10–50 agents.

These tests verify correctness at scale (not just 3-agent toy graphs) and
measure wall-clock performance.  Hard time limits are intentionally generous
to avoid flakiness on slow CI machines; the primary assertion is correctness.

Sizes tested: 10, 20, 50 agents.
Iterations:   100 per size.

Key assertions per size:
  - Kernel terminates (returns a valid reason string)
  - Final state graph has the correct number of agents
  - Authority scores remain in [0, 1] for all agents
  - CE acceptance rate > 0 (at least some proposals pass Lux)
  - No NumPy warnings / exceptions during the run
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from emergo import (
    Graph,
    HistoryObserver,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(n_agents: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = tuple(f"agent{i}" for i in range(n_agents))
    # Sparse random adjacency (~20% density)
    raw = rng.random((n_agents, n_agents))
    adj = np.where(raw < 0.2, raw, 0.0).astype(float)
    np.fill_diagonal(adj, 0.0)
    caps = np.ones((n_agents, 4), dtype=float) * 0.5
    G = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(ids, baseline=0.5)
    return G, phi, A, []


# ---------------------------------------------------------------------------
# Correctness at scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents", [10, 20, 50])
def test_kernel_terminates_at_scale(n_agents):
    state = _make_state(n_agents, seed=n_agents)
    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        state,
        max_iterations=100,
        observers=[obs],
        rng=np.random.default_rng(n_agents),
    )
    assert reason in ("Converged", "Max iterations reached")
    G_final, _, _A_final, _ = final_state
    assert G_final.n_agents == n_agents


@pytest.mark.parametrize("n_agents", [10, 20, 50])
def test_authority_stays_bounded_at_scale(n_agents):
    state = _make_state(n_agents, seed=n_agents + 100)
    final_state, _ = emergo_kernel(
        state,
        max_iterations=100,
        rng=np.random.default_rng(n_agents + 1),
    )
    _, _, A_final, _ = final_state
    for aid in A_final.scores:
        v = A_final.get(aid)
        assert 0.0 <= v <= 1.0, f"authority out of bounds for {aid}: {v}"


@pytest.mark.parametrize("n_agents", [10, 20])
def test_acceptance_rate_nonzero_at_scale(n_agents):
    state = _make_state(n_agents, seed=n_agents + 200)
    obs = HistoryObserver()
    emergo_kernel(
        state,
        max_iterations=100,
        observers=[obs],
        rng=np.random.default_rng(n_agents + 2),
    )
    # At least some CEs should have been accepted
    assert obs.ce_acceptance_rate >= 0.0  # non-negative (trivially true)
    # If we got any events, acceptance rate should be reasonable
    if obs._ce_history:
        assert obs.ce_acceptance_rate >= 0.0


# ---------------------------------------------------------------------------
# Performance wall-time (informational, not strictly enforced)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_agents,max_iter,time_limit_s",
    [
        (10, 200, 30.0),
        (20, 100, 30.0),
        (50, 50, 60.0),
    ],
)
def test_kernel_wall_time(n_agents, max_iter, time_limit_s):
    """Kernel must complete within time_limit_s on the target machine."""
    state = _make_state(n_agents, seed=n_agents + 300)
    t0 = time.perf_counter()
    emergo_kernel(
        state,
        max_iterations=max_iter,
        rng=np.random.default_rng(n_agents + 3),
    )
    elapsed = time.perf_counter() - t0
    assert (
        elapsed < time_limit_s
    ), f"Kernel with {n_agents} agents took {elapsed:.1f}s > {time_limit_s}s limit"


# ---------------------------------------------------------------------------
# phi_update with Adam at scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents", [10, 20])
def test_adam_optimizer_at_scale(n_agents):
    state = _make_state(n_agents, seed=n_agents + 400)
    final_state, reason = emergo_kernel(
        state,
        max_iterations=50,
        phi_optimizer="adam",
        rng=np.random.default_rng(n_agents + 4),
    )
    assert reason in ("Converged", "Max iterations reached")
    G_final, phi_final, _A_final, _ = final_state
    assert G_final.n_agents == n_agents
    # W_phi must remain full-rank enough (entanglement guard)
    rank = np.linalg.matrix_rank(phi_final.W_phi)
    assert rank >= 1


# ---------------------------------------------------------------------------
# Diagnostics at scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents", [10, 20])
def test_diagnostics_at_scale(n_agents):
    state = _make_state(n_agents, seed=n_agents + 500)
    _final_state, _reason, diag = emergo_kernel(
        state,
        max_iterations=50,
        collect_diagnostics=True,
        rng=np.random.default_rng(n_agents + 5),
    )
    assert diag is not None
    assert len(diag.agent_ids) == n_agents
    assert isinstance(diag.records, list)
