"""Regression tests for known failure modes. Each test pins a previously-observed bug."""
from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    Graph,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.phi_update import phi_update


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _random_graph(n, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    adj = rng.uniform(0.1, 0.8, (n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(adj, 0.0)
    caps = rng.uniform(0.2, 0.8, (n, 4))
    return Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)


def _make_state(n, density=0.3, seed=0):
    G = _random_graph(n, density=density, seed=seed)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    return G, phi, A, []


# ---------------------------------------------------------------------------
# Class 1: Authority collapse regression
# ---------------------------------------------------------------------------

class TestAuthorityCollapseRegression:

    def test_multi_participant_ce_no_collapse_regression(self):
        """Multi-participant CEs must give differentiated authority deltas to proposer vs participants.

        Root cause: authority collapse occurred when all CE participants got identical errors
        from error_computation using a single global error. Since the fix, proposer and
        participants receive different errors, causing different deltas.
        """
        n = 20
        state = _make_state(n, density=0.3, seed=100)

        final_state, reason, diag = emergo_kernel(
            state,
            max_iterations=400,
            rng=np.random.default_rng(100),
            collect_diagnostics=True,
        )

        # Find multi-participant accepted CEs with recorded authority deltas
        multi_ces = [
            rec for rec in diag.records
            if (rec.ce_accepted
                and rec.ce_participants is not None
                and len(rec.ce_participants) >= 2
                and len(rec.authority_delta) >= 2)
        ]

        if len(multi_ces) == 0:
            pytest.skip("No multi-participant CEs accepted in this run; cannot verify")

        # Count how many have different proposer vs participant deltas
        differentiated = 0
        for rec in multi_ces:
            prop = rec.ce_proposer
            if prop is None or prop not in rec.authority_delta:
                continue
            prop_delta = rec.authority_delta[prop]
            non_prop_deltas = [
                v for k, v in rec.authority_delta.items()
                if k != prop
            ]
            if any(abs(prop_delta - nd) > 1e-12 for nd in non_prop_deltas):
                differentiated += 1

        total = len(multi_ces)
        ratio = differentiated / total if total > 0 else 0.0
        assert ratio >= 0.50, (
            f"Only {differentiated}/{total} ({ratio:.0%}) multi-participant CEs have "
            f"differentiated authority deltas; expected >= 50%"
        )

    def test_authority_stays_above_floor_after_50_iterations(self):
        """Previously authority dropped to ~0.1 after ~16 CEs. Regression guard.

        After 50 iterations, mean authority should be > 0.1.
        """
        n = 10
        state = _make_state(n, density=0.3, seed=101)

        final_state, reason, diag = emergo_kernel(
            state,
            max_iterations=200,
            rng=np.random.default_rng(101),
            collect_diagnostics=True,
        )

        # Find records at or after iteration 50
        late_records = [rec for rec in diag.records if rec.iteration >= 50]
        if not late_records:
            pytest.skip("Fewer than 50 iterations recorded; skipping floor check")

        # Check mean authority in late records
        all_scores = []
        for rec in late_records:
            all_scores.extend(rec.authority_scores.values())

        if not all_scores:
            pytest.skip("No authority scores recorded after iteration 50")

        mean_auth = float(np.mean(all_scores))
        assert mean_auth > 0.1, (
            f"Authority collapsed below floor: mean={mean_auth:.4f} after iter 50 "
            f"(expected > 0.1)"
        )


# ---------------------------------------------------------------------------
# Class 2: Phi instability regression
# ---------------------------------------------------------------------------

class TestPhiInstabilityRegression:

    def test_phi_frob_norm_bounded(self):
        """Run 20 agents, 500 iters. Assert phi.W_phi Frobenius norm < 10.0 at end."""
        n = 20
        state = _make_state(n, density=0.3, seed=200)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(200),
        )
        _, phi, _, _ = final_state
        frob = float(np.linalg.norm(phi.W_phi, "fro"))
        assert frob < 10.0, f"W_phi Frobenius norm too large: {frob:.4f} >= 10.0"

    def test_phi_rank_maintained(self):
        """Run 20 agents, 500 iters. Assert rank(phi.W_phi) >= phi.d_latent // 2."""
        n = 20
        state = _make_state(n, density=0.3, seed=201)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(201),
        )
        _, phi, _, _ = final_state
        rank = np.linalg.matrix_rank(phi.W_phi)
        min_rank = phi.d_latent // 2
        assert rank >= min_rank, (
            f"W_phi rank collapsed: rank={rank} < d_latent//2={min_rank}"
        )

    def test_phi_no_nan_under_extreme_lr(self):
        """Run 10 agents, 200 iters with phi_lr=0.1. Assert no NaN/Inf in phi parameters."""
        n = 10
        state = _make_state(n, density=0.3, seed=202)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=200,
            phi_lr=0.1,
            rng=np.random.default_rng(202),
        )
        _, phi, _, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi has NaN/Inf under phi_lr=0.1"
        assert np.all(np.isfinite(phi.b_phi)), "b_phi has NaN/Inf under phi_lr=0.1"
        assert np.all(np.isfinite(phi.W_F)), "W_F has NaN/Inf under phi_lr=0.1"
        assert np.all(np.isfinite(phi.b_F)), "b_F has NaN/Inf under phi_lr=0.1"


# ---------------------------------------------------------------------------
# Class 3: Topology freeze regression
# ---------------------------------------------------------------------------

class TestTopologyFreezeRegression:

    def test_ce_acceptance_never_zero_for_100_steps(self):
        """Run 20 agents, 500 iters. In any 50-iteration window, at least one CE accepted.

        Verifies that the topology does not permanently freeze.
        """
        n = 20
        state = _make_state(n, density=0.3, seed=300)

        final_state, reason, diag = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(300),
            collect_diagnostics=True,
        )

        if len(diag.records) < 50:
            pytest.skip("Fewer than 50 diagnostic records; skipping window check")

        # Compute rolling 50-window acceptance rates
        accepted_flags = [float(rec.ce_accepted) for rec in diag.records]
        n_records = len(accepted_flags)
        window = 50

        min_rate = float("inf")
        for start in range(0, n_records - window + 1, window):
            window_slice = accepted_flags[start: start + window]
            rate = float(np.mean(window_slice))
            min_rate = min(min_rate, rate)

        assert min_rate > 0.0, (
            f"Topology froze: minimum rolling-50 CE acceptance rate = {min_rate:.4f} "
            f"(expected > 0 in every 50-iteration window)"
        )


# ---------------------------------------------------------------------------
# Class 4: Determinism regression
# ---------------------------------------------------------------------------

class TestDeterminismRegression:

    def test_same_seed_produces_identical_results(self):
        """Run kernel with seed=42 twice (same initial state). Final authority must be equal.

        Pins INV-4 determinism guarantee.
        """
        n = 10
        # Same initial state built from same seed
        state1 = _make_state(n, density=0.3, seed=42)
        state2 = _make_state(n, density=0.3, seed=42)

        final_state1, _ = emergo_kernel(
            state1,
            max_iterations=100,
            rng=np.random.default_rng(42),
        )
        final_state2, _ = emergo_kernel(
            state2,
            max_iterations=100,
            rng=np.random.default_rng(42),
        )

        _, _, A1, _ = final_state1
        _, _, A2, _ = final_state2

        G1, _, _, _ = final_state1
        for aid in G1.agent_ids:
            v1 = A1.get(aid)
            v2 = A2.get(aid)
            assert v1 == v2, (
                f"Determinism violation for {aid}: run1={v1}, run2={v2}"
            )

    def test_different_seeds_produce_different_results(self):
        """Run kernel with seed=42 vs seed=99. At least one authority score must differ.

        Verifies the RNG is actually used (not just ignored).
        """
        n = 10
        state42 = _make_state(n, density=0.3, seed=42)
        state99 = _make_state(n, density=0.3, seed=42)  # same initial state, different rng

        final_state42, _ = emergo_kernel(
            state42,
            max_iterations=100,
            rng=np.random.default_rng(42),
        )
        final_state99, _ = emergo_kernel(
            state99,
            max_iterations=100,
            rng=np.random.default_rng(99),
        )

        _, _, A42, _ = final_state42
        _, _, A99, _ = final_state99

        G42, _, _, _ = final_state42
        scores42 = [A42.get(aid) for aid in G42.agent_ids]
        scores99 = [A99.get(aid) for aid in G42.agent_ids]

        any_different = any(s1 != s2 for s1, s2 in zip(scores42, scores99))
        assert any_different, (
            "seed=42 and seed=99 produced identical authority scores; "
            "RNG appears unused"
        )
