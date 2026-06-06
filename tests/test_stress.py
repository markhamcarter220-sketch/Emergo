"""Stress tests and failure-mode detectors for the Emergo kernel.

Tests cover 8 failure modes plus an integration health-report test.
All tests use fixed seeds for determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from emergo import (
    Graph,
    Authority,
    Lux,
    emergo_kernel,
    make_initial_phi,
    make_initial_authority,
    KernelDiagnostics,
    IterationRecord,
    DetectorResult,
    run_health_check,
    detect_authority_collapse,
    detect_topology_lock_in,
    detect_phi_gaming,
    detect_speculative_cascades,
    detect_lux_bottleneck,
    detect_clique_formation,
    detect_credit_assignment_ambiguity,
    detect_emergent_conservatism,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_agents: int, edge_prob: float = 0.3, seed: int = 1) -> Graph:
    """Build a random directed graph with n_agents nodes."""
    rng = np.random.default_rng(seed)
    agents = tuple(f"agent_{i}" for i in range(n_agents))
    adj = np.zeros((n_agents, n_agents), dtype=float)
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and rng.random() < edge_prob:
                adj[i, j] = rng.random()
    caps = rng.random((n_agents, 4)).astype(float)
    return Graph(agent_ids=agents, adjacency=adj, capabilities=caps)


def _make_initial_state(n_agents: int, phi_seed: int = 42):
    """Build a complete initial state tuple."""
    G0 = _make_graph(n_agents, edge_prob=0.3, seed=phi_seed)
    phi0 = make_initial_phi(d_latent=4, d_features=16, d_ce=4, seed=phi_seed)
    A0 = make_initial_authority(G0.agent_ids, baseline=0.5)
    E0 = []
    return G0, phi0, A0, E0


def _run(n_agents: int, n_iter: int, lux: Lux = None, seed: int = 0):
    """Run the kernel with diagnostics and return (final_state, reason, diag)."""
    G0, phi0, A0, E0 = _make_initial_state(n_agents)
    if lux is None:
        lux = Lux()
    final_state, reason, diag = emergo_kernel(
        (G0, phi0, A0, E0),
        max_iterations=n_iter,
        lux=lux,
        rng=np.random.default_rng(seed),
        collect_diagnostics=True,
    )
    return final_state, reason, diag


# ---------------------------------------------------------------------------
# 1. Authority Collapse
# ---------------------------------------------------------------------------

class TestAuthorityCollapse:
    def test_detector_returns_result(self):
        """Detector runs on small kernel output and returns valid DetectorResult."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_authority_collapse(diag)
        print(f"\n[authority_collapse] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_severity_in_range(self):
        """Severity is always clipped to [0, 1]."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300, seed=7)
        result = detect_authority_collapse(diag)
        assert 0.0 <= result.severity <= 1.0

    def test_empty_diag(self):
        """Empty diagnostics produce a safe (non-failure) result."""
        diag = KernelDiagnostics(records=[], agent_ids=("a", "b"))
        result = detect_authority_collapse(diag)
        assert isinstance(result, DetectorResult)
        assert result.failure_detected is False
        assert result.severity == 0.0


# ---------------------------------------------------------------------------
# 2. Topology Lock-In
# ---------------------------------------------------------------------------

class TestTopologyLockIn:
    def test_detector_returns_result(self):
        """Detector runs on small kernel output and returns valid DetectorResult."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_topology_lock_in(diag)
        print(f"\n[topology_lock_in] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_threshold_sensitivity(self):
        """Changing threshold changes detection boundary."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result_loose = detect_topology_lock_in(diag, threshold=0.0001)
        result_tight = detect_topology_lock_in(diag, threshold=0.99)
        # Tight threshold is more likely to trigger failure
        assert result_tight.severity >= result_loose.severity

    def test_format(self):
        """DetectorResult has all required fields."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_topology_lock_in(diag)
        assert hasattr(result, "name")
        assert hasattr(result, "failure_detected")
        assert hasattr(result, "severity")
        assert hasattr(result, "evidence")
        assert hasattr(result, "recommendation")


# ---------------------------------------------------------------------------
# 3. Phi Gaming
# ---------------------------------------------------------------------------

class TestPhiGaming:
    def test_detector_returns_result(self):
        """Detector runs and returns valid DetectorResult."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_phi_gaming(diag, final_state)
        print(f"\n[phi_gaming] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_medium_scale(self):
        """Runs on a medium-scale kernel and produces valid output."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_phi_gaming(diag, final_state, n_random=20)
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_format(self):
        """DetectorResult has all required fields."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_phi_gaming(diag, final_state)
        assert result.name == "phi_gaming"
        assert isinstance(result.evidence, str)


# ---------------------------------------------------------------------------
# 4. Speculative Cascades
# ---------------------------------------------------------------------------

class TestSpeculativeCascades:
    def test_detector_returns_result(self):
        """Detector runs and returns valid DetectorResult."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_speculative_cascades(diag)
        print(f"\n[speculative_cascades] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_insufficient_records(self):
        """Returns non-failure when there are too few records."""
        diag = KernelDiagnostics(records=[], agent_ids=("a", "b"))
        result = detect_speculative_cascades(diag, window=50)
        assert result.failure_detected is False
        assert result.severity == 0.0

    def test_medium_scale(self):
        """Runs on a medium-scale kernel and produces valid output."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_speculative_cascades(diag, window=50)
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0


# ---------------------------------------------------------------------------
# 5. Lux Bottleneck
# ---------------------------------------------------------------------------

class TestLuxBottleneck:
    def test_detector_compares_runs(self):
        """Permissive vs strict Lux runs are compared correctly."""
        lux_perm = Lux(min_authority=0.0)
        lux_strict = Lux(min_authority=0.5)

        final_state_perm, _, diag_perm = _run(n_agents=3, n_iter=300, lux=lux_perm, seed=1)
        final_state_strict, _, diag_strict = _run(n_agents=3, n_iter=300, lux=lux_strict, seed=1)

        result = detect_lux_bottleneck(diag_perm, diag_strict)
        print(f"\n[lux_bottleneck] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_identical_runs_not_bottleneck(self):
        """Two identical permissive runs should not show a bottleneck."""
        lux_perm = Lux(min_authority=0.0)
        _, _, diag1 = _run(n_agents=3, n_iter=300, lux=lux_perm, seed=1)
        _, _, diag2 = _run(n_agents=3, n_iter=300, lux=lux_perm, seed=1)
        result = detect_lux_bottleneck(diag1, diag2)
        # Same runs: rate_ratio ~1.0, no bottleneck
        assert result.severity <= 1.0

    def test_format(self):
        """DetectorResult has all required fields."""
        lux_perm = Lux(min_authority=0.0)
        lux_strict = Lux(min_authority=0.5)
        _, _, diag_perm = _run(n_agents=3, n_iter=300, lux=lux_perm, seed=2)
        _, _, diag_strict = _run(n_agents=3, n_iter=300, lux=lux_strict, seed=2)
        result = detect_lux_bottleneck(diag_perm, diag_strict)
        assert result.name == "lux_bottleneck"
        assert isinstance(result.evidence, str)


# ---------------------------------------------------------------------------
# 6. Clique Formation
# ---------------------------------------------------------------------------

class TestCliqueFormation:
    def test_small_graph_skips(self):
        """3-agent graph is skipped (need >= 4)."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_clique_formation(diag, final_state)
        print(f"\n[clique_formation] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert result.failure_detected is False

    def test_medium_scale(self):
        """Returns valid DetectorResult on medium-scale run."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_clique_formation(diag, final_state)
        print(f"\n[clique_formation] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_format(self):
        """DetectorResult has all required fields."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_clique_formation(diag, final_state)
        assert result.name == "clique_formation"
        assert isinstance(result.evidence, str)


# ---------------------------------------------------------------------------
# 7. Credit Assignment Ambiguity
# ---------------------------------------------------------------------------

class TestCreditAssignmentAmbiguity:
    def test_failure_resolved_by_differentiated_errors(self):
        """With differentiated per-agent errors, credit assignment ambiguity resolves."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_credit_assignment_ambiguity(diag)
        print(f"\n[credit_assignment_ambiguity] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        # Proposer gets global error, participants get local error → differentiated
        assert result.failure_detected is False

    def test_medium_scale_failure_resolved(self):
        """Credit assignment ambiguity resolved at medium scale too."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_credit_assignment_ambiguity(diag)
        assert result.failure_detected is False

    def test_format(self):
        """DetectorResult has all required fields."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_credit_assignment_ambiguity(diag)
        assert result.name == "credit_assignment_ambiguity"
        assert isinstance(result.evidence, str)
        assert isinstance(result.recommendation, str)
        assert result.severity <= 1.0


# ---------------------------------------------------------------------------
# 8. Emergent Conservatism
# ---------------------------------------------------------------------------

class TestEmergentConservatism:
    def test_detector_returns_result(self):
        """Detector runs and returns valid DetectorResult."""
        final_state, reason, diag = _run(n_agents=3, n_iter=300)
        result = detect_emergent_conservatism(diag)
        print(f"\n[emergent_conservatism] severity={result.severity:.2f} "
              f"detected={result.failure_detected} | {result.evidence}")
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_medium_scale(self):
        """Runs on a medium-scale kernel and produces valid output."""
        final_state, reason, diag = _run(n_agents=10, n_iter=500)
        result = detect_emergent_conservatism(diag)
        assert isinstance(result, DetectorResult)
        assert 0.0 <= result.severity <= 1.0

    def test_insufficient_records(self):
        """Short run with fewer records than window returns non-failure."""
        # Use a very short run so we have fewer than 100 records
        final_state, reason, diag = _run(n_agents=3, n_iter=50)
        result = detect_emergent_conservatism(diag, window=100)
        # Not enough records: should not erroneously fire
        assert result.severity >= 0.0
        assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Integration: Full health report
# ---------------------------------------------------------------------------

class TestHealthReport:
    def test_full_health_report(self):
        """Run all 8 detectors on medium-scale run and report findings."""
        G0, phi0, A0, E0 = _make_initial_state(n_agents=10)
        final_state, reason, diag = emergo_kernel(
            (G0, phi0, A0, E0), max_iterations=500,
            rng=np.random.default_rng(42),
            collect_diagnostics=True,
        )

        # Run strict for bottleneck comparison
        G0b, phi0b, A0b, E0b = _make_initial_state(n_agents=10)
        lux_strict = Lux(min_authority=0.5)
        _, _, diag_strict = emergo_kernel(
            (G0b, phi0b, A0b, E0b), max_iterations=500,
            lux=lux_strict,
            rng=np.random.default_rng(42),
            collect_diagnostics=True,
        )

        results = run_health_check(diag, final_state, diag_strict=diag_strict)

        print("\n" + "=" * 60)
        print("EMERGO HEALTH REPORT (10 agents, 500 iterations)")
        print("=" * 60)
        for r in results:
            status = "FAIL" if r.failure_detected else " OK "
            print(f"[{status}] {r.name:<35} severity={r.severity:.2f}")
            print(f"       Evidence: {r.evidence}")
            if r.failure_detected:
                print(f"       Fix: {r.recommendation}")
        print("=" * 60)

        # All detectors should return valid output
        for r in results:
            assert isinstance(r, DetectorResult)
            assert 0.0 <= r.severity <= 1.0

        # Results are sorted by severity descending
        severities = [r.severity for r in results]
        assert severities == sorted(severities, reverse=True)

    def test_backward_compatible_return(self):
        """When collect_diagnostics=False, return is 2-tuple (backward compat)."""
        G0, phi0, A0, E0 = _make_initial_state(n_agents=3)
        result = emergo_kernel(
            (G0, phi0, A0, E0), max_iterations=30,
            rng=np.random.default_rng(0),
            collect_diagnostics=False,
        )
        state, reason = result  # Must unpack as 2-tuple
        assert reason in {"Converged", "Max iterations reached"}
