"""Large-scale stress tests: 50-100 agents, varied topologies, failure injection."""

from __future__ import annotations

import numpy as np

from emergo import (
    CoordinationEvent,
    Graph,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.proposal import DefaultProposalGenerator, SequenceProposalGenerator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _random_graph(n, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    adj = rng.uniform(0.1, 0.8, (n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(adj, 0.0)
    caps = rng.uniform(0.2, 0.8, (n, 4))
    return Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)


def _scale_free_graph(n, seed=0):
    """Barabasi-Albert attachment (simplified): each new node connects to 2 existing."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    degrees = np.ones(n)
    for i in range(2, n):
        probs = degrees[:i] / degrees[:i].sum()
        targets = rng.choice(i, size=min(2, i), replace=False, p=probs)
        for t in targets:
            adj[i, t] = rng.uniform(0.2, 0.8)
            degrees[t] += 1
        degrees[i] += len(targets)
    caps = rng.uniform(0.2, 0.8, (n, 4))
    return Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)


def _make_state(G: Graph, seed: int = 0):
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    return G, phi, A, []


# ---------------------------------------------------------------------------
# Class 1: Large-scale convergence
# ---------------------------------------------------------------------------


class TestLargeScaleConvergence:

    def test_50_agent_ring_convergence(self):
        """50 agents in a ring topology, 500 iterations."""
        n = 50
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 0.5
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=1)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(1),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        max_auth = max(A.get(aid) for aid in G.agent_ids)
        assert max_auth <= 1.0, f"Authority exceeded 1.0: {max_auth}"

    def test_100_agent_sparse_graph(self):
        """100 agents, random sparse adjacency (density 0.05), 300 iterations."""
        n = 100
        G = _random_graph(n, density=0.05, seed=2)
        state = _make_state(G, seed=2)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=300,
            rng=np.random.default_rng(2),
        )
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        for aid in G.agent_ids:
            v = A.get(aid)
            assert 0.0 <= v <= 1.0, f"Authority out of bounds for {aid}: {v}"

    def test_50_agent_dense_graph(self):
        """50 agents, dense adjacency (density 0.8), 300 iterations."""
        n = 50
        G = _random_graph(n, density=0.8, seed=3)
        state = _make_state(G, seed=3)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=300,
            rng=np.random.default_rng(3),
        )
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        for aid in G.agent_ids:
            v = A.get(aid)
            assert 0.0 <= v <= 1.0, f"Authority out of bounds for {aid}: {v}"

    def test_scale_free_topology(self):
        """Scale-free graph with 10 agents, 200 iterations. Assert convergence, no collapse."""
        n = 10
        G = _scale_free_graph(n, seed=4)
        state = _make_state(G, seed=4)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=200,
            rng=np.random.default_rng(4),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        # No total authority collapse
        mean_auth = np.mean([A.get(aid) for aid in G.agent_ids])
        assert mean_auth > 0.0, "Total authority collapsed to zero"


# ---------------------------------------------------------------------------
# Class 2: Varied topologies
# ---------------------------------------------------------------------------


class TestVariedTopologies:

    def test_ring_topology(self):
        """20 agents, ring, 400 iters. CE acceptance rate > 10%."""
        n = 20
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 0.5
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=10)

        from emergo import HistoryObserver

        obs = HistoryObserver()
        emergo_kernel(
            state,
            max_iterations=400,
            rng=np.random.default_rng(10),
            observers=[obs],
        )
        # At least some CEs should be accepted
        if obs._ce_history:
            n_accepted = sum(1 for _, ok in obs._ce_history if ok)
            total = len(obs._ce_history)
            rate = n_accepted / total if total > 0 else 0.0
            assert rate > 0.10, f"CE acceptance rate too low: {rate:.2%}"

    def test_complete_graph_topology(self):
        """10 agents, fully connected (all edges weight 0.5), 300 iters. No collapse."""
        n = 10
        adj = np.full((n, n), 0.5)
        np.fill_diagonal(adj, 0.0)
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=11)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=300,
            rng=np.random.default_rng(11),
        )
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        scores = [A.get(aid) for aid in G.agent_ids]
        assert max(scores) > 0.1, "Complete graph collapsed all authority"

    def test_star_topology(self):
        """15 agents, star (agent_0 connected to all others), 300 iters."""
        n = 15
        adj = np.zeros((n, n))
        for i in range(1, n):
            adj[0, i] = 0.5
            adj[i, 0] = 0.3
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=12)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=300,
            rng=np.random.default_rng(12),
        )
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        # High-degree center should not monopolize: other agents must have >0 authority
        non_center_scores = [A.get(f"a{i}") for i in range(1, n)]
        assert max(non_center_scores) > 0.0, "Star center monopolized all authority"

    def test_disconnected_components(self):
        """20 agents, two separate 10-agent rings (no edges between groups), 400 iters."""
        n = 20
        adj = np.zeros((n, n))
        # Component 1: agents 0-9 form a ring
        for i in range(10):
            adj[i, (i + 1) % 10] = 0.5
        # Component 2: agents 10-19 form a ring
        for i in range(10, 20):
            adj[i, 10 + (i - 10 + 1) % 10] = 0.5
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=13)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=400,
            rng=np.random.default_rng(13),
        )
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf"
        comp1_scores = [A.get(f"a{i}") for i in range(10)]
        comp2_scores = [A.get(f"a{i}") for i in range(10, 20)]
        assert np.mean(comp1_scores) > 0.0, "Component 1 authority collapsed"
        assert np.mean(comp2_scores) > 0.0, "Component 2 authority collapsed"

    def test_dynamic_agent_join(self):
        """Start with 10 agents, use add_agent CEs to add 5 more mid-run."""
        n = 10
        G = _random_graph(n, density=0.3, seed=14)
        state = _make_state(G, seed=14)

        # Build a sequence of add_agent CEs for 5 new agents
        add_ces = [
            CoordinationEvent(
                event_type="add_agent",
                participants=(f"a{n + k}",),
                params=frozenset(
                    [("agent_id", f"a{n + k}"), ("capabilities", (0.5, 0.5, 0.5, 0.5))]
                ),
            )
            for k in range(5)
        ]
        seq_gen = SequenceProposalGenerator(add_ces, loop=False)
        default_gen = DefaultProposalGenerator()

        class _MixGen:
            def __init__(self):
                self._seq = seq_gen
                self._default = default_gen

            def propose(self, A_t, G_t, rng):
                ce = self._seq.propose(A_t, G_t, rng)
                if ce is not None:
                    return ce
                return self._default.propose(A_t, G_t, rng)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=300,
            rng=np.random.default_rng(14),
            proposal_generator=_MixGen(),
        )
        assert reason in {"Converged", "Max iterations reached"}
        G_final, _phi, A_final, _ = final_state
        assert G_final.n_agents >= n, "Final graph has fewer agents than initial"
        for aid in G_final.agent_ids:
            v = A_final.get(aid)
            assert 0.0 <= v <= 1.0, f"Authority out of bounds for {aid}: {v}"


# ---------------------------------------------------------------------------
# Class 3: Failure injection
# ---------------------------------------------------------------------------


class TestFailureInjection:

    def test_noisy_predictions_stable(self):
        """50 agents, high phi_lr=0.01. Assert no NaN/Inf, kernel completes."""
        n = 50
        G = _random_graph(n, density=0.2, seed=20)
        state = _make_state(G, seed=20)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=300,
            phi_lr=0.01,
            rng=np.random.default_rng(20),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, phi, _A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi contains NaN/Inf under noisy lr"
        assert np.all(np.isfinite(phi.W_F)), "W_F contains NaN/Inf under noisy lr"

    def test_partial_observability_null_proposals(self):
        """Custom generator that returns None 80% of the time."""
        n = 30
        G = _random_graph(n, density=0.3, seed=21)
        state = _make_state(G, seed=21)

        default_gen = DefaultProposalGenerator()

        class _NullMostlyGen:
            def propose(self, A_t, G_t, rng):
                if rng.random() < 0.80:
                    return None
                return default_gen.propose(A_t, G_t, rng)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=1000,
            rng=np.random.default_rng(21),
            proposal_generator=_NullMostlyGen(),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, phi, A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi NaN under partial observability"
        for aid in G.agent_ids:
            v = A.get(aid)
            assert 0.0 <= v <= 1.0

    def test_malicious_agent_self_loops(self):
        """10-agent graph. SequenceProposalGenerator repeatedly proposes add_edge(a0, a0)."""
        n = 10
        G = _random_graph(n, density=0.3, seed=22)
        state = _make_state(G, seed=22)

        self_loop_ce = CoordinationEvent(
            event_type="add_edge",
            participants=("a0", "a0"),
            params=frozenset([("weight", 0.5)]),
        )
        seq_gen = SequenceProposalGenerator([self_loop_ce] * 100, loop=True)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=200,
            rng=np.random.default_rng(22),
            proposal_generator=seq_gen,
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, _phi, A, _ = final_state
        for aid in G.agent_ids:
            v = A.get(aid)
            assert 0.0 <= v <= 1.0, f"Authority out of bounds for {aid}: {v}"

    def test_adversarial_all_same_proposal(self):
        """Generator always proposes same CE (add_edge between a0 and a1). 20 agents, 500 iters."""
        n = 20
        G = _random_graph(n, density=0.2, seed=23)
        state = _make_state(G, seed=23)

        same_ce = CoordinationEvent(
            event_type="add_edge",
            participants=("a0", "a1"),
            params=frozenset([("weight", 0.5)]),
        )

        class _SameGen:
            def propose(self, A_t, G_t, rng):
                return same_ce

        final_state, reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(23),
            proposal_generator=_SameGen(),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, _phi, A, _ = final_state
        scores = np.array([A.get(aid) for aid in G.agent_ids])
        assert np.std(scores) > 0.0, "All authority scores identical — no diversity"

    def test_zero_weight_edges_initial_graph(self):
        """Start with all-zero adjacency, 20 agents, 500 iters. Assert no crash."""
        n = 20
        adj = np.zeros((n, n))
        caps = np.ones((n, 4)) * 0.5
        G = Graph(agent_ids=tuple(f"a{i}" for i in range(n)), adjacency=adj, capabilities=caps)
        state = _make_state(G, seed=24)

        final_state, reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(24),
        )
        assert reason in {"Converged", "Max iterations reached"}
        _, phi, _A, _ = final_state
        assert np.all(np.isfinite(phi.W_phi)), "W_phi NaN with zero-weight initial graph"


# ---------------------------------------------------------------------------
# Class 4: Authority stability
# ---------------------------------------------------------------------------


class TestAuthorityStability:

    def test_authority_always_in_unit_interval(self):
        """50 agents, 500 iters, collect_diagnostics=True. All authority scores in [0,1]."""
        n = 50
        G = _random_graph(n, density=0.2, seed=30)
        state = _make_state(G, seed=30)

        _final_state, _reason, diag = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(30),
            collect_diagnostics=True,
        )
        for rec in diag.records:
            for aid, score in rec.authority_scores.items():
                assert (
                    0.0 <= score <= 1.0
                ), f"Authority out of [0,1] at iter {rec.iteration}, agent {aid}: {score}"

    def test_authority_not_always_decreasing(self):
        """20 agents, 400 iters. At least one agent has final authority >= initial (0.5)."""
        n = 20
        G = _random_graph(n, density=0.3, seed=31)
        state = _make_state(G, seed=31)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=400,
            rng=np.random.default_rng(31),
        )
        _, _phi, A, _ = final_state
        scores = [A.get(aid) for aid in G.agent_ids]
        assert max(scores) >= 0.5, f"All authority scores below baseline 0.5; max={max(scores):.4f}"

    def test_authority_diversity_maintained(self):
        """30 agents, 500 iters. std(final_authority_scores) > 0.01."""
        n = 30
        G = _random_graph(n, density=0.3, seed=32)
        state = _make_state(G, seed=32)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=500,
            rng=np.random.default_rng(32),
        )
        _, _phi, A, _ = final_state
        scores = np.array([A.get(aid) for aid in G.agent_ids])
        assert np.std(scores) > 0.01, f"Authority diversity too low: std={np.std(scores):.4f}"

    def test_min_authority_baseline_respected(self):
        """Regression test for authority collapse bug. After 200 iters, max authority > 0.3."""
        n = 10
        G = _random_graph(n, density=0.3, seed=33)
        state = _make_state(G, seed=33)

        final_state, _reason = emergo_kernel(
            state,
            max_iterations=200,
            rng=np.random.default_rng(33),
        )
        _, _phi, A, _ = final_state
        scores = [A.get(aid) for aid in G.agent_ids]
        assert max(scores) > 0.3, f"Authority collapsed below 0.3; max score={max(scores):.4f}"
