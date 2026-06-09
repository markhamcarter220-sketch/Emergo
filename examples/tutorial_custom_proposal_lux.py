#!/usr/bin/env python3
"""Tutorial: Custom Proposal Generator and Lux Integration.

This script demonstrates:
  1. Writing a custom ProposalGenerator
  2. Mixing it with DefaultProposalGenerator via WeightedMixGenerator
  3. Integrating Lux for capability-gated task execution
  4. Running the full pipeline

Run:
    python examples/tutorial_custom_proposal_lux.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    DefaultProposalGenerator,
    Executor,
    Goal,
    Graph,
    HistoryObserver,
    Lux,
    WeightedMixGenerator,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
    mock_task_runner,
)
from emergo.types import Authority, CoordinationEvent

# ---------------------------------------------------------------------------
# 1. Custom ProposalGenerator: always proposes from the highest-authority agent
# ---------------------------------------------------------------------------


class TopAuthorityProposalGenerator:
    """Always selects the highest-authority agent as the CE proposer.

    This is a pure-exploit strategy.  High-authority agents get to drive all
    graph mutations, concentrating proposals on the currently best predictor.

    Pair with DefaultProposalGenerator via WeightedMixGenerator to balance
    exploration (random) and exploitation (top-authority).
    """

    def __init__(self) -> None:
        self._n_proposed: int = 0
        self._proposer_counts: Counter = Counter()

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> CoordinationEvent | None:
        if G_t.n_agents < 2:
            return None

        agents = list(G_t.agent_ids)

        # Deterministic: pick the agent with the highest authority score.
        # Break ties with the lowest index (consistent ordering).
        top_agent = max(agents, key=lambda a: (A_t.get(a), -agents.index(a)))
        from_idx = G_t.agent_index(top_agent)

        # Pick a random target (uniform over all other agents)
        candidates = [i for i in range(G_t.n_agents) if i != from_idx]
        to_idx = int(rng.choice(candidates))
        to_agent = agents[to_idx]

        self._n_proposed += 1
        self._proposer_counts[top_agent] += 1

        i, j = from_idx, to_idx
        if G_t.adjacency[i, j] == 0.0:
            return CoordinationEvent(
                event_type="add_edge",
                participants=(top_agent, to_agent),
                params=frozenset([("weight", float(A_t.get(top_agent)))]),
            )
        else:
            return CoordinationEvent(
                event_type="remove_edge",
                participants=(top_agent, to_agent),
                params=frozenset(),
            )

    @property
    def proposal_distribution(self) -> dict:
        """Fraction of proposals made by each agent."""
        total = max(1, self._n_proposed)
        return {k: v / total for k, v in self._proposer_counts.items()}


def print_banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def build_test_graph(n: int = 6, seed: int = 1) -> Graph:
    rng = np.random.default_rng(seed)
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.35:
                adj[i, j] = rng.uniform(0.25, 0.75)
    caps = rng.uniform(0.3, 0.7, (n, 4))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def main() -> None:
    n_iter = 500

    # -----------------------------------------------------------------------
    # Step 1: The ProposalGenerator protocol
    # -----------------------------------------------------------------------
    print_banner("Step 1: ProposalGenerator Protocol")
    print(
        "  Any object with a .propose(A_t, G_t, rng) method that returns\n"
        "  Optional[CoordinationEvent] qualifies as a ProposalGenerator.\n"
        "  The kernel calls it once per iteration."
    )

    # -----------------------------------------------------------------------
    # Step 2: TopAuthorityProposalGenerator
    # -----------------------------------------------------------------------
    print_banner("Step 2: TopAuthorityProposalGenerator")
    G0 = build_test_graph(n=6, seed=1)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=1)
    A0 = make_initial_authority(G0.agent_ids, baseline=0.5)

    top_gen = TopAuthorityProposalGenerator()
    obs_top = HistoryObserver()

    state_top, reason_top = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=n_iter,
        proposal_generator=top_gen,
        observers=[obs_top],
        rng=np.random.default_rng(10),
    )

    _, _, A_top, _ = state_top
    top_agent = max(A_top.scores, key=A_top.scores.get)
    top_dist = top_gen.proposal_distribution

    print(f"  Termination    : {reason_top}")
    print(f"  Accept rate    : {obs_top.ce_acceptance_rate:.1%}")
    print(f"  Top agent      : {top_agent} (authority={A_top.get(top_agent):.3f})")
    print("\n  Proposal distribution (TopAuthorityProposalGenerator):")
    for agent_id, frac in sorted(top_dist.items(), key=lambda kv: -kv[1]):
        bar = "█" * int(frac * 30)
        print(f"    {agent_id:12s}  {frac:.1%}  {bar}")

    # -----------------------------------------------------------------------
    # Step 3: WeightedMixGenerator — 30% TopAuthority + 70% Default
    # -----------------------------------------------------------------------
    print_banner("Step 3: WeightedMixGenerator (70% Default + 30% TopAuthority)")

    default_gen = DefaultProposalGenerator()
    mix_top = TopAuthorityProposalGenerator()
    mixed_gen = WeightedMixGenerator(
        [
            (default_gen, 0.70),
            (mix_top, 0.30),
        ]
    )

    obs_mix = HistoryObserver()
    state_mix, reason_mix, diag_mix = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=n_iter,
        proposal_generator=mixed_gen,
        observers=[obs_mix],
        collect_diagnostics=True,
        rng=np.random.default_rng(10),
    )

    _, _, A_mix, _ = state_mix
    print(f"  Termination    : {reason_mix}")
    print(f"  Accept rate    : {obs_mix.ce_acceptance_rate:.1%}")

    # CE proposer distribution from diagnostics
    proposer_counts = Counter(
        r.ce_proposer for r in diag_mix.records if r.ce_accepted and r.ce_proposer is not None
    )
    total_accepted = sum(proposer_counts.values()) or 1
    print("\n  Accepted-CE distribution (mixed generator):")
    for agent_id, cnt in proposer_counts.most_common():
        frac = cnt / total_accepted
        auth = A_mix.get(agent_id)
        bar = "█" * int(frac * 30)
        print(f"    {agent_id:12s}  {frac:.1%}  (auth={auth:.3f})  {bar}")

    top_proposer = proposer_counts.most_common(1)[0][0] if proposer_counts else "?"
    top_authority = max(A_mix.scores, key=A_mix.scores.get)
    if top_proposer == top_authority:
        print(f"\n  ✓ Top proposer ({top_proposer}) == top-authority agent — exploit is working.")
    else:
        print(f"\n  Top proposer: {top_proposer}, top authority: {top_authority}")

    # -----------------------------------------------------------------------
    # Step 4: Lux integration — granting capabilities
    # -----------------------------------------------------------------------
    print_banner("Step 4: Lux Integration — Granting Capabilities")

    lux = Lux()
    # Grant execute_task to all agents in the swarm
    for agent_id in G0.agent_ids:
        lux.grant_capability(agent_id, "execute_task")

    print("  Capabilities granted:")
    for agent_id in G0.agent_ids:
        has_cap = lux.check_capability(agent_id, "execute_task")
        print(f"    {agent_id:12s}  execute_task={has_cap}")

    # -----------------------------------------------------------------------
    # Step 5: Run kernel again with Lux + mixed generator, then execute a goal
    # -----------------------------------------------------------------------
    print_banner("Step 5: Full Pipeline — Custom Proposals + Lux + Executor")

    final_gen = WeightedMixGenerator(
        [
            (DefaultProposalGenerator(), 0.70),
            (TopAuthorityProposalGenerator(), 0.30),
        ]
    )

    final_state, final_reason = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=n_iter,
        proposal_generator=final_gen,
        lux=lux,
        rng=np.random.default_rng(42),
    )

    _, _, A_final, _ = final_state
    best_agent = max(A_final.scores, key=A_final.scores.get)
    print(f"  Kernel done    : {final_reason}")
    print(f"  Best agent     : {best_agent} (authority={A_final.get(best_agent):.3f})")

    # Execute a goal through the best agent
    executor = Executor(lux=lux, task_runner=mock_task_runner)
    goal = Goal(
        goal_id="tutorial_goal",
        description="Coordinate swarm topology adjustment",
        required_capability="execute_task",
        initiating_agent=best_agent,
        resource_budget=5.0,
        max_depth=3,
    )

    result = executor.execute(goal, final_state)
    print("\n  Execution result:")
    print(f"    goal_id        : {result.goal_id}")
    print(f"    success        : {result.success}")
    print(f"    tasks_attempted: {result.tasks_attempted}")
    print(f"    tasks_succeeded: {result.tasks_succeeded}")
    print(f"    resources_spent: {result.resources_spent:.1f}")
    print(
        f"    audit IDs      : {result.audit_ids[:2]}{'...' if len(result.audit_ids) > 2 else ''}"
    )

    print_banner("Done")
    print("  Demonstrated: TopAuthorityProposalGenerator + WeightedMixGenerator")
    print("  + Lux capability grants + Executor task execution.")
    print()


if __name__ == "__main__":
    main()
