#!/usr/bin/env python3
"""Custom ProposalGenerator demo.

Shows three ways to customise CE sampling in the Emergo kernel:

  1. DefaultProposalGenerator  — built-in softmax authority edge-flip (baseline)
  2. SequenceProposalGenerator — replay a hand-crafted sequence (e.g. LLM output)
  3. WeightedMixGenerator      — mix 80% default + 20% structured add-edge proposals

The demo runs each generator for 100 iterations and compares CE acceptance rates.

Run:
    python examples/custom_proposal_generator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    DefaultProposalGenerator,
    Graph,
    HistoryObserver,
    SequenceProposalGenerator,
    WeightedMixGenerator,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.types import CoordinationEvent


def _star_graph(n: int = 5) -> Graph:
    """Hub-and-spoke: agent0 connects to all others."""
    ids = tuple(f"a{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        adj[0, i] = 0.4
    caps = np.ones((n, 4), dtype=float) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _state(G: Graph, seed: int = 0):
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    return G, phi, A, []


def _make_ce(event_type: str, participants: tuple, **params) -> CoordinationEvent:
    return CoordinationEvent(
        event_type=event_type,
        participants=participants,
        params=frozenset(params.items()),
    )


def _run(label: str, gen, G: Graph, seed: int, iterations: int = 100) -> None:
    state = _state(G, seed=seed)
    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        state,
        max_iterations=iterations,
        proposal_generator=gen,
        observers=[obs],
        rng=np.random.default_rng(seed),
    )
    _, _, A_final, _ = final_state
    mean_auth = np.mean([A_final.get(a) for a in A_final.scores])
    print(f"[{label:30s}]  "
          f"reason={reason:<25}  "
          f"accept={obs.ce_acceptance_rate:.1%}  "
          f"mean_authority={mean_auth:.3f}")


def main() -> None:
    G = _star_graph(5)
    ids = G.agent_ids

    print("=== Custom ProposalGenerator Comparison ===\n")
    print(f"Graph: {G.n_agents} agents, star topology\n")

    # 1. Default
    _run("DefaultProposalGenerator", DefaultProposalGenerator(), G, seed=1)

    # 2. Sequence: alternate add/remove on the hub→spoke edge
    structured_ces = [
        _make_ce("add_edge",    (ids[0], ids[3]), weight=0.6),
        _make_ce("remove_edge", (ids[0], ids[3])),
        _make_ce("add_edge",    (ids[1], ids[2]), weight=0.7),
        _make_ce("remove_edge", (ids[1], ids[2])),
        _make_ce("add_edge",    (ids[2], ids[4]), weight=0.5),
    ]
    seq_gen = SequenceProposalGenerator(structured_ces, loop=True)
    _run("SequenceProposalGenerator(loop)", seq_gen, G, seed=2)

    # 3. WeightedMix: 80% default + 20% structured
    default_gen = DefaultProposalGenerator()
    structured_gen = SequenceProposalGenerator(structured_ces, loop=True)
    mix_gen = WeightedMixGenerator([(default_gen, 0.8), (structured_gen, 0.2)])
    _run("WeightedMixGenerator(0.8/0.2)", mix_gen, G, seed=3)

    print("\nDone. Tweak the generators or graph topology to explore different dynamics.")


if __name__ == "__main__":
    main()
