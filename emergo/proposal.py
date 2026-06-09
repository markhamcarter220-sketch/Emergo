"""ProposalGenerator — extensible CE sampling protocol.

The default kernel loop samples CEs via authority-weighted softmax over agents,
alternating add_edge/remove_edge based on current adjacency state.  This works
well as a bootstrap, but rich applications need richer proposals:

  - Inject coordination events derived from a DependencyPlanner
  - Pipe LLM-generated structural proposals through an adapter
  - Mix weighted strategies (explore vs. exploit)

The `ProposalGenerator` protocol is the single extension point.  Pass any
conforming object as `proposal_generator=` to `emergo_kernel`.

Concrete implementations:

  DefaultProposalGenerator   — current softmax edge-flip strategy (default)
  SequenceProposalGenerator  — replays a pre-built list; useful for replay /
                               testing / injecting LLM-generated CEs
  WeightedMixGenerator       — mixes N generators by sampling weight
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from emergo.types import Authority, CoordinationEvent, Graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProposalGenerator(Protocol):
    """Read-only CE sampling interface.

    Implementations must be deterministic given the same (A_t, G_t, rng)
    inputs so that the kernel's sequential determinism guarantee (INV-4) is
    preserved.  Side effects (logging, metrics) are acceptable; state
    mutations on A_t or G_t are not.

    Returns None for degenerate inputs (e.g., fewer than 2 agents).
    """

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> CoordinationEvent | None:
        """Sample a CE proposal.  May return None to skip this iteration."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class DefaultProposalGenerator:
    """Softmax authority-weighted edge-flip generator.

    High-authority agents are more likely to be selected as initiators.
    CE type alternates between add_edge and remove_edge based on whether the
    proposed directed edge already exists.

    This is the strategy the kernel used before ProposalGenerator was
    introduced — the default preserves all prior behaviour.
    """

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> CoordinationEvent | None:
        if G_t.n_agents < 2:
            return None

        agents = list(G_t.agent_ids)
        auth_vec = np.array([A_t.get(a) for a in agents])

        shifted = auth_vec - auth_vec.max()
        weights = np.exp(shifted)
        weights /= weights.sum()

        from_idx = int(rng.choice(len(agents), p=weights))
        candidates = [i for i in range(len(agents)) if i != from_idx]
        to_idx = int(rng.choice(candidates))

        from_agent = agents[from_idx]
        to_agent = agents[to_idx]
        i = G_t.agent_index(from_agent)
        j = G_t.agent_index(to_agent)

        if G_t.adjacency[i, j] == 0.0:
            return CoordinationEvent(
                event_type="add_edge",
                participants=(from_agent, to_agent),
                params=frozenset([("weight", float(auth_vec[from_idx]))]),
            )
        else:
            return CoordinationEvent(
                event_type="remove_edge",
                participants=(from_agent, to_agent),
                params=frozenset(),
            )


class SequenceProposalGenerator:
    """Replays a fixed sequence of CEs, then loops or returns None.

    Useful for:
      - Replay / deterministic testing
      - Injecting LLM-generated or DependencyPlanner-derived CE sequences
      - Mixing with WeightedMixGenerator for structured exploration phases

    Args:
        ces:       Ordered sequence of CEs to emit.
        loop:      If True, restart from the beginning after exhausting the
                   sequence.  If False, return None after exhaustion (kernel
                   skips degenerate iterations and keeps going).
    """

    def __init__(self, ces: list[CoordinationEvent], *, loop: bool = False) -> None:
        self._ces = list(ces)
        self._loop = loop
        self._idx = 0

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> CoordinationEvent | None:
        if not self._ces:
            return None
        ce = self._ces[self._idx % len(self._ces)]
        self._idx += 1
        if not self._loop and self._idx > len(self._ces):
            return None
        # Skip CEs whose participants are no longer in the graph
        for p in ce.participants:
            if p not in G_t.agent_ids:
                logger.debug(
                    "SequenceProposalGenerator: skipping CE %s — participant %r not in graph",
                    ce.event_type,
                    p,
                )
                return None
        return ce

    def reset(self) -> None:
        """Restart the sequence from the beginning."""
        self._idx = 0

    @property
    def exhausted(self) -> bool:
        """True when the sequence is done and loop=False."""
        return not self._loop and self._idx >= len(self._ces)


class WeightedMixGenerator:
    """Probabilistic mix of multiple ProposalGenerators.

    At each step, randomly picks one generator according to `weights` and
    delegates to it.  Falls back to the next generator in weight order if
    the chosen one returns None.

    Args:
        generators: Sequence of (generator, weight) pairs.  Weights need not
                    sum to 1 — they are normalised internally.
    """

    def __init__(self, generators: list[tuple[ProposalGenerator, float]]) -> None:
        if not generators:
            raise ValueError("WeightedMixGenerator requires at least one generator")
        self._gens = [g for g, _ in generators]
        raw = np.array([w for _, w in generators], dtype=float)
        if raw.min() < 0:
            raise ValueError("Weights must be non-negative")
        self._weights = raw / raw.sum()

    def propose(
        self,
        A_t: Authority,
        G_t: Graph,
        rng: np.random.Generator,
    ) -> CoordinationEvent | None:
        order = list(
            rng.choice(len(self._gens), size=len(self._gens), p=self._weights, replace=False)
        )
        for idx in order:
            ce: CoordinationEvent | None = self._gens[idx].propose(A_t, G_t, rng)
            if ce is not None:
                return ce
        return None
