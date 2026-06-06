"""Core immutable data structures for the Emergo Kernel state machine.

State = (G_t, φ_t, A_t, E_t) — the only mutable state; all operations are pure functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Immutable graph G_t. Agents are nodes; directed weighted edges encode relationships.

    adjacency[i, j] = weight of edge from agent_ids[i] → agent_ids[j] (0 = no edge).
    capabilities[i, :] = capability feature vector for agent_ids[i].
    """

    agent_ids: tuple          # ordered tuple of agent ID strings
    adjacency: np.ndarray     # (n, n) float — read-only after construction
    capabilities: np.ndarray  # (n, d_cap) float — read-only after construction

    def __post_init__(self) -> None:
        adj = np.array(self.adjacency, dtype=float)
        adj.flags.writeable = False
        object.__setattr__(self, "adjacency", adj)

        cap = np.array(self.capabilities, dtype=float)
        cap.flags.writeable = False
        object.__setattr__(self, "capabilities", cap)

        n = len(self.agent_ids)
        assert adj.shape == (n, n), f"adjacency must be ({n},{n}), got {adj.shape}"
        assert cap.shape[0] == n, f"capabilities must have {n} rows, got {cap.shape[0]}"

    def __hash__(self) -> int:
        return hash((self.agent_ids, self.adjacency.tobytes(), self.capabilities.tobytes()))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Graph)
            and self.agent_ids == other.agent_ids
            and np.array_equal(self.adjacency, other.adjacency)
            and np.array_equal(self.capabilities, other.capabilities)
        )

    @property
    def n_agents(self) -> int:
        return len(self.agent_ids)

    def agent_index(self, agent_id: str) -> int:
        return self.agent_ids.index(agent_id)


@dataclass(frozen=True)
class CoordinationEvent:
    """A proposed graph transformation. Atomic and deterministic when authorized.

    event_type: one of "add_edge" | "remove_edge" | "update_capabilities" |
                        "add_agent" | "remove_agent"
    participants: ordered tuple of agent IDs involved in the event
    params: frozenset of (key, value) pairs carrying event-specific parameters
    """

    event_type: str
    participants: tuple
    params: frozenset

    def get_param(self, key: str, default=None):
        return dict(self.params).get(key, default)


@dataclass
class PhiMap:
    """Learned latent map φ and stationary transition operator F.

    φ(G) = W_phi @ extract_features(G) + b_phi    shape: (d_latent,)
    F(z, c) = W_F @ concat(z, c) + b_F            shape: (d_latent,)

    Entanglement invariant: W_phi must have rank ≥ d_latent // 2 at all times.
    F is the "Axiom 5.4.3" approximately-linear transition in φ-space.
    """

    W_phi: np.ndarray   # (d_latent, d_features)
    b_phi: np.ndarray   # (d_latent,)
    W_F: np.ndarray     # (d_latent, d_latent + d_ce)
    b_F: np.ndarray     # (d_latent,)
    d_latent: int
    d_features: int
    d_ce: int

    def embed(self, G: Graph) -> np.ndarray:
        """φ(G): project graph into latent space."""
        from emergo.features import extract_graph_features
        f = extract_graph_features(G, self.d_features)
        return self.W_phi @ f + self.b_phi

    def transition(self, z: np.ndarray, ce_encoding: np.ndarray) -> np.ndarray:
        """F(z, c): predict next latent state from current state and CE."""
        return self.W_F @ np.concatenate([z, ce_encoding]) + self.b_F

    def copy(self) -> PhiMap:
        return PhiMap(
            W_phi=self.W_phi.copy(),
            b_phi=self.b_phi.copy(),
            W_F=self.W_F.copy(),
            b_F=self.b_F.copy(),
            d_latent=self.d_latent,
            d_features=self.d_features,
            d_ce=self.d_ce,
        )


@dataclass
class Authority:
    """Per-agent authority scores in [0, 1].

    High authority → broader topological reachability (influences CE sampling).
    Updates are continuous (no discrete jumps); sum is not conserved by design.
    """

    scores: Dict[str, float]
    baseline: float = 0.5       # historical mean used for Δ calibration

    def get(self, agent_id: str) -> float:
        return float(np.clip(self.scores.get(agent_id, self.baseline), 0.0, 1.0))

    def set(self, agent_id: str, value: float) -> None:
        self.scores[agent_id] = float(np.clip(value, 0.0, 1.0))

    def copy(self) -> Authority:
        return Authority(scores=dict(self.scores), baseline=self.baseline)


@dataclass
class Errors:
    """Per-agent L2 prediction errors from one iteration."""

    per_agent: Dict[str, float]   # agent_id → error magnitude ≥ 0

    def max_error(self) -> float:
        return max(self.per_agent.values()) if self.per_agent else 0.0

    def mean_error(self) -> float:
        if not self.per_agent:
            return 0.0
        return sum(self.per_agent.values()) / len(self.per_agent)


# The complete mutable state of the system.  All four components are written
# atomically by each operation; no shared mutable sub-state exists.
State = Tuple[Graph, PhiMap, Authority, List[Errors]]
