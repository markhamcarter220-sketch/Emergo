"""Core immutable data structures for the Emergo Kernel state machine.

State = (G_t, φ_t, A_t, E_t) — the only mutable state; all operations are pure functions.
Task and Goal are execution-layer types defined here to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Immutable graph G_t. Agents are nodes; directed weighted edges encode relationships.

    adjacency[i, j] = weight of edge from agent_ids[i] → agent_ids[j] (0 = no edge).
    capabilities[i, :] = capability feature vector for agent_ids[i].
    """

    agent_ids: tuple  # ordered tuple of agent ID strings
    adjacency: np.ndarray  # (n, n) float — read-only after construction
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

    def get_param(self, key: str, default: object = None) -> object:
        return dict(self.params).get(key, default)


@dataclass
class PhiMap:
    """Learned latent map φ and stationary transition operator F.

    φ(G) = W_phi @ extract_features(G) + b_phi    shape: (d_latent,)
    F(z, c) = W_F @ concat(z, c) + b_F            shape: (d_latent,)

    Entanglement invariant: W_phi must have rank ≥ d_latent // 2 at all times.
    F is the "Axiom 5.4.3" approximately-linear transition in φ-space.
    """

    W_phi: np.ndarray  # (d_latent, d_features)
    b_phi: np.ndarray  # (d_latent,)
    W_F: np.ndarray  # (d_latent, d_latent + d_ce)
    b_F: np.ndarray  # (d_latent,)
    d_latent: int
    d_features: int
    d_ce: int

    def embed(self, G: Graph) -> np.ndarray:
        """φ(G): project graph into latent space."""
        from emergo.features import extract_graph_features

        f = extract_graph_features(G, self.d_features)
        result: np.ndarray = self.W_phi @ f + self.b_phi
        return result

    def transition(self, z: np.ndarray, ce_encoding: np.ndarray) -> np.ndarray:
        """F(z, c): predict next latent state from current state and CE."""
        result: np.ndarray = self.W_F @ np.concatenate([z, ce_encoding]) + self.b_F
        return result

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

    scores: dict[str, float]
    baseline: float = 0.5  # historical mean used for Δ calibration

    def get(self, agent_id: str) -> float:
        return float(np.clip(self.scores.get(agent_id, self.baseline), 0.0, 1.0))

    def set(self, agent_id: str, value: float) -> None:
        self.scores[agent_id] = float(np.clip(value, 0.0, 1.0))

    def copy(self) -> Authority:
        return Authority(scores=dict(self.scores), baseline=self.baseline)


@dataclass
class Errors:
    """Per-agent L2 prediction errors from one iteration."""

    per_agent: dict[str, float]  # agent_id → error magnitude ≥ 0
    proposer_id: str | None = None  # agent that received the global phi-prediction error

    def max_error(self) -> float:
        return max(self.per_agent.values()) if self.per_agent else 0.0

    def mean_error(self) -> float:
        if not self.per_agent:
            return 0.0
        return sum(self.per_agent.values()) / len(self.per_agent)


@dataclass(frozen=True)
class ErrorScales:
    """Running EMA scale estimates for the two error channels.

    Kept separate so global phi-prediction error and local structural
    delta error are each normalized against their own history,
    preventing the channel with larger magnitude from always dominating.

    Attributes:
        global_scale: EMA of proposer (global phi-prediction) errors.
        local_scale:  EMA of participant (local structural delta) errors.
        alpha:        EMA decay — higher = faster adaptation to recent errors.
    """

    global_scale: float = 1.0
    local_scale: float = 1.0
    alpha: float = 0.1

    def update(self, global_errors: list[float], local_errors: list[float]) -> ErrorScales:
        """Return new ErrorScales updated from this iteration's observations.

        Only updates a channel if it received at least one observation.
        Scale floor of 1e-6 prevents division by zero in authority_update.
        """
        new_global = self.global_scale
        new_local = self.local_scale
        if global_errors:
            obs_global = float(np.mean(global_errors))
            new_global = self.alpha * obs_global + (1.0 - self.alpha) * self.global_scale
        if local_errors:
            obs_local = float(np.mean(local_errors))
            new_local = self.alpha * obs_local + (1.0 - self.alpha) * self.local_scale
        return ErrorScales(
            global_scale=max(new_global, 1e-6),
            local_scale=max(new_local, 1e-6),
            alpha=self.alpha,
        )


@dataclass(frozen=True)
class EligibilityTraces:
    """Per-agent eligibility traces for multi-step credit assignment.

    Traces decay by ``decay`` each iteration and reset to 1.0 when an agent
    participates in an accepted CE.  Multiplying the authority delta by the
    trace weight amplifies recent contributions and dampens stale ones.

    Passing ``traces=None`` to ``authority_update`` is equivalent to uniform
    traces (weight 1.0 for all agents) — identical to pre-trace behavior.
    """

    traces: dict[str, float]
    decay: float = 0.8

    def __hash__(self) -> int:
        return hash((frozenset(self.traces.items()), self.decay))

    def step(
        self, accepted_participants: tuple[str, ...] | None = None
    ) -> "EligibilityTraces":
        """Decay all traces; reset participating agents to 1.0."""
        decayed = {a: v * self.decay for a, v in self.traces.items()}
        if accepted_participants:
            for a in accepted_participants:
                if a in decayed:
                    decayed[a] = 1.0
        return EligibilityTraces(traces=decayed, decay=self.decay)

    def get(self, agent_id: str) -> float:
        """Return trace weight for agent; defaults to 1.0 for unknown agents."""
        return self.traces.get(agent_id, 1.0)

    @classmethod
    def uniform(
        cls, agent_ids: tuple[str, ...], decay: float = 0.8
    ) -> "EligibilityTraces":
        """Construct traces with all agents at weight 1.0."""
        return cls(traces={a: 1.0 for a in agent_ids}, decay=decay)


# The complete mutable state of the system.  All four components are written
# atomically by each operation; no shared mutable sub-state exists.
State = tuple[Graph, PhiMap, Authority, list[Errors]]


# ---------------------------------------------------------------------------
# Execution-layer types (used by Executor and Planner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """A unit of work to be executed by an agent through the Lux-authorized path.

    INV-8: depth tracks recursion level; rejected when depth > Goal.max_depth.
    INV-5: required_capability is verified by Lux before execution begins.
    INV-6: resource_cost is pre-deducted by Lux; refunded on failure.
    """

    task_id: str
    description: str
    required_capability: str  # Lux capability name required to execute
    initiating_agent: str  # agent ID proposing this task
    resource_cost: float = 1.0
    resource_type: str = "compute"
    depth: int = 0  # recursion depth (for INV-8)
    parent_task_id: str | None = None


@dataclass(frozen=True)
class Goal:
    """High-level goal decomposed by the Planner into Tasks.

    resource_budget caps total resource spend across all tasks.
    max_depth enforces INV-8 for the decomposition tree.
    """

    goal_id: str
    description: str
    required_capability: str
    initiating_agent: str
    resource_budget: float = 10.0
    max_depth: int = 5
