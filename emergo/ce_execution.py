"""Operation 1 — CE_Execution.

ce_execute(G_t, CE, lux, A_t) → (G_{t+1}, success, participants)

Atomic and deterministic: either the full transformation is applied to produce
G_{t+1}, or G_t is returned unchanged with success=False.  No partial writes.

Boundary note: ce_execute handles GRAPH MUTATION CEs only:
  add_edge, remove_edge, update_capabilities, add_agent, remove_agent

Execution-layer CEs (execute_task, decompose_goal, delegate) are intentionally
rejected here — they MUST go through Executor, which enforces INV-5/6/7/8.
Routing them through ce_execute would bypass resource deduction and audit.
"""

from __future__ import annotations

import numpy as np

from emergo.lux import Lux
from emergo.types import Authority, CoordinationEvent, Graph

_HIGH_AUTH_REMOVE_THRESHOLD: float = 0.5  # protect agents above this level from removal
_MAX_AGENTS_DEFAULT: int = 100  # cap graph growth; override via CE param "max_agents"


def ce_execute(
    G_t: Graph,
    CE: CoordinationEvent,
    lux: Lux,
    A_t: Authority,
) -> tuple[Graph, bool, tuple]:
    """Apply CE to G_t, producing an immutable G_{t+1}.

    Returns:
        G_next      — new graph (equals G_t if CE failed)
        success     — True iff CE was authorized and transformation applied
        participants — CE.participants on success, () on failure
    """
    if not lux.authorize(CE, G_t, A_t):
        return G_t, False, ()

    # Governance guard: protect high-authority agents from removal.
    if CE.event_type == "remove_agent" and CE.participants:
        target = CE.participants[0]
        if A_t.get(target) > _HIGH_AUTH_REMOVE_THRESHOLD:
            return G_t, False, ()

    # Governance guard: cap graph size to prevent unbounded growth.
    if CE.event_type == "add_agent":
        max_agents = int(dict(CE.params).get("max_agents", _MAX_AGENTS_DEFAULT))
        if G_t.n_agents >= max_agents:
            return G_t, False, ()

    try:
        G_next = _apply(G_t, CE, dict(CE.params))
    except (ValueError, IndexError, KeyError):
        return G_t, False, ()

    return G_next, True, CE.participants


# ---------------------------------------------------------------------------
# Internal transformations — each returns a fresh immutable Graph
# ---------------------------------------------------------------------------

# CE types that must go through Executor (not ce_execute).
# Explicitly listed so violations produce a clear error, not a silent failure.
_EXECUTOR_ONLY_TYPES = frozenset({"execute_task", "decompose_goal", "delegate"})


def _apply(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    if CE.event_type in _EXECUTOR_ONLY_TYPES:
        raise ValueError(
            f"CE type {CE.event_type!r} must be handled by Executor.execute(), "
            "not ce_execute(). Routing it here bypasses resource deduction and audit."
        )
    dispatch = {
        "add_edge": _add_edge,
        "remove_edge": _remove_edge,
        "update_capabilities": _update_capabilities,
        "add_agent": _add_agent,
        "remove_agent": _remove_agent,
    }
    fn = dispatch.get(CE.event_type)
    if fn is None:
        raise ValueError(f"Unknown CE event_type: {CE.event_type!r}")
    return fn(G, CE, params)


def _add_edge(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    from_id, to_id = CE.participants[0], CE.participants[1]
    if from_id == to_id:  # INV-12: self-loops bypass authority accountability
        raise ValueError(f"Self-loop {from_id}→{from_id} not permitted (INV-12)")
    weight = float(params.get("weight", 1.0))
    i, j = G.agent_index(from_id), G.agent_index(to_id)
    adj = _writeable_copy(G.adjacency)
    adj[i, j] = weight
    return Graph(agent_ids=G.agent_ids, adjacency=adj, capabilities=G.capabilities)


def _remove_edge(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    from_id, to_id = CE.participants[0], CE.participants[1]
    i, j = G.agent_index(from_id), G.agent_index(to_id)
    adj = _writeable_copy(G.adjacency)
    adj[i, j] = 0.0
    return Graph(agent_ids=G.agent_ids, adjacency=adj, capabilities=G.capabilities)


def _update_capabilities(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    agent_id = CE.participants[0]
    new_caps = np.array(params["capabilities"], dtype=float)
    i = G.agent_index(agent_id)
    cap = _writeable_copy(G.capabilities)
    cap[i] = new_caps[: cap.shape[1]]
    return Graph(agent_ids=G.agent_ids, adjacency=G.adjacency, capabilities=cap)


def _add_agent(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    agent_id = params["agent_id"]
    n = G.n_agents
    d_cap = G.capabilities.shape[1] if G.capabilities.ndim > 1 else 1
    initial_caps = np.array(params.get("capabilities", [0.0] * d_cap), dtype=float)

    new_adj = np.zeros((n + 1, n + 1), dtype=float)
    new_adj[:n, :n] = G.adjacency

    new_caps = np.zeros((n + 1, d_cap), dtype=float)
    new_caps[:n] = G.capabilities
    new_caps[n] = initial_caps[:d_cap]

    return Graph(
        agent_ids=(*G.agent_ids, agent_id),
        adjacency=new_adj,
        capabilities=new_caps,
    )


def _remove_agent(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
    agent_id = CE.participants[0]
    i = G.agent_index(agent_id)
    keep = [j for j in range(G.n_agents) if j != i]
    return Graph(
        agent_ids=tuple(G.agent_ids[k] for k in keep),
        adjacency=G.adjacency[np.ix_(keep, keep)],
        capabilities=G.capabilities[keep],
    )


def _writeable_copy(arr: np.ndarray) -> np.ndarray:
    copy: np.ndarray = arr.copy()
    copy.flags.writeable = True
    return copy
