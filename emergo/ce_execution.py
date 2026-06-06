"""Operation 1 — CE_Execution.

ce_execute(G_t, CE, lux, A_t) → (G_{t+1}, success, participants)

Atomic and deterministic: either the full transformation is applied to produce
G_{t+1}, or G_t is returned unchanged with success=False.  No partial writes.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from emergo.lux import Lux
from emergo.types import Authority, CoordinationEvent, Graph


def ce_execute(
    G_t: Graph,
    CE: CoordinationEvent,
    lux: Lux,
    A_t: Authority,
) -> Tuple[Graph, bool, tuple]:
    """Apply CE to G_t, producing an immutable G_{t+1}.

    Returns:
        G_next      — new graph (equals G_t if CE failed)
        success     — True iff CE was authorized and transformation applied
        participants — CE.participants on success, () on failure
    """
    if not lux.authorize(CE, G_t, A_t):
        return G_t, False, ()

    try:
        G_next = _apply(G_t, CE, dict(CE.params))
    except (ValueError, IndexError, KeyError):
        return G_t, False, ()

    return G_next, True, CE.participants


# ---------------------------------------------------------------------------
# Internal transformations — each returns a fresh immutable Graph
# ---------------------------------------------------------------------------

def _apply(G: Graph, CE: CoordinationEvent, params: dict) -> Graph:
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
        agent_ids=G.agent_ids + (agent_id,),
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
    copy = arr.copy()
    copy.flags.writeable = True
    return copy
