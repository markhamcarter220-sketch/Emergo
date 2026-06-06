"""Fixed-dimensional feature extraction from variable-size graphs and CEs.

Spectral features (eigenvalues of the global adjacency matrix) are used for φ
because they are inherently non-factorizable: every agent's connections influence
every eigenvalue, satisfying the entanglement invariant.
"""
from __future__ import annotations

import numpy as np

from emergo.types import CoordinationEvent, Graph

_EVENT_TYPE_INDEX: dict[str, int] = {
    "add_edge": 0,
    "remove_edge": 1,
    "update_capabilities": 2,
    "add_agent": 3,
    "remove_agent": 4,
}


def extract_graph_features(G: Graph, d_features: int) -> np.ndarray:
    """Map variable-size graph → fixed R^{d_features} feature vector.

    Layout (before final pad/truncate to d_features):
      [spectral (k_s values)] [cap_mean (k_c values)] [cap_std (k_c values)] [graph_stats (4)]

    Spectral features use eigvalsh on the symmetric part of adjacency to capture
    global topology: no single agent's contribution is isolable (entanglement).
    """
    n = G.n_agents
    k_s = max(1, d_features // 3)
    k_c = max(1, (d_features - k_s - 4) // 2)

    # --- Spectral features ---
    if n > 0:
        sym_adj = (G.adjacency + G.adjacency.T) / 2.0
        eigs = np.sort(np.linalg.eigvalsh(sym_adj))[::-1]
        spectral = np.zeros(k_s)
        spectral[: min(k_s, len(eigs))] = eigs[:k_s]
    else:
        spectral = np.zeros(k_s)

    # --- Capability statistics ---
    d_cap = G.capabilities.shape[1] if G.capabilities.ndim > 1 else 1
    if n > 0:
        raw_mean = G.capabilities.mean(axis=0).ravel()
        raw_std = G.capabilities.std(axis=0).ravel() if n > 1 else np.zeros(d_cap)
    else:
        raw_mean = np.zeros(d_cap)
        raw_std = np.zeros(d_cap)

    cap_mean = np.zeros(k_c)
    cap_mean[: min(k_c, len(raw_mean))] = raw_mean[: k_c]

    cap_std = np.zeros(k_c)
    cap_std[: min(k_c, len(raw_std))] = raw_std[: k_c]

    # --- Graph-level scalars ---
    graph_stats = np.array(
        [
            float(n),
            float(np.count_nonzero(G.adjacency)),
            float(G.adjacency.sum()),
            float(G.adjacency.mean()) if n > 0 else 0.0,
        ]
    )

    raw = np.concatenate([spectral, cap_mean, cap_std, graph_stats])

    if len(raw) >= d_features:
        return raw[:d_features].astype(float)
    return np.pad(raw, (0, d_features - len(raw))).astype(float)


def encode_ce(CE: CoordinationEvent, d_ce: int) -> np.ndarray:
    """Encode a CoordinationEvent as a fixed R^{d_ce} vector."""
    enc = np.zeros(d_ce, dtype=float)
    if d_ce < 1:
        return enc

    n_types = max(len(_EVENT_TYPE_INDEX), 1)
    enc[0] = _EVENT_TYPE_INDEX.get(CE.event_type, -1) / n_types

    if d_ce > 1:
        enc[1] = min(len(CE.participants), 10) / 10.0

    if d_ce > 2:
        weight = CE.get_param("weight")
        enc[2] = float(weight) if weight is not None else 0.0

    if d_ce > 3:
        enc[3] = float(len(CE.params)) / 10.0

    return enc
