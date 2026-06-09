"""Fixed-dimensional feature extraction from variable-size graphs and CEs.

Spectral features (eigenvalues of the symmetrized adjacency matrix) are used for φ
because they are inherently non-factorizable: every agent's connections influence
every eigenvalue, satisfying the entanglement invariant.

Layout (d_features vector):
  [k_eig normalized eigenvalues]   k_eig = d_features // 2
  [k_met structural metrics]       k_met = d_features - k_eig

All features are in [-1, 1].  Never NaN.  Always shape (d_features,).
Handles any n_agents ≥ 0, including dynamic topology changes mid-run.
"""

from __future__ import annotations

import numpy as np

from emergo.types import CoordinationEvent, Graph

_EVENT_TYPE_INDEX: dict[str, int] = {
    # Graph-mutation CEs (handled by ce_execute)
    "add_edge": 0,
    "remove_edge": 1,
    "update_capabilities": 2,
    "add_agent": 3,
    "remove_agent": 4,
    # Execution-layer CEs (handled by Executor; included for φ-space encoding)
    "execute_task": 5,
    "decompose_goal": 6,
    "delegate": 7,
}

# Number of structural metrics returned by _structural_metrics().
# This constant must match the length of the array returned by that function.
_N_STRUCTURAL = 8


def extract_graph_features(G: Graph, d_features: int) -> np.ndarray:
    """Map variable-size graph → fixed R^{d_features} feature vector.

    Layout:
      Positions 0 … k_eig-1 : top k_eig eigenvalues of the symmetrized
          adjacency, normalized by the largest absolute eigenvalue (→ [-1, 1]).
          Positions beyond the graph rank are padded with 0.
      Positions k_eig … d_features-1 : k_met = d_features - k_eig structural
          metrics, each normalized to [-1, 1].

    Guarantees:
      • shape  == (d_features,)  for any n_agents ≥ 0
      • no NaN / Inf
      • all values in [-1, 1]
    """
    k_eig = max(1, d_features // 2)
    k_met = d_features - k_eig

    n = G.n_agents
    spectral = np.zeros(k_eig)
    all_eigs: np.ndarray = np.zeros(0)

    if n > 0:
        sym_adj = (G.adjacency + G.adjacency.T) / 2.0
        all_eigs = np.linalg.eigvalsh(sym_adj)  # ascending
        all_eigs = all_eigs[::-1]  # descending (largest first)
        max_abs = float(np.abs(all_eigs).max())
        if max_abs > 1e-12:
            norm_eigs = all_eigs / max_abs
        else:
            norm_eigs = all_eigs
        copy_k = min(k_eig, len(norm_eigs))
        spectral[:copy_k] = norm_eigs[:copy_k]

    structural = np.zeros(k_met)
    if k_met > 0:
        raw_metrics = _structural_metrics(G, n, all_eigs)
        copy_k = min(k_met, len(raw_metrics))
        structural[:copy_k] = raw_metrics[:copy_k]

    out = np.concatenate([spectral, structural])
    # Final safety: clamp NaN/Inf and enforce [-1, 1]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.clip(out, -1.0, 1.0)
    result: np.ndarray = out.astype(float)
    return result


def _structural_metrics(G: Graph, n: int, eigs: np.ndarray) -> np.ndarray:
    """Return _N_STRUCTURAL=8 structural metrics, each normalized to [-1, 1].

    Metrics (index, name, normalization):
      0  spectral_entropy   entropy of |eigenvalue| distribution / log(n)  → [0, 1]
      1  edge_density       nonzero entries / n²                            → [0, 1]
      2  mean_edge_weight   adj.mean()                                      → [0, 1]
      3  weight_std         adj.std() / 0.5                                 → [0, 1]
      4  out_degree_mean    row-sums mean / n                               → [0, 1]
      5  out_degree_std     row-sums std  / n                               → [0, 1]
      6  cap_mean           capabilities.mean()                             → [0, 1]
      7  cap_std            capabilities.std() / 0.5                       → [0, 1]
    """
    if n == 0:
        return np.zeros(_N_STRUCTURAL)

    adj = G.adjacency
    n_sq = float(n * n)

    # 0: spectral entropy
    if len(eigs) > 1:
        abs_eigs = np.abs(eigs)
        s = abs_eigs.sum()
        if s > 1e-12:
            p = abs_eigs / s
            p = p[p > 1e-12]
            raw_entropy = float(-np.sum(p * np.log(p)))
            max_entropy = float(np.log(len(eigs)))
            spectral_entropy = raw_entropy / max_entropy if max_entropy > 1e-12 else 0.0
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0

    # 1: edge_density
    edge_density = float(np.count_nonzero(adj)) / n_sq

    # 2: mean_edge_weight (adj values assumed ∈ [0, 1])
    mean_weight = float(adj.mean())

    # 3: weight_std normalized by 0.5 (max std for values in [0, 1] is 0.5)
    weight_std = float(adj.std()) / 0.5

    # 4-5: out-degree statistics
    out_deg = adj.sum(axis=1)
    out_deg_mean = float(out_deg.mean()) / float(n)
    out_deg_std = float(out_deg.std()) / float(n)

    # 6-7: capability statistics
    caps = G.capabilities
    if caps.size > 0:
        cap_mean = float(caps.mean())
        cap_std = float(caps.std()) / 0.5
    else:
        cap_mean = 0.0
        cap_std = 0.0

    return np.array(
        [
            spectral_entropy,
            edge_density,
            mean_weight,
            weight_std,
            out_deg_mean,
            out_deg_std,
            cap_mean,
            cap_std,
        ],
        dtype=float,
    )


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
        enc[2] = float(weight) if weight is not None else 0.0  # type: ignore[arg-type]

    if d_ce > 3:
        enc[3] = float(len(CE.params)) / 10.0

    return enc
