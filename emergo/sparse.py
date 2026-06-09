"""Sparse graph representations for large Emergo networks.

Provides optional scipy.sparse integration for memory-efficient adjacency
storage when agent count is large and graph density is low.

scipy.sparse is an optional dependency. All functions degrade gracefully to
dense numpy when scipy is not installed, using a warning.

Usage::

    from emergo.sparse import to_sparse_adjacency, sparse_graph_features, is_sparse_beneficial

    # Check if sparse would help
    if is_sparse_beneficial(G):
        features = sparse_graph_features(G, d_features=16)
    else:
        features = extract_graph_features(G, d_features=16)  # existing dense path
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from emergo.types import Graph

# ---------------------------------------------------------------------------
# Optional scipy import — all scipy usage must stay inside try/except blocks
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE: bool = False

try:
    import scipy.sparse as _sp  # type: ignore[import-untyped]
    import scipy.sparse.linalg as _spl  # type: ignore[import-untyped]

    _SCIPY_AVAILABLE = True
except ImportError:
    _sp = None
    _spl = None

# ---------------------------------------------------------------------------
# Module-level flag: warn about missing scipy only once per process
# ---------------------------------------------------------------------------

_WARNED_SCIPY: bool = False


def _warn_scipy_once() -> None:
    """Emit a one-time UserWarning that scipy is not installed."""
    global _WARNED_SCIPY
    if not _WARNED_SCIPY:
        warnings.warn(
            "scipy is not installed; falling back to dense numpy representation. "
            "Install scipy for memory-efficient sparse adjacency support.",
            UserWarning,
            stacklevel=2,
        )
        _WARNED_SCIPY = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_sparse_beneficial(G: Graph, density_threshold: float = 0.1) -> bool:
    """Return True if sparse storage would be beneficial.

    Heuristic: density < threshold AND n_agents >= 20.
    density = n_edges / (n_agents * (n_agents - 1))

    Args:
        G: The graph to evaluate.
        density_threshold: Maximum density to consider sparse beneficial (default 0.1).

    Returns:
        True if sparse representation is recommended, False otherwise.
    """
    n = G.n_agents
    if n < 20:
        return False
    max_edges = n * (n - 1)
    if max_edges == 0:
        return False
    n_edges = int(np.count_nonzero(G.adjacency))
    density = n_edges / max_edges
    return density < density_threshold


def to_sparse_adjacency(G: Graph) -> Any:
    """Convert Graph.adjacency to scipy.sparse.csr_matrix.

    Returns csr_matrix if scipy is available, otherwise returns G.adjacency (dense).
    Warns if scipy is not installed (once, using warnings.warn with stacklevel=2).

    Args:
        G: The graph whose adjacency matrix to convert.

    Returns:
        scipy.sparse.csr_matrix if scipy is installed, else np.ndarray.
    """
    if not _SCIPY_AVAILABLE:
        _warn_scipy_once()
        return G.adjacency
    return _sp.csr_matrix(G.adjacency)


def from_sparse_adjacency(
    sparse_adj: Any,
    agent_ids: tuple[str, ...],
    capabilities: np.ndarray,
) -> Graph:
    """Create a Graph from a sparse adjacency matrix.

    Converts to dense numpy before creating Graph (Graph stores dense internally).

    Args:
        sparse_adj: A scipy.sparse matrix or a dense np.ndarray.
        agent_ids: Ordered tuple of agent ID strings.
        capabilities: (n, d_cap) capability feature array.

    Returns:
        A new Graph with a dense adjacency array.
    """
    if _SCIPY_AVAILABLE and _sp.issparse(sparse_adj):
        dense_adj: np.ndarray = np.asarray(sparse_adj.todense(), dtype=float)
    else:
        dense_adj = np.asarray(sparse_adj, dtype=float)

    return Graph(
        agent_ids=agent_ids,
        adjacency=dense_adj,
        capabilities=np.asarray(capabilities, dtype=float),
    )


def sparse_graph_features(G: Graph, d_features: int = 16) -> np.ndarray:
    """Extract graph features using sparse eigenvalue decomposition.

    Uses scipy.sparse.linalg.eigsh for O(k * n) eigenvalue computation
    instead of O(n^3) dense eigvalsh.

    Falls back to emergo.features.extract_graph_features when:
    - scipy is not installed
    - n_agents < 20 (dense is faster for small graphs)
    - graph is not actually sparse

    Args:
        G: The graph to extract features from.
        d_features: Size of the output feature vector.

    Returns:
        Feature vector of length d_features, dtype float64, values in [-1, 1].
        Same semantics as emergo.features.extract_graph_features().
    """
    from emergo.features import extract_graph_features

    # Always use dense for small graphs or if scipy unavailable
    if not _SCIPY_AVAILABLE or G.n_agents < 20 or not is_sparse_beneficial(G):
        return extract_graph_features(G, d_features)

    n = G.n_agents
    k_eig = max(1, d_features // 2)
    k_met = d_features - k_eig

    # Build symmetric sparse adjacency
    sym_dense = (G.adjacency + G.adjacency.T) / 2.0
    sym_sparse = _sp.csr_matrix(sym_dense)

    # Compute sparse eigenvalues — k must be < n for eigsh
    k = min(k_eig, n - 1)
    spectral = np.zeros(k_eig)

    if k >= 1:
        try:
            # which='LM': largest magnitude eigenvalues; return_eigenvectors=False
            eig_vals: np.ndarray = _spl.eigsh(
                sym_sparse, k=k, which="LM", return_eigenvectors=False
            )
            # Sort descending by magnitude (largest first)
            eig_vals = eig_vals[np.argsort(np.abs(eig_vals))[::-1]]

            max_abs = float(np.abs(eig_vals).max())
            if max_abs > 1e-12:
                norm_eigs = eig_vals / max_abs
            else:
                norm_eigs = eig_vals

            copy_k = min(k_eig, len(norm_eigs))
            spectral[:copy_k] = norm_eigs[:copy_k]
        except Exception:
            # eigsh can fail for degenerate cases; fall back gracefully
            return extract_graph_features(G, d_features)

    # Reuse dense structural metrics (we need all eigenvalues for spectral entropy)
    from emergo.features import _structural_metrics

    all_eigs_dense = np.linalg.eigvalsh(sym_dense)[::-1]

    structural = np.zeros(k_met)
    if k_met > 0:
        raw_metrics = _structural_metrics(G, n, all_eigs_dense)
        copy_k = min(k_met, len(raw_metrics))
        structural[:copy_k] = raw_metrics[:copy_k]

    out = np.concatenate([spectral, structural])
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.clip(out, -1.0, 1.0)
    result: np.ndarray = out.astype(float)
    return result


def estimate_memory_bytes(n_agents: int, density: float) -> dict[str, int]:
    """Estimate memory usage for dense vs sparse adjacency.

    Dense: n_agents^2 * 8 bytes (float64)
    Sparse: n_edges * (8 + 4) bytes approx (data + col_indices, CSR format)
    where n_edges = n_agents * (n_agents-1) * density

    Args:
        n_agents: Number of agents in the graph.
        density: Edge density in [0, 1].

    Returns:
        {"dense_bytes": int, "sparse_bytes": int, "savings_bytes": int}
    """
    dense_bytes = n_agents * n_agents * 8  # float64 = 8 bytes per element

    n_edges = int(n_agents * (n_agents - 1) * density)
    # CSR: data array (float64 = 8 bytes) + col_indices (int32 = 4 bytes) per nonzero
    sparse_bytes = n_edges * (8 + 4)

    savings_bytes = max(0, dense_bytes - sparse_bytes)

    return {
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "savings_bytes": savings_bytes,
    }
