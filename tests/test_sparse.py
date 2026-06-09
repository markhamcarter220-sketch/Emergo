"""Tests for emergo/sparse.py — sparse adjacency support."""

from __future__ import annotations

import sys
import unittest.mock

import numpy as np
import pytest

from emergo.types import Graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(n: int, density: float = 0.5, seed: int = 0) -> Graph:
    """Create a random graph with approximately the given edge density."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                adj[i, j] = rng.uniform(0.1, 1.0)
    caps = rng.uniform(0, 1, (n, 4))
    ids = tuple(f"a{i}" for i in range(n))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


# ---------------------------------------------------------------------------
# TestIsSparsebeneficial
# ---------------------------------------------------------------------------


class TestIsSparsebeneficial:
    def test_dense_small_graph_not_beneficial(self):
        """Small graphs (< 20 agents) are never considered sparse-beneficial."""
        from emergo.sparse import is_sparse_beneficial

        G = _make_graph(5, density=0.02)
        assert is_sparse_beneficial(G) is False

    def test_large_sparse_is_beneficial(self):
        """Large graphs with low density should be sparse-beneficial."""
        from emergo.sparse import is_sparse_beneficial

        G = _make_graph(50, density=0.02, seed=1)
        assert is_sparse_beneficial(G) is True

    def test_large_dense_not_beneficial(self):
        """Large graphs with high density should NOT be sparse-beneficial."""
        from emergo.sparse import is_sparse_beneficial

        G = _make_graph(50, density=0.80, seed=2)
        assert is_sparse_beneficial(G) is False

    def test_exactly_20_agents_with_low_density(self):
        """Boundary: exactly 20 agents with low density should be beneficial."""
        from emergo.sparse import is_sparse_beneficial

        G = _make_graph(20, density=0.02, seed=3)
        assert is_sparse_beneficial(G) is True

    def test_custom_density_threshold(self):
        """Custom density threshold should be respected."""
        from emergo.sparse import is_sparse_beneficial

        G = _make_graph(50, density=0.15, seed=4)
        # Default threshold is 0.1 — this graph should NOT be beneficial
        assert is_sparse_beneficial(G, density_threshold=0.1) is False
        # But with a higher threshold it should be
        assert is_sparse_beneficial(G, density_threshold=0.2) is True


# ---------------------------------------------------------------------------
# TestMemoryEstimate
# ---------------------------------------------------------------------------


class TestMemoryEstimate:
    def test_dense_larger_than_sparse_for_sparse_graph(self):
        """Dense memory should exceed sparse memory for a sparse graph."""
        from emergo.sparse import estimate_memory_bytes

        result = estimate_memory_bytes(n_agents=100, density=0.05)
        assert result["dense_bytes"] > result["sparse_bytes"]

    def test_savings_zero_for_dense_graph(self):
        """For a fully dense graph, sparse representation offers no savings."""
        from emergo.sparse import estimate_memory_bytes

        result = estimate_memory_bytes(n_agents=50, density=1.0)
        # CSR overhead per nonzero is 12 bytes vs 8 bytes for dense, so savings <= 0
        assert result["savings_bytes"] == 0

    def test_keys_present(self):
        """All expected keys must be present in the result dict."""
        from emergo.sparse import estimate_memory_bytes

        result = estimate_memory_bytes(n_agents=10, density=0.1)
        assert set(result.keys()) == {"dense_bytes", "sparse_bytes", "savings_bytes"}

    def test_dense_bytes_scales_quadratically(self):
        """Dense memory should scale as n^2."""
        from emergo.sparse import estimate_memory_bytes

        r10 = estimate_memory_bytes(n_agents=10, density=0.1)
        r20 = estimate_memory_bytes(n_agents=20, density=0.1)
        assert r20["dense_bytes"] == 4 * r10["dense_bytes"]

    def test_zero_density_means_zero_sparse_bytes(self):
        """Empty graph has zero sparse storage cost."""
        from emergo.sparse import estimate_memory_bytes

        result = estimate_memory_bytes(n_agents=50, density=0.0)
        assert result["sparse_bytes"] == 0


# ---------------------------------------------------------------------------
# TestToSparseAdjacency
# ---------------------------------------------------------------------------


class TestToSparseAdjacency:
    def test_returns_numpy_when_scipy_not_available(self):
        """When scipy is unavailable, to_sparse_adjacency returns a numpy array."""
        import emergo.sparse as sparse_mod

        G = _make_graph(5, density=0.4)
        mock_modules = {
            "scipy": None,
            "scipy.sparse": None,
            "scipy.sparse.linalg": None,
        }
        with unittest.mock.patch.dict(sys.modules, mock_modules):
            from importlib import reload

            reload(sparse_mod)
            result = sparse_mod.to_sparse_adjacency(G)
            assert isinstance(result, np.ndarray)
        # Reload back to normal state
        reload(sparse_mod)

    def test_warns_when_scipy_not_available(self):
        """A warning should be emitted when scipy is not installed."""
        import emergo.sparse as sparse_mod

        G = _make_graph(5, density=0.4)
        mock_modules = {
            "scipy": None,
            "scipy.sparse": None,
            "scipy.sparse.linalg": None,
        }
        with unittest.mock.patch.dict(sys.modules, mock_modules):
            from importlib import reload

            reload(sparse_mod)
            with pytest.warns(UserWarning, match="scipy is not installed"):
                sparse_mod.to_sparse_adjacency(G)
        reload(sparse_mod)

    def test_round_trip_preserves_values(self):
        """to_sparse_adjacency followed by todense should preserve values."""
        pytest.importorskip("scipy", reason="scipy not installed")
        import scipy.sparse as sp

        from emergo.sparse import to_sparse_adjacency

        G = _make_graph(8, density=0.3, seed=42)
        sparse_adj = to_sparse_adjacency(G)
        assert sp.issparse(sparse_adj)
        dense_back = np.asarray(sparse_adj.todense())
        np.testing.assert_allclose(dense_back, G.adjacency)

    def test_result_type_when_scipy_available(self):
        """to_sparse_adjacency should return a csr_matrix when scipy is installed."""
        pytest.importorskip("scipy", reason="scipy not installed")
        import scipy.sparse as sp

        from emergo.sparse import to_sparse_adjacency

        G = _make_graph(6, density=0.4)
        result = to_sparse_adjacency(G)
        assert sp.issparse(result)
        assert isinstance(result, sp.csr_matrix)


# ---------------------------------------------------------------------------
# TestFromSparseAdjacency
# ---------------------------------------------------------------------------


class TestFromSparseAdjacency:
    def test_creates_valid_graph(self):
        """from_sparse_adjacency should produce a valid Graph from a sparse matrix."""
        pytest.importorskip("scipy", reason="scipy not installed")
        import scipy.sparse as sp

        from emergo.sparse import from_sparse_adjacency

        n = 5
        adj_dense = np.eye(n, dtype=float)
        sparse_adj = sp.csr_matrix(adj_dense)
        caps = np.ones((n, 2), dtype=float)
        ids = tuple(f"x{i}" for i in range(n))

        G = from_sparse_adjacency(sparse_adj, ids, caps)

        assert isinstance(G, Graph)
        assert G.agent_ids == ids
        assert G.n_agents == n
        np.testing.assert_allclose(G.adjacency, adj_dense)

    def test_from_dense_array_also_works(self):
        """from_sparse_adjacency should accept a plain np.ndarray too."""
        from emergo.sparse import from_sparse_adjacency

        n = 4
        adj = np.random.default_rng(0).uniform(0, 1, (n, n))
        caps = np.ones((n, 2))
        ids = tuple(f"y{i}" for i in range(n))

        G = from_sparse_adjacency(adj, ids, caps)
        assert isinstance(G, Graph)
        np.testing.assert_allclose(G.adjacency, adj)


# ---------------------------------------------------------------------------
# TestSparseGraphFeatures
# ---------------------------------------------------------------------------


class TestSparseGraphFeatures:
    def test_output_shape_matches_d_features(self):
        """Output must be a 1D array of length d_features."""
        from emergo.sparse import sparse_graph_features

        G = _make_graph(10, density=0.3)
        for d in [4, 8, 16, 32]:
            out = sparse_graph_features(G, d_features=d)
            assert out.shape == (d,), f"Expected ({d},), got {out.shape}"

    def test_values_in_valid_range(self):
        """All output values must be in [-1, 1]."""
        from emergo.sparse import sparse_graph_features

        G = _make_graph(30, density=0.05, seed=7)
        out = sparse_graph_features(G, d_features=16)
        assert np.all(out >= -1.0 - 1e-9), "Values below -1 found"
        assert np.all(out <= 1.0 + 1e-9), "Values above +1 found"

    def test_no_nan_or_inf(self):
        """Output must not contain NaN or Inf."""
        from emergo.sparse import sparse_graph_features

        G = _make_graph(25, density=0.04, seed=99)
        out = sparse_graph_features(G, d_features=16)
        assert not np.any(np.isnan(out)), "NaN found in sparse features"
        assert not np.any(np.isinf(out)), "Inf found in sparse features"

    def test_fallback_when_scipy_unavailable(self):
        """When scipy is not available, sparse_graph_features falls back to dense."""
        from emergo.features import extract_graph_features
        import emergo.sparse as sparse_mod

        G = _make_graph(30, density=0.04, seed=5)
        mock_modules = {
            "scipy": None,
            "scipy.sparse": None,
            "scipy.sparse.linalg": None,
        }
        with unittest.mock.patch.dict(sys.modules, mock_modules):
            from importlib import reload

            reload(sparse_mod)
            out = sparse_mod.sparse_graph_features(G, d_features=16)
            expected = extract_graph_features(G, d_features=16)
            np.testing.assert_allclose(out, expected)
        reload(sparse_mod)

    def test_fallback_for_small_graph(self):
        """Graphs with n_agents < 20 always use the dense path."""
        from emergo.features import extract_graph_features
        from emergo.sparse import sparse_graph_features

        G = _make_graph(10, density=0.3, seed=11)
        sparse_out = sparse_graph_features(G, d_features=16)
        dense_out = extract_graph_features(G, d_features=16)
        np.testing.assert_allclose(sparse_out, dense_out)

    def test_fallback_for_dense_large_graph(self):
        """Dense large graphs should also fall back to dense path."""
        from emergo.features import extract_graph_features
        from emergo.sparse import sparse_graph_features

        # 50 agents, 80% density — not sparse-beneficial
        G = _make_graph(50, density=0.80, seed=22)
        sparse_out = sparse_graph_features(G, d_features=16)
        dense_out = extract_graph_features(G, d_features=16)
        np.testing.assert_allclose(sparse_out, dense_out)

    def test_output_dtype_is_float64(self):
        """Output dtype must be float64."""
        from emergo.sparse import sparse_graph_features

        G = _make_graph(8, density=0.3)
        out = sparse_graph_features(G, d_features=16)
        assert out.dtype == np.float64
