"""Tests for emergo/features.py — fixed-dimensional feature extraction."""
import numpy as np
import pytest

from emergo.features import extract_graph_features, encode_ce
from emergo.types import CoordinationEvent, Graph


def _make_graph(n: int, seed: int = 0, sparse: bool = False) -> Graph:
    rng = np.random.default_rng(seed)
    adj = rng.uniform(0, 1, (n, n))
    if sparse:
        adj *= rng.random((n, n)) > 0.7
    np.fill_diagonal(adj, 0)
    caps = rng.uniform(0, 1, (n, 4))
    ids = tuple(f"a{i}" for i in range(n))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _make_empty_graph() -> Graph:
    return Graph(agent_ids=(), adjacency=np.zeros((0, 0)), capabilities=np.zeros((0, 4)))


def _make_ce(event_type: str, participants: tuple, **params) -> CoordinationEvent:
    return CoordinationEvent(
        event_type=event_type,
        participants=participants,
        params=frozenset(params.items()),
    )


class TestDimensionalityInvariant:
    @pytest.mark.parametrize("n_agents", [1, 2, 3, 5, 10, 20, 50, 100])
    @pytest.mark.parametrize("d_features", [4, 8, 16, 32])
    def test_output_shape_is_always_d_features(self, n_agents, d_features):
        G = _make_graph(n_agents)
        out = extract_graph_features(G, d_features)
        assert out.shape == (d_features,), (
            f"n_agents={n_agents} d_features={d_features}: got shape {out.shape}"
        )

    def test_empty_graph_returns_d_features(self):
        G = _make_empty_graph()
        for d in [4, 8, 16]:
            out = extract_graph_features(G, d)
            assert out.shape == (d,)

    def test_single_agent_graph(self):
        adj = np.zeros((1, 1))
        caps = np.ones((1, 4)) * 0.5
        G = Graph(agent_ids=("solo",), adjacency=adj, capabilities=caps)
        out = extract_graph_features(G, 16)
        assert out.shape == (16,)

    def test_dynamic_agent_count_features_consistent_shape(self):
        """Graphs of different sizes always produce the same (d_features,) shape."""
        d = 16
        shapes = set()
        for n in [3, 5, 7, 10]:
            G = _make_graph(n)
            out = extract_graph_features(G, d)
            shapes.add(out.shape)
        assert shapes == {(d,)}


class TestNoNanOrInf:
    @pytest.mark.parametrize("n_agents", [0, 1, 2, 5, 20])
    def test_no_nan(self, n_agents):
        G = _make_graph(n_agents) if n_agents > 0 else _make_empty_graph()
        out = extract_graph_features(G, 16)
        assert not np.any(np.isnan(out)), f"NaN in output for n_agents={n_agents}"

    @pytest.mark.parametrize("n_agents", [0, 1, 2, 5, 20])
    def test_no_inf(self, n_agents):
        G = _make_graph(n_agents) if n_agents > 0 else _make_empty_graph()
        out = extract_graph_features(G, 16)
        assert not np.any(np.isinf(out)), f"Inf in output for n_agents={n_agents}"

    def test_all_zeros_adjacency_no_nan(self):
        n = 5
        G = Graph(
            agent_ids=tuple(f"a{i}" for i in range(n)),
            adjacency=np.zeros((n, n)),
            capabilities=np.zeros((n, 4)),
        )
        out = extract_graph_features(G, 16)
        assert np.all(np.isfinite(out))

    def test_identity_adjacency_no_nan(self):
        n = 4
        G = Graph(
            agent_ids=tuple(f"a{i}" for i in range(n)),
            adjacency=np.eye(n),
            capabilities=np.ones((n, 4)),
        )
        out = extract_graph_features(G, 16)
        assert np.all(np.isfinite(out))


class TestBoundedFeatures:
    @pytest.mark.parametrize("n_agents", [1, 3, 10, 50])
    def test_max_abs_feature_le_1(self, n_agents):
        G = _make_graph(n_agents)
        out = extract_graph_features(G, 16)
        assert float(np.abs(out).max()) <= 1.0 + 1e-9, (
            f"Feature exceeds ±1 for n_agents={n_agents}: max={np.abs(out).max()}"
        )

    def test_bounded_for_dense_graph(self):
        n = 10
        G = Graph(
            agent_ids=tuple(f"a{i}" for i in range(n)),
            adjacency=np.ones((n, n)) - np.eye(n),
            capabilities=np.ones((n, 4)),
        )
        out = extract_graph_features(G, 16)
        assert np.all(np.abs(out) <= 1.0 + 1e-9)

    def test_bounded_for_sparse_graph(self):
        G = _make_graph(10, sparse=True)
        out = extract_graph_features(G, 16)
        assert np.all(np.abs(out) <= 1.0 + 1e-9)

    def test_bounded_large_graph(self):
        G = _make_graph(100)
        out = extract_graph_features(G, 32)
        assert np.all(np.abs(out) <= 1.0 + 1e-9)


class TestSpectralLayout:
    def test_first_half_are_spectral_eigenvalues(self):
        """First d//2 features should be consistent with top eigenvalues."""
        n, d = 5, 16
        G = _make_graph(n, seed=42)
        out = extract_graph_features(G, d)
        k = d // 2

        sym_adj = (G.adjacency + G.adjacency.T) / 2.0
        eigs = np.sort(np.linalg.eigvalsh(sym_adj))[::-1]
        max_abs = float(np.abs(eigs).max())
        if max_abs > 1e-12:
            norm_eigs = eigs / max_abs
        else:
            norm_eigs = eigs

        # The graph has n eigenvalues; remaining positions up to k are zero-padded
        compare_k = min(k, n)
        np.testing.assert_allclose(out[:compare_k], norm_eigs[:compare_k], atol=1e-9)
        np.testing.assert_array_equal(out[compare_k:k], 0.0)

    def test_spectral_features_change_with_topology(self):
        """Different graphs must produce different feature vectors."""
        d = 16
        G1 = _make_graph(5, seed=1)
        G2 = _make_graph(5, seed=2)
        out1 = extract_graph_features(G1, d)
        out2 = extract_graph_features(G2, d)
        assert not np.allclose(out1, out2)


class TestEncodeCE:
    @pytest.mark.parametrize("d_ce", [1, 2, 4, 8])
    def test_output_shape(self, d_ce):
        ce = _make_ce("add_edge", ("a0", "a1"), weight=0.5)
        enc = encode_ce(ce, d_ce)
        assert enc.shape == (d_ce,)

    def test_known_event_type_index(self):
        ce = _make_ce("add_edge", ("a0",))
        enc = encode_ce(ce, 4)
        # add_edge → index 0, normalized by n_types=8 → 0.0
        assert enc[0] == pytest.approx(0.0 / 8)

    def test_unknown_event_type_gets_minus_one_normalized(self):
        ce = _make_ce("unknown_type", ("a0",))
        enc = encode_ce(ce, 4)
        assert enc[0] == pytest.approx(-1.0 / 8)

    def test_no_nan_in_encoding(self):
        ce = _make_ce("execute_task", ("a0", "a1", "a2"), weight=0.3)
        for d in [1, 2, 4, 8]:
            enc = encode_ce(ce, d)
            assert np.all(np.isfinite(enc))

    def test_zero_d_ce_returns_empty(self):
        ce = _make_ce("add_edge", ("a0",))
        enc = encode_ce(ce, 0)
        assert enc.shape == (0,)
