#!/usr/bin/env python3
"""Benchmark graph feature extraction.

Compares:
- Dense numpy extract_graph_features()
- Sparse scipy sparse_graph_features() (if scipy installed)

Scales tested: 10, 50, 100, 200, 500 agents at 5% density
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from emergo.features import extract_graph_features
from emergo.sparse import estimate_memory_bytes, sparse_graph_features
from emergo.types import Graph

# ---------------------------------------------------------------------------
# Check scipy availability at benchmark time
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE = False
try:
    import scipy.sparse  # noqa: F401

    _SCIPY_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

_BENCH_DENSITY = 0.05


def make_sparse_graph(n: int, density: float = _BENCH_DENSITY, seed: int = 0) -> Graph:
    """Create a random sparse graph with approximately `density` fraction of edges."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                adj[i, j] = rng.uniform(0.1, 1.0)
    caps = rng.uniform(0.1, 1.0, (n, 4))
    agent_ids = tuple(f"a{i}" for i in range(n))
    return Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def benchmark_features(n_agents: int, d_features: int = 16, n_runs: int = 10) -> dict:
    """Run both dense and sparse feature extraction n_runs times.

    Returns:
        {
            "n_agents": int,
            "density": float,
            "dense_ms_mean": float,
            "dense_ms_std": float,
            "sparse_ms_mean": float | None,  # None if scipy not installed
            "sparse_ms_std": float | None,
            "sparse_speedup": float | None,  # dense_ms_mean / sparse_ms_mean
            "memory_savings_mb": float,       # from estimate_memory_bytes
        }
    """
    G = make_sparse_graph(n_agents, density=_BENCH_DENSITY, seed=0)

    # --- Dense timings ---
    dense_times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        extract_graph_features(G, d_features=d_features)
        dense_times_ms.append((time.perf_counter() - t0) * 1000.0)

    dense_ms_mean = float(np.mean(dense_times_ms))
    dense_ms_std = float(np.std(dense_times_ms))

    # --- Sparse timings (if scipy available) ---
    sparse_ms_mean: float | None = None
    sparse_ms_std: float | None = None
    sparse_speedup: float | None = None

    if _SCIPY_AVAILABLE:
        sparse_times_ms: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sparse_graph_features(G, d_features=d_features)
            sparse_times_ms.append((time.perf_counter() - t0) * 1000.0)

        sparse_ms_mean = float(np.mean(sparse_times_ms))
        sparse_ms_std = float(np.std(sparse_times_ms))
        if sparse_ms_mean > 0.0:
            sparse_speedup = round(dense_ms_mean / sparse_ms_mean, 3)

    # --- Memory estimate ---
    mem = estimate_memory_bytes(n_agents, density=_BENCH_DENSITY)
    memory_savings_mb = mem["savings_bytes"] / (1024 * 1024)

    return {
        "n_agents": n_agents,
        "density": _BENCH_DENSITY,
        "dense_ms_mean": round(dense_ms_mean, 4),
        "dense_ms_std": round(dense_ms_std, 4),
        "sparse_ms_mean": round(sparse_ms_mean, 4) if sparse_ms_mean is not None else None,
        "sparse_ms_std": round(sparse_ms_std, 4) if sparse_ms_std is not None else None,
        "sparse_speedup": sparse_speedup,
        "memory_savings_mb": round(memory_savings_mb, 4),
    }


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_benchmarks(sizes: list[int] | None = None, n_runs: int = 10) -> list[dict]:
    """Run the full benchmark suite and return results."""
    if sizes is None:
        sizes = [10, 50, 100, 200, 500]

    results = []
    for n in sizes:
        print(f"  Benchmarking n={n} agents ({n_runs} runs) ...", flush=True, file=sys.stderr)
        metrics = benchmark_features(n_agents=n, d_features=16, n_runs=n_runs)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_table(results: list[dict]) -> None:
    """Print results as an ASCII table."""
    scipy_avail = any(r["sparse_ms_mean"] is not None for r in results)

    headers = ["n_agents", "dense_ms", "dense_std"]
    col_widths = [10, 10, 10]
    if scipy_avail:
        headers += ["sparse_ms", "sparse_std", "speedup"]
        col_widths += [10, 10, 10]
    headers += ["mem_save_mb"]
    col_widths += [12]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "-" * len(header_line)
    print(sep)
    print(header_line)
    print(sep)

    for r in results:
        row = [
            str(r["n_agents"]).ljust(col_widths[0]),
            f"{r['dense_ms_mean']:.4f}".ljust(col_widths[1]),
            f"{r['dense_ms_std']:.4f}".ljust(col_widths[2]),
        ]
        if scipy_avail:
            sm = r["sparse_ms_mean"]
            ss = r["sparse_ms_std"]
            sp = r["sparse_speedup"]
            row += [
                (f"{sm:.4f}" if sm is not None else "N/A").ljust(col_widths[3]),
                (f"{ss:.4f}" if ss is not None else "N/A").ljust(col_widths[4]),
                (f"{sp:.3f}x" if sp is not None else "N/A").ljust(col_widths[5]),
            ]
        row += [f"{r['memory_savings_mb']:.4f}".ljust(col_widths[-1])]
        print("  ".join(row))

    print(sep)
    if not scipy_avail:
        print("  (scipy not installed — sparse benchmarks skipped)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergo feature extraction benchmark")
    parser.add_argument("--quick", action="store_true", help="Test smaller sizes only")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--runs", type=int, default=10, help="Number of timing runs per config")
    args = parser.parse_args()

    sizes = [10, 50, 100] if args.quick else [10, 50, 100, 200, 500]
    print(
        f"Running Emergo feature extraction benchmarks (scipy={'available' if _SCIPY_AVAILABLE else 'not installed'}) ...",
        file=sys.stderr,
    )
    results = run_benchmarks(sizes=sizes, n_runs=args.runs)

    if args.json:
        import json

        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
