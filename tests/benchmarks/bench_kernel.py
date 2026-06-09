#!/usr/bin/env python3
"""Benchmark emergo_kernel throughput at various scales.

Measures:
- Iterations per second (accepted CEs per second)
- Wall time to convergence or max_iterations
- Memory usage (peak RSS via resource.getrusage)

Scales tested: 5, 10, 20, 50 agents (skips 100 if --quick flag set)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from emergo.kernel import emergo_kernel, make_initial_authority, make_initial_phi
from emergo.types import CoordinationEvent, Errors, Graph, State

# ---------------------------------------------------------------------------
# Observer to count accepted CEs
# ---------------------------------------------------------------------------


class _AcceptedCECounter:
    """Minimal observer that counts accepted coordination events."""

    def __init__(self) -> None:
        self.count = 0

    def on_ce_result(
        self,
        t: int,
        ce: CoordinationEvent,
        accepted: bool,
        errors: Errors | None,
    ) -> None:
        if accepted:
            self.count += 1

    def on_iteration_start(self, t: int, state: State) -> None:
        pass

    def on_phi_updated(self, t: int, loss: float) -> None:
        pass

    def on_kernel_done(self, reason: str, state: State, n_iter: int) -> None:
        pass


# ---------------------------------------------------------------------------
# State factory
# ---------------------------------------------------------------------------


def make_ring_state(n: int, seed: int = 0) -> State:
    """Create a ring topology with n agents.

    Ring: agent_i → agent_{(i+1) % n} with weight 0.5.
    """
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = rng.uniform(0.1, 1.0, (n, 4))
    agent_ids = tuple(f"agent_{i}" for i in range(n))
    G = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    authority = make_initial_authority(agent_ids, baseline=0.5)
    return G, phi, authority, []


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def benchmark_kernel(n_agents: int, max_iterations: int = 200, seed: int = 0) -> dict:
    """Run one benchmark configuration, return metrics dict.

    Returns:
        {
            "n_agents": int,
            "max_iterations": int,
            "wall_time_s": float,
            "iterations_run": int,  # actual iterations before convergence/limit
            "accepted_ces": int,
            "ces_per_second": float,
            "reason": str,          # "Converged" or "Max iterations reached"
            "peak_memory_mb": float,  # RSS memory peak
        }
    """
    initial_state = make_ring_state(n_agents, seed=seed)
    counter = _AcceptedCECounter()

    t0 = time.perf_counter()
    result = emergo_kernel(
        initial_state,
        max_iterations=max_iterations,
        observers=[counter],
        collect_diagnostics=False,
    )
    wall_time = time.perf_counter() - t0

    # Peak RSS after run
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    _final_state, reason = result

    accepted_ces = counter.count
    ces_per_second = accepted_ces / wall_time if wall_time > 0.0 else float("inf")

    # On Linux ru_maxrss is in KB; on macOS it is in bytes
    if sys.platform == "darwin":
        peak_memory_mb = rss_after_kb / (1024 * 1024)
    else:
        peak_memory_mb = rss_after_kb / 1024

    return {
        "n_agents": n_agents,
        "max_iterations": max_iterations,
        "wall_time_s": round(wall_time, 4),
        "iterations_run": max_iterations,  # kernel doesn't expose iteration count directly
        "accepted_ces": accepted_ces,
        "ces_per_second": round(ces_per_second, 2),
        "reason": reason,
        "peak_memory_mb": round(peak_memory_mb, 2),
    }


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def run_benchmarks(quick: bool = False) -> list[dict]:
    """Run the full benchmark suite and return results."""
    sizes = [5, 10, 20, 50] if quick else [5, 10, 20, 50, 100]

    results = []
    for n in sizes:
        print(f"  Benchmarking n={n} agents ...", flush=True, file=sys.stderr)
        metrics = benchmark_kernel(n_agents=n, max_iterations=200, seed=42)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_table(results: list[dict]) -> None:
    """Print results as an ASCII table."""
    headers = [
        "n_agents",
        "wall_time_s",
        "accepted_ces",
        "ces/sec",
        "reason",
        "peak_mem_mb",
    ]
    col_widths = [10, 12, 13, 10, 28, 12]

    # Header row
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "-" * len(header_line)
    print(sep)
    print(header_line)
    print(sep)

    for r in results:
        row = [
            str(r["n_agents"]).ljust(col_widths[0]),
            f"{r['wall_time_s']:.4f}".ljust(col_widths[1]),
            str(r["accepted_ces"]).ljust(col_widths[2]),
            f"{r['ces_per_second']:.2f}".ljust(col_widths[3]),
            r["reason"].ljust(col_widths[4]),
            f"{r['peak_memory_mb']:.2f}".ljust(col_widths[5]),
        ]
        print("  ".join(row))

    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergo kernel benchmark")
    parser.add_argument("--quick", action="store_true", help="Skip 100-agent test")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    print("Running Emergo kernel benchmarks ...", file=sys.stderr)
    results = run_benchmarks(quick=args.quick)

    if args.json:
        import json

        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
