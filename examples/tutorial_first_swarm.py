#!/usr/bin/env python3
"""Tutorial: Build Your First Emergo Swarm.

This script demonstrates how to set up an 8-agent mesh network, run the Emergo
kernel, interpret authority evolution, and run a health check.

Run:
    python examples/tutorial_first_swarm.py
    python examples/tutorial_first_swarm.py --viz   # requires pip install "emergo[viz]"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    Graph,
    HistoryObserver,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
    run_health_check,
)

# ---------------------------------------------------------------------------
# 1. Build an 8-agent mesh graph (density ~0.3)
# ---------------------------------------------------------------------------


def build_mesh_graph(n: int = 8, density: float = 0.3, seed: int = 42) -> Graph:
    """Build a random sparse directed graph with the given edge density."""
    rng = np.random.default_rng(seed)
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                adj[i, j] = rng.uniform(0.2, 0.8)

    # 4-dimensional capability vectors per agent
    caps = rng.uniform(0.3, 0.7, (n, 4))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def print_banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_authority_table(A, title: str = "Authority Scores") -> None:
    """Print a simple bar-chart table of agent authority scores."""
    print(f"\n{title}")
    print("-" * 44)
    ranked = sorted(A.scores.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (agent_id, score) in enumerate(ranked, 1):
        bar = "█" * int(score * 24)
        print(f"  {rank:2d}. {agent_id:12s}  {score:.3f}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="First Emergo Swarm tutorial")
    parser.add_argument(
        "--viz", action="store_true", help="Generate plots (requires pip install 'emergo[viz]')"
    )
    parser.add_argument("--agents", type=int, default=8, help="Number of agents")
    parser.add_argument("--iterations", type=int, default=1000, help="Max kernel iterations")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Build graph
    # -----------------------------------------------------------------------
    print_banner("Step 1: Build Mesh Graph")
    G0 = build_mesh_graph(n=args.agents, density=0.3, seed=42)
    edge_count = int((G0.adjacency > 0).sum())
    print(f"  Agents   : {G0.n_agents}")
    print(f"  Edges    : {edge_count}")
    print(f"  Density  : {edge_count / (G0.n_agents * (G0.n_agents - 1)):.1%}")

    # -----------------------------------------------------------------------
    # 2. Initialize phi and authority
    # -----------------------------------------------------------------------
    print_banner("Step 2: Initialize phi and Authority")
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=42)
    A0 = make_initial_authority(G0.agent_ids, baseline=0.5)
    print(f"  Latent dims  : {phi0.d_latent}")
    print(f"  Feature dims : {phi0.d_features}")
    print(f"  CE dims      : {phi0.d_ce}")
    print(f"  Baseline auth: {A0.baseline}")

    # -----------------------------------------------------------------------
    # 3. Attach HistoryObserver
    # -----------------------------------------------------------------------
    print_banner("Step 3: Attach HistoryObserver")
    obs = HistoryObserver()
    print("  HistoryObserver attached — will record per-iteration metrics.")

    # -----------------------------------------------------------------------
    # 4. Run kernel
    # -----------------------------------------------------------------------
    print_banner(f"Step 4: Run Kernel ({args.iterations} iterations)")
    final_state, reason, diag = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=args.iterations,
        observers=[obs],
        collect_diagnostics=True,
    )

    _G_final, _phi_final, A_final, _E_history = final_state
    n_accepted = sum(1 for _, ok in obs._ce_history if ok)
    print(f"  Termination  : {reason}")
    print(f"  Accepted CEs : {n_accepted} / {len(obs._ce_history)}")
    print(f"  Accept rate  : {obs.ce_acceptance_rate:.1%}")
    if obs.mean_errors:
        print(f"  Phi error    : {obs.mean_errors[0]:.4f} → {obs.mean_errors[-1]:.4f}")

    # -----------------------------------------------------------------------
    # 5. Interpret output
    # -----------------------------------------------------------------------
    print_banner("Step 5: Final Authority Ranking")
    print_authority_table(A_final, title="Who earned authority?")

    top_agent = max(A_final.scores, key=A_final.scores.get)
    top_score = A_final.get(top_agent)
    print(f"  Top performer : {top_agent} ({top_score:.3f})")
    print(
        "  Interpretation: agents that accurately predicted how their CE\n"
        "  proposals would reshape the graph accumulated higher authority.\n"
        "  The kernel's default proposal generator samples CEs proportionally\n"
        "  to authority, creating a positive feedback loop stabilised by phi."
    )

    # -----------------------------------------------------------------------
    # 6. Health checks
    # -----------------------------------------------------------------------
    print_banner("Step 6: Health Check (8 Failure-Mode Detectors)")
    results = run_health_check(diag, final_state)
    triggered = [r for r in results if r.triggered]
    ok_count = len(results) - len(triggered)
    print(f"  Passed  : {ok_count}/{len(results)}")
    print(f"  Warnings: {len(triggered)}/{len(results)}")
    for r in results:
        status = "WARN" if r.triggered else " OK "
        print(f"    [{status}] {r.detector_name}: {r.message}")

    # -----------------------------------------------------------------------
    # 7. Visualize (optional)
    # -----------------------------------------------------------------------
    if args.viz:
        print_banner("Step 7: Visualizations")
        try:
            from emergo.visualize import render_health_dashboard

            output_dir = Path("./tutorial_plots")
            render_health_dashboard(diag, final_state, output_dir=str(output_dir))
            print(f"  Plots written to {output_dir.resolve()}/")
        except ImportError:
            print("  matplotlib/networkx not installed.")
            print("  Install with: pip install 'emergo[viz]'")
        except Exception as exc:
            print(f"  Visualization error: {exc}")
    else:
        print("\n  Tip: run with --viz to generate authority/topology/phi-loss plots.")

    print_banner("Done")
    print(f"  Swarm of {G0.n_agents} agents ran for up to {args.iterations} iterations.")
    print(f"  Termination: {reason}")
    print()


if __name__ == "__main__":
    main()
