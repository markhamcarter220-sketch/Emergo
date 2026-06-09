#!/usr/bin/env python3
"""Convergence demonstration: authority shifts to better predictors over time.

This script runs the Emergo kernel on a 5-agent ring graph for 500 iterations
with a HistoryObserver attached.  It then prints:

  - Final authority distribution (who "won" prediction authority)
  - CE acceptance rate
  - phi_loss trajectory (first, middle, last)
  - ASCII authority-vs-iteration table (sampled rows)

No matplotlib required; all output is plain text.

Run:
    python examples/convergence_demo.py
    python examples/convergence_demo.py --agents 8 --iterations 1000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    Graph,
    HistoryObserver,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
    run_health_check,
)
from emergo.visualize import print_health_report


def build_ring_graph(n: int) -> Graph:
    """n-agent directed ring: agent_i → agent_{(i+1) % n}."""
    ids = tuple(f"agent{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4), dtype=float) * 0.5
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def print_authority_table(obs: HistoryObserver, n_agents: int, sample_rows: int = 10) -> None:
    history = obs.authority_history
    if not history:
        print("  (no authority history recorded)")
        return

    total = len(history)
    step = max(1, total // sample_rows)
    sampled = history[::step]

    # Header
    ids = list(sampled[0].keys())
    col_w = 8
    header = f"{'iter':>6}  " + "  ".join(f"{a:>{col_w}}" for a in ids)
    print(header)
    print("-" * len(header))

    for row_idx, snap in enumerate(sampled):
        iter_num = row_idx * step
        vals = "  ".join(f"{snap.get(a, 0.0):>{col_w}.4f}" for a in ids)
        print(f"{iter_num:>6}  {vals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergo convergence demo")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--iterations", type=int, default=500, help="Max iterations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--phi-lr", type=float, default=None)
    parser.add_argument("--no-health", action="store_true", help="Skip health report")
    args = parser.parse_args()

    n = args.agents
    print(
        f"=== Emergo Convergence Demo  agents={n}  iterations={args.iterations}  "
        f"optimizer={args.optimizer}  seed={args.seed} ===\n"
    )

    G = build_ring_graph(n)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=args.seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    initial_state = (G, phi, A, [])

    obs = HistoryObserver()

    result = emergo_kernel(
        initial_state,
        max_iterations=args.iterations,
        observers=[obs],
        phi_optimizer=args.optimizer,
        phi_lr=args.phi_lr,
        collect_diagnostics=not args.no_health,
        rng=np.random.default_rng(args.seed),
    )

    if args.no_health:
        final_state, reason = result
        diag = None
    else:
        final_state, reason, diag = result

    _G_final, _phi_final, A_final, _E_history = final_state

    print(f"Termination: {reason}")
    print(f"Iterations seen by observer: {len(obs._ce_history)}")
    print(f"CE acceptance rate: {obs.ce_acceptance_rate:.1%}")
    if obs.phi_losses:
        losses = [loss for _, loss in obs.phi_losses]
        print(
            f"phi_loss: first={losses[0]:.6f}  "
            f"mid={losses[len(losses)//2]:.6f}  "
            f"last={losses[-1]:.6f}"
        )
    print()

    print("--- Final authority scores ---")
    for aid in sorted(A_final.scores):
        bar_len = int(A_final.get(aid) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {aid:>10}: {A_final.get(aid):.4f}  |{bar}|")
    print()

    print("--- Authority history (sampled) ---")
    print_authority_table(obs, n)
    print()

    summary = obs.summary()
    print("--- HistoryObserver summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    if diag is not None and not args.no_health:
        print("--- Health check ---")
        health_results = run_health_check(diag, final_state)
        print_health_report(health_results)


if __name__ == "__main__":
    main()
