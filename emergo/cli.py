"""CLI entry points installed by pyproject.toml [project.scripts].

Commands:
  emergo-demo    — 5-agent ring graph, 500 iterations, prints summary
  emergo-kernel  — configurable run (mirrors examples/convergence_demo.py)
  emergo-health  — run + print health-check report with failure detectors

These are thin wrappers that use only the public emergo API; no examples/
dependency.  They serve as "does my installation work?" smoke-tests.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional


def demo() -> None:
    """Quick smoke-test: 5-agent ring, 500 iterations, print summary."""
    import numpy as np

    from emergo import (
        Graph,
        HistoryObserver,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
    )

    n = 5
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)

    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=0)
    A0 = make_initial_authority(ids, baseline=0.5)

    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=500,
        observers=[obs],
        rng=np.random.default_rng(0),
    )
    _, _, A_final, _ = final_state

    print("emergo-demo")
    print(f"  Termination      : {reason}")
    print(f"  CE acceptance    : {obs.ce_acceptance_rate:.1%}")
    if obs.phi_losses:
        losses = [l for _, l in obs.phi_losses]
        print(f"  φ loss (first→last): {losses[0]:.4f} → {losses[-1]:.4f}")
    print("  Final authority  :")
    for aid in sorted(A_final.scores):
        score = A_final.get(aid)
        bar = "█" * int(score * 20)
        print(f"    {aid}: {score:.3f}  |{bar:<20}|")
    print("\nInstallation OK.")


def run_kernel() -> None:
    """Configurable kernel run (--agents N --iterations N --optimizer sgd|adam)."""
    import numpy as np

    from emergo import (
        Graph,
        HistoryObserver,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
    )

    parser = argparse.ArgumentParser(
        prog="emergo-kernel",
        description="Run the Emergo kernel on a ring graph.",
    )
    parser.add_argument("--agents",      type=int,   default=5,     help="Number of agents")
    parser.add_argument("--iterations",  type=int,   default=1000,  help="Max iterations")
    parser.add_argument("--seed",        type=int,   default=0,     help="RNG seed")
    parser.add_argument("--optimizer",   choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--phi-lr",      type=float, default=None,  help="phi_update lr")
    parser.add_argument("--threshold",   type=float, default=1e-4,  help="Convergence threshold")
    args = parser.parse_args()

    n = args.agents
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)

    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=args.seed)
    A0 = make_initial_authority(ids, baseline=0.5)

    obs = HistoryObserver()

    print(f"Running Emergo kernel: {n} agents, max {args.iterations} iterations, "
          f"optimizer={args.optimizer}")

    final_state, reason = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=args.iterations,
        convergence_threshold=args.threshold,
        phi_optimizer=args.optimizer,
        phi_lr=args.phi_lr,
        observers=[obs],
        rng=np.random.default_rng(args.seed),
    )
    _, _, A_final, _ = final_state

    print(f"Termination: {reason}")
    print(f"CE acceptance rate: {obs.ce_acceptance_rate:.1%}")
    if obs.phi_losses:
        losses = [l for _, l in obs.phi_losses]
        print(f"φ loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    print("Final authority:")
    for aid in sorted(A_final.scores):
        print(f"  {aid}: {A_final.get(aid):.4f}")


def health_check() -> None:
    """Run kernel with diagnostics and print failure-detector report."""
    import numpy as np

    from emergo import (
        Graph,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
        run_health_check,
    )

    parser = argparse.ArgumentParser(
        prog="emergo-health",
        description="Run kernel and print health-check report.",
    )
    parser.add_argument("--agents",     type=int, default=5,   help="Number of agents")
    parser.add_argument("--iterations", type=int, default=200, help="Max iterations")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    n = args.agents
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)

    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=args.seed)
    A0 = make_initial_authority(ids, baseline=0.5)

    print(f"Running health check: {n} agents, {args.iterations} iterations …")

    result = emergo_kernel(
        initial_state=(G0, phi0, A0, []),
        max_iterations=args.iterations,
        collect_diagnostics=True,
        rng=np.random.default_rng(args.seed),
    )
    final_state, reason, diag = result

    print(f"Termination: {reason}\n")

    try:
        from emergo.visualize import print_health_report
        results = run_health_check(diag, final_state)
        print_health_report(results)
    except Exception as exc:
        print(f"Health check error: {exc}", file=sys.stderr)
        sys.exit(1)
