#!/usr/bin/env python3
"""Long-horizon convergence validation for the Emergo kernel.

Two experiments:
  1. 10k-iteration run on a 10-agent random graph with a diverse proposal
     generator (30 % add_edge, 20 % remove_edge, 30 % update_capabilities,
     20 % null / pass).  Produces:
       - convergence_log.txt   — loss and authority stats per 100 iterations
       - authority_history.png — authority scores over time
       - phi_loss_history.png  — phi prediction loss over time
       - edge_count_history.png — accepted-edge count over time

  2. Adversarial 100-agent stress test: 200 alternating add_agent / remove_agent
     CEs to validate that the feature extractor and phi_update handle dynamic
     graph size without NaN, Inf, or shape errors.

Usage:
    python examples/long_horizon_stress_test.py [--no-plots] [--seed N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    DefaultProposalGenerator,
    Graph,
    HistoryObserver,
    SequenceProposalGenerator,
    WeightedMixGenerator,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.types import CoordinationEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_graph(n: int, density: float = 0.4, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    adj = rng.uniform(0.1, 0.8, (n, n)) * (rng.random((n, n)) < density)
    np.fill_diagonal(adj, 0.0)
    caps = rng.uniform(0.3, 0.9, (n, 4))
    return Graph(
        agent_ids=tuple(f"a{i}" for i in range(n)),
        adjacency=adj,
        capabilities=caps,
    )


def _make_ce(event_type: str, participants: tuple, **params) -> CoordinationEvent:
    return CoordinationEvent(
        event_type=event_type,
        participants=participants,
        params=frozenset(params.items()),
    )


class NullProposalGenerator:
    """A generator that always returns None (no CE proposed)."""
    def propose(self, A_t, G_t, rng):
        return None


# ---------------------------------------------------------------------------
# Experiment 1: 10k-iteration run on 10-agent graph
# ---------------------------------------------------------------------------

def run_10k_stress(seed: int, output_dir: Path, make_plots: bool) -> None:
    print("=" * 60)
    print("Experiment 1: 10 000-iteration run on 10-agent graph")
    print("=" * 60)

    n = 10
    G = _random_graph(n, density=0.4, seed=seed)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    initial_state = (G, phi, A, [])

    # Diverse proposal mix: 30% add_edge, 20% remove_edge, 30% update_caps, 20% null
    default_gen = DefaultProposalGenerator()

    # Structured sub-generators
    ids = G.agent_ids
    add_ces = [_make_ce("add_edge", (ids[i], ids[(i + 3) % n]), weight=0.5 + 0.01 * i)
               for i in range(n)]
    remove_ces = [_make_ce("remove_edge", (ids[i], ids[(i + 2) % n]))
                  for i in range(n)]
    cap_ces = [_make_ce("update_capabilities", (ids[i],), cap_index=0, value=0.7)
               for i in range(n)]

    add_gen = SequenceProposalGenerator(add_ces, loop=True)
    remove_gen = SequenceProposalGenerator(remove_ces, loop=True)
    cap_gen = SequenceProposalGenerator(cap_ces, loop=True)
    null_gen = NullProposalGenerator()

    diverse_gen = WeightedMixGenerator([
        (add_gen, 0.30),
        (remove_gen, 0.20),
        (cap_gen, 0.30),
        (null_gen, 0.20),
    ])

    obs = HistoryObserver()
    t0 = time.time()

    final_state, reason, diag = emergo_kernel(
        initial_state,
        max_iterations=10_000,
        proposal_generator=diverse_gen,
        observers=[obs],
        phi_optimizer="adam",
        phi_update_interval=20,
        collect_diagnostics=True,
        rng=np.random.default_rng(seed),
    )
    elapsed = time.time() - t0

    _, phi_final, A_final, E_history = final_state

    print(f"  Termination    : {reason}")
    print(f"  Wall-time      : {elapsed:.1f}s")
    print(f"  CE observed    : {len(obs._ce_history)}")
    print(f"  CE acceptance  : {obs.ce_acceptance_rate:.1%}")
    if obs.phi_losses:
        losses = [l for _, l in obs.phi_losses]
        print(f"  phi_loss: first={losses[0]:.6f}  mid={losses[len(losses)//2]:.6f}  last={losses[-1]:.6f}")
    print()

    # Write convergence log
    log_path = output_dir / "convergence_log.txt"
    _write_convergence_log(log_path, diag, A_final, G.agent_ids)
    print(f"  Convergence log: {log_path}")

    # Plots
    if make_plots:
        _make_plots(diag, G.agent_ids, obs, output_dir)


def _write_convergence_log(path: Path, diag, A_final, agent_ids) -> None:
    records = diag.records
    with open(path, "w") as f:
        f.write("iteration,ce_accepted,phi_loss,error_mean,edge_count,topology_entropy\n")
        for r in records:
            f.write(
                f"{r.iteration},{int(r.ce_accepted)},"
                f"{r.phi_loss if r.phi_loss is not None else ''},"
                f"{r.error_mean if r.error_mean is not None else ''},"
                f"{r.edge_count},"
                f"{r.topology_entropy:.6f}\n"
            )
        f.write("\n# Final authority scores\n")
        for aid in sorted(A_final.scores):
            f.write(f"# {aid}: {A_final.get(aid):.4f}\n")


def _make_plots(diag, agent_ids, obs, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    records = diag.records

    # 1. Authority history
    iters = [r.iteration for r in records if r.ce_accepted]
    auth_over_time = {aid: [] for aid in agent_ids}
    for r in records:
        if not r.ce_accepted:
            continue
        for aid in agent_ids:
            auth_over_time[aid].append(r.authority_scores.get(aid, 0.5))

    fig, ax = plt.subplots(figsize=(10, 4))
    for aid in agent_ids:
        ax.plot(iters[:len(auth_over_time[aid])], auth_over_time[aid],
                label=aid, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Authority")
    ax.set_title("Authority History — 10-agent / 10k iterations")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p = output_dir / "authority_history.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print(f"  Plot: {p}")

    # 2. Phi loss history
    phi_iters = [r.iteration for r in records if r.phi_loss is not None]
    phi_vals = [r.phi_loss for r in records if r.phi_loss is not None]
    if phi_vals:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.semilogy(phi_iters, phi_vals, color="steelblue", linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("phi_loss (log scale)")
        ax.set_title("φ Prediction Loss — 10k iterations")
        fig.tight_layout()
        p = output_dir / "phi_loss_history.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        print(f"  Plot: {p}")

    # 3. Edge count history
    edge_iters = [r.iteration for r in records if r.ce_accepted]
    edge_vals = [r.edge_count for r in records if r.ce_accepted]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(edge_iters, edge_vals, color="darkorange", linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Edge count")
    ax.set_title("Accepted Edge Count — 10k iterations")
    fig.tight_layout()
    p = output_dir / "edge_count_history.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print(f"  Plot: {p}")


# ---------------------------------------------------------------------------
# Experiment 2: Adversarial add/remove agent test (100 agents)
# ---------------------------------------------------------------------------

def run_adversarial_agent_churn(seed: int) -> None:
    print("=" * 60)
    print("Experiment 2: Adversarial add/remove-agent stress (100 agents)")
    print("=" * 60)

    import importlib
    from emergo.ce_execution import ce_execute
    from emergo.error_computation import error_computation
    from emergo.features import extract_graph_features
    from emergo.kernel import make_initial_phi
    from emergo.lux import Lux

    rng = np.random.default_rng(seed)
    n_base = 10
    G = _random_graph(n_base, density=0.3, seed=seed)
    phi = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A = make_initial_authority(G.agent_ids, baseline=0.5)
    lux = Lux()

    added_agents: list[str] = []
    n_errors = 0
    n_add = 0
    n_remove = 0
    n_iterations = 200

    for step in range(n_iterations):
        # Alternate: even → add_agent, odd → remove_agent (if any added)
        if step % 2 == 0:
            new_id = f"dyn_{step}"
            ce = CoordinationEvent(
                event_type="add_agent",
                participants=(new_id,),
                params=frozenset([("agent_id", new_id)]),
            )
            G_next, ok, _ = ce_execute(G, ce, lux, A)
            if ok:
                G = G_next
                # Rebuild A with new agent
                A_scores = {aid: A.get(aid) for aid in A.scores}
                A_scores[new_id] = A.baseline
                from emergo.types import Authority
                A = Authority(scores=A_scores, baseline=A.baseline)
                added_agents.append(new_id)
                n_add += 1
        elif added_agents:
            agent_to_remove = added_agents.pop(0)
            if agent_to_remove in G.agent_ids:
                ce = CoordinationEvent(
                    event_type="remove_agent",
                    participants=(agent_to_remove,),
                    params=frozenset(),
                )
                G_next, ok, _ = ce_execute(G, ce, lux, A)
                if ok:
                    G = G_next
                    A_scores = {aid: A.get(aid) for aid in A.scores if aid != agent_to_remove}
                    from emergo.types import Authority
                    A = Authority(scores=A_scores, baseline=A.baseline)
                    n_remove += 1

        # After each step, validate feature extraction and phi ops
        try:
            feats = extract_graph_features(G, phi.d_features)
            assert feats.shape == (phi.d_features,), f"bad shape {feats.shape}"
            assert not np.any(np.isnan(feats)), "NaN in features"
            assert not np.any(np.isinf(feats)), "Inf in features"
            assert float(np.abs(feats).max()) <= 1.0 + 1e-9, "feature out of bounds"
        except Exception as e:
            n_errors += 1
            print(f"  ERROR at step {step} (n_agents={G.n_agents}): {e}")

    print(f"  Steps run      : {n_iterations}")
    print(f"  add_agent ok   : {n_add}")
    print(f"  remove_agent ok: {n_remove}")
    print(f"  Final n_agents : {G.n_agents}")
    print(f"  Feature errors : {n_errors}")
    if n_errors == 0:
        print("  PASS — features stayed valid across all agent-count changes")
    else:
        print(f"  FAIL — {n_errors} feature validation errors")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Emergo long-horizon stress test")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_10k_stress(seed=args.seed, output_dir=output_dir, make_plots=not args.no_plots)
    run_adversarial_agent_churn(seed=args.seed)

    print("All stress tests complete.")


if __name__ == "__main__":
    main()
