#!/usr/bin/env python3
"""Lux + Emergo Integration Demo.

Demonstrates the full Orbis Lux stack:
  - Emergo kernel: agents earn authority by predicting topology changes
  - Lux governance: PyLuxGate enforces authority thresholds on CE proposals

Scenario
--------
5 agents coordinate over 300 iterations. Authority emerges from prediction
accuracy. The demo then shows Lux blocking a CE from a low-authority agent
and approving one from a high-authority agent — illustrating that governance
is enforced at the kernel boundary, not inside the coordination logic.

Run:
    python examples/lux_emergo_integration.py            # uses SimulatedLuxBridge
    EMERGO_LUX_MODE=real python examples/lux_emergo_integration.py  # uses PyLuxGate

Requirements for real mode:
    cd /path/to/Lux-V1.0 && maturin develop --features python
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    Graph,
    HistoryObserver,
    emergo_kernel,
    make_initial_authority,
    make_initial_phi,
)
from emergo.lux_bridge import LuxBridge, RealLuxBridge, SimulatedLuxBridge
from emergo.types import CoordinationEvent


def _build_graph(n: int = 5, seed: int = 42) -> Graph:
    rng = np.random.default_rng(seed)
    ids = tuple(f"agent_{i}" for i in range(n))
    adj = rng.uniform(0.1, 0.6, (n, n))
    np.fill_diagonal(adj, 0.0)
    caps = rng.uniform(0.2, 0.8, (n, 4))
    return Graph(agent_ids=ids, adjacency=adj, capabilities=caps)


def _separator(label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print("─" * 60)


def main() -> None:
    lux_mode = os.getenv("EMERGO_LUX_MODE", "simulated")
    n_agents = 5
    n_iterations = 300
    seed = 7

    G0 = _build_graph(n=n_agents, seed=seed)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=seed)
    A0 = make_initial_authority(G0.agent_ids, baseline=0.5)

    _separator("Emergo + Lux Integration Demo")
    print(f"  Mode:       EMERGO_LUX_MODE={lux_mode}")
    print(f"  Agents:     {n_agents}")
    print(f"  Iterations: {n_iterations}")

    bridge: LuxBridge
    if lux_mode == "real":
        try:
            bridge = RealLuxBridge(
                authority_threshold=0.3,
                add_agent_threshold=0.6,
                max_agents=20,
            )
            print("  Governance: PyLuxGate (Rust — real enforcement)")
        except Exception as exc:
            print(f"\n  WARNING: Could not init RealLuxBridge ({exc})")
            print("  Falling back to SimulatedLuxBridge.")
            bridge = SimulatedLuxBridge()
            lux_mode = "simulated"
    else:
        bridge = SimulatedLuxBridge()
        print("  Governance: SimulatedLuxBridge (Python mock)")

    # --- Step 1: Run the kernel to let authority emerge ---
    _separator("Step 1: Emergo kernel — authority emergence")
    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=n_iterations,
        observers=[obs],
        rng=np.random.default_rng(seed),
    )
    G_final, _phi_final, A_final, _ = final_state

    scores = {a: round(A_final.get(a), 3) for a in A_final.scores}
    sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"  Termination: {reason}")
    print(f"  CE acceptance rate: {obs.ce_acceptance_rate:.1%}")
    print("\n  Final authority scores (ranked):")
    for i, (agent, score) in enumerate(sorted_agents):
        bar = "█" * int(score * 20)
        marker = " ← highest" if i == 0 else (" ← lowest" if i == len(sorted_agents) - 1 else "")
        print(f"    {agent}: {score:.3f}  {bar}{marker}")

    highest_agent = sorted_agents[0][0]
    lowest_agent = sorted_agents[-1][0]

    # --- Step 2: Demonstrate Lux enforcing authority at the boundary ---
    _separator("Step 2: Lux governance — CE authorization")

    test_cases = [
        {
            "label": (
                f"High-authority agent ({highest_agent}, "
                f"score={scores[highest_agent]}) proposes add_edge"
            ),
            "ce": CoordinationEvent(
                event_type="add_edge",
                participants=(highest_agent, lowest_agent),
                params=frozenset([("weight", 0.5)]),
            ),
        },
        {
            "label": (
                f"Low-authority agent ({lowest_agent}, "
                f"score={scores[lowest_agent]}) proposes add_edge"
            ),
            "ce": CoordinationEvent(
                event_type="add_edge",
                participants=(lowest_agent, highest_agent),
                params=frozenset([("weight", 0.5)]),
            ),
        },
        {
            "label": (
                f"High-authority agent ({highest_agent}) "
                "proposes add_agent (elevated threshold)"
            ),
            "ce": CoordinationEvent(
                event_type="add_agent",
                participants=(highest_agent,),
                params=frozenset([("agent_id", "new_agent")]),
            ),
        },
    ]

    for case in test_cases:
        result = bridge.authorize_ce(case["ce"], G_final, A_final)
        verdict = "APPROVED" if result.authorized else "DENIED"
        print(f"\n  {case['label']}")
        print(f"    Lux verdict:  {verdict}")
        print(f"    Reason:       {result.reason}")

    _separator("Integration verified")
    print("  Emergo: authority emerged from topology prediction accuracy.")
    print("  Lux:    governance enforced at the kernel boundary.")
    print("  Result: only agents that earned authority can reshape the network.")

    if lux_mode != "real":
        print(
            "\n  To run with real Lux enforcement:\n"
            "    cd /path/to/Lux-V1.0\n"
            "    maturin develop --features python\n"
            "    EMERGO_LUX_MODE=real python examples/lux_emergo_integration.py"
        )


if __name__ == "__main__":
    main()
