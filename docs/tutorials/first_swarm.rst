Build Your First Emergo Swarm
=============================

This tutorial walks through building an 8-agent mesh network from scratch,
running the Emergo kernel loop, and interpreting the results. By the end you
will understand how agents earn (or lose) authority and how to monitor the
system with observers and health checks.

Concepts
--------

In Emergo, a **swarm** is the combination of four components that travel
together as the kernel state:

``G`` (Graph)
    A directed weighted graph. Nodes are agents; edges encode coordination
    relationships. The adjacency matrix ``G.adjacency[i, j]`` stores the
    weight of the directed edge from agent ``i`` to agent ``j``.

``φ`` (PhiMap)
    The latent map that embeds a graph into a low-dimensional space and
    predicts how a coordination event (CE) transforms that embedding.
    Agents that predict well earn authority.

``A`` (Authority)
    Per-agent scalar scores in ``[0, 1]``. High authority means the agent's
    CE proposals are more likely to be accepted by Lux and to be sampled by
    the default proposal generator.

``E`` (Error history)
    A list of ``Errors`` objects — one per accepted CE — recording each
    agent's L2 prediction error at that step. The kernel uses this history
    to drive both authority updates and ``φ`` learning.

Step 1: Build an 8-Agent Mesh Graph
------------------------------------

A **mesh** graph connects every agent to a random sparse subset of others.
We target 30 % edge density.

.. code-block:: python

    import numpy as np
    from emergo import Graph, make_initial_phi, make_initial_authority

    rng = np.random.default_rng(42)
    n = 8
    ids = tuple(f"agent_{i}" for i in range(n))

    # Random sparse adjacency — ~30 % edge density
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.3:
                adj[i, j] = rng.uniform(0.2, 0.8)

    # 4-dimensional capability vectors per agent
    caps = rng.uniform(0.3, 0.7, (n, 4))

    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    print(f"Agents: {n}")
    print(f"Edges:  {int((adj > 0).sum())}")

The ``Graph`` dataclass is **immutable** — the kernel never mutates it in
place; every accepted CE produces a brand-new ``Graph`` object.

Step 2: Initialize ``φ`` and Authority
---------------------------------------

.. code-block:: python

    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=42)
    A0   = make_initial_authority(ids, baseline=0.5)

    print(f"All agents start at authority {A0.baseline}")

``make_initial_phi`` creates small random weights (near-zero). The latent
dimension ``d_latent=8`` controls the expressiveness of the embedding.
``d_features=16`` must match the output of ``extract_graph_features``.

Step 3: Attach a HistoryObserver
---------------------------------

``HistoryObserver`` accumulates per-iteration metrics without touching kernel
state (INV-10). It is the easiest way to understand what happened after a run.

.. code-block:: python

    from emergo import HistoryObserver

    obs = HistoryObserver()

Step 4: Run the Kernel for 1000 Iterations
-------------------------------------------

.. code-block:: python

    from emergo import emergo_kernel

    initial_state = (G0, phi0, A0, [])

    final_state, reason = emergo_kernel(
        initial_state=initial_state,
        max_iterations=1000,
        observers=[obs],
    )

    G_final, phi_final, A_final, E_history = final_state
    print(f"Termination reason : {reason}")
    print(f"Accepted CEs       : {sum(1 for _, ok in obs._ce_history if ok)}")
    print(f"CE acceptance rate : {obs.ce_acceptance_rate:.1%}")

The kernel terminates either when the mean ``φ``-prediction error has
plateaued (``reason == "Converged"``) or when ``max_iterations`` is
reached.

Step 5: Interpret the Output
------------------------------

Authority scores shift as agents make and miss predictions. Agents that
correctly predict how their coordination actions reshape the graph accumulate
high authority; poor predictors drift toward zero.

.. code-block:: python

    print("\nFinal authority scores (sorted):")
    ranked = sorted(A_final.scores.items(), key=lambda kv: kv[1], reverse=True)
    for agent_id, score in ranked:
        bar = "█" * int(score * 20)
        print(f"  {agent_id:10s}  {score:.3f}  {bar}")

    # Error trajectory
    if obs.mean_errors:
        first_err = obs.mean_errors[0]
        last_err  = obs.mean_errors[-1]
        print(f"\nPhi error: {first_err:.4f} → {last_err:.4f}")

**Why did the top agent win?** In the default setup, the proposal generator
uses softmax-weighted sampling over authority scores — high-authority agents
propose more CEs, and if their proposals are accurate they earn even more
authority. This creates a positive feedback loop that the error signal
eventually stabilises through ``φ``-update regularisation.

Step 6: Run Health Checks
--------------------------

The diagnostics module exposes eight failure-mode detectors. Run them on a
``KernelDiagnostics`` object (set ``collect_diagnostics=True``):

.. code-block:: python

    from emergo import run_health_check

    final_state2, reason2, diag = emergo_kernel(
        initial_state,
        max_iterations=1000,
        collect_diagnostics=True,
    )

    results = run_health_check(diag, final_state2)
    print(f"\nHealth check: {len(results)} detectors")
    for r in results:
        status = "OK" if not r.triggered else "WARNING"
        print(f"  [{status:7s}] {r.detector_name}: {r.message}")

Triggered detectors are hints — not hard failures. Common ones on small
graphs: ``authority_collapse`` (all scores converge to the same value) and
``topology_lock_in`` (edge count stops changing).

Step 7: Visualize (optional — requires ``emergo[viz]``)
---------------------------------------------------------

Install visualization extras::

    pip install "emergo[viz]"

Then:

.. code-block:: python

    import sys

    try:
        from emergo.visualize import render_health_dashboard
        render_health_dashboard(diag, final_state2, output_dir="./plots")
        print("Plots written to ./plots/")
    except Exception as e:
        print(f"Viz skipped: {e}")

``render_health_dashboard`` writes several PNG files:

- ``authority_history.png`` — per-agent authority over time
- ``topology_entropy.png`` — graph entropy (structural complexity)
- ``phi_loss.png`` — ``φ``-update loss curve

Putting It All Together
------------------------

.. code-block:: python

    import numpy as np
    from emergo import (
        Graph,
        HistoryObserver,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
        run_health_check,
    )

    rng = np.random.default_rng(42)
    n = 8
    ids = tuple(f"agent_{i}" for i in range(n))

    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.3:
                adj[i, j] = rng.uniform(0.2, 0.8)

    caps = rng.uniform(0.3, 0.7, (n, 4))
    G0   = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=42)
    A0   = make_initial_authority(ids, baseline=0.5)

    obs = HistoryObserver()
    final_state, reason, diag = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=1000,
        observers=[obs],
        collect_diagnostics=True,
    )

    _, _, A_final, _ = final_state
    print(f"Termination : {reason}")
    print(f"Acceptance  : {obs.ce_acceptance_rate:.1%}")

    ranked = sorted(A_final.scores.items(), key=lambda kv: kv[1], reverse=True)
    for aid, score in ranked:
        print(f"  {aid:10s}  {score:.3f}")

    results = run_health_check(diag, final_state)
    triggered = [r for r in results if r.triggered]
    print(f"\nHealth: {len(triggered)}/{len(results)} detectors triggered")

Next Steps
----------

- :doc:`custom_proposal_lux` — control which agents propose CEs and
  integrate capability-gated task execution.
- Explore ``emergo.coordinator.MultiAgentCoordinator`` for running parallel
  proposal rounds.
- Try ``emergo run --agents 12 --optimizer adam`` from the CLI.
