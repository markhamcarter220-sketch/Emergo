Custom Proposal Generator and Lux Integration
===============================================

This tutorial shows how to write a custom ``ProposalGenerator``, blend it with
the built-in default using ``WeightedMixGenerator``, and integrate the Lux
authorization layer for capability-gated task execution.

The ProposalGenerator Protocol
--------------------------------

The kernel samples one coordination event (CE) per iteration by calling
``proposal_generator.propose(A_t, G_t, rng)``. Any object that implements
this single method can serve as the generator:

.. code-block:: python

    from typing import Optional
    import numpy as np
    from emergo.types import Authority, CoordinationEvent, Graph

    class MyGenerator:
        def propose(
            self,
            A_t: Authority,
            G_t: Graph,
            rng: np.random.Generator,
        ) -> Optional[CoordinationEvent]:
            ...  # return a CE or None to skip this iteration

The contract:

- **Read-only**: never mutate ``A_t`` or ``G_t``.
- **Deterministic given rng**: same inputs → same output (INV-4).
- **Return None for degenerate inputs**: e.g. fewer than 2 agents.

Writing a PriorityProposalGenerator
-------------------------------------

``PriorityProposalGenerator`` always selects the agent with the highest
current authority as the proposer. This is an *exploit* strategy — it
concentrates proposals on the proven best performer.

.. code-block:: python

    import numpy as np
    from typing import Optional
    from emergo.types import Authority, CoordinationEvent, Graph


    class PriorityProposalGenerator:
        """Always proposes from the highest-authority agent.

        This is a pure-exploit strategy: the agent that has been most accurate
        so far gets to make all proposals.  Combine with DefaultProposalGenerator
        via WeightedMixGenerator to balance exploration and exploitation.
        """

        def propose(
            self,
            A_t: Authority,
            G_t: Graph,
            rng: np.random.Generator,
        ) -> Optional[CoordinationEvent]:
            if G_t.n_agents < 2:
                return None

            agents = list(G_t.agent_ids)

            # Find the agent with the highest authority score
            top_agent = max(agents, key=lambda a: A_t.get(a))
            from_idx  = G_t.agent_index(top_agent)

            # Pick a random target (uniform — not authority-weighted)
            candidates = [i for i in range(G_t.n_agents) if i != from_idx]
            to_idx = int(rng.choice(candidates))
            to_agent = agents[to_idx]

            i = from_idx
            j = to_idx

            if G_t.adjacency[i, j] == 0.0:
                return CoordinationEvent(
                    event_type="add_edge",
                    participants=(top_agent, to_agent),
                    params=frozenset([("weight", float(A_t.get(top_agent)))]),
                )
            else:
                return CoordinationEvent(
                    event_type="remove_edge",
                    participants=(top_agent, to_agent),
                    params=frozenset(),
                )

Blending with WeightedMixGenerator
------------------------------------

Pure exploitation can cause **topology lock-in** — the top agent keeps
rewiring the same edges. ``WeightedMixGenerator`` lets you blend exploration
(``DefaultProposalGenerator``) with exploitation (``PriorityProposalGenerator``):

.. code-block:: python

    from emergo import DefaultProposalGenerator, WeightedMixGenerator

    default_gen  = DefaultProposalGenerator()
    priority_gen = PriorityProposalGenerator()

    # 70 % default (explore) + 30 % priority (exploit)
    mixed_gen = WeightedMixGenerator([
        (default_gen,  0.70),
        (priority_gen, 0.30),
    ])

The mixer samples one generator each iteration according to the given weights,
then falls back to the next one if the chosen generator returns ``None``.

Running the Kernel with a Custom Generator
-------------------------------------------

.. code-block:: python

    import numpy as np
    from emergo import (
        Graph,
        HistoryObserver,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
    )

    rng = np.random.default_rng(0)
    n   = 6
    ids = tuple(f"a{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.4:
                adj[i, j] = rng.uniform(0.3, 0.7)

    G0   = Graph(agent_ids=ids, adjacency=adj, capabilities=np.ones((n, 4)) * 0.5)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=0)
    A0   = make_initial_authority(ids, baseline=0.5)

    obs_default  = HistoryObserver()
    obs_priority = HistoryObserver()

    # Baseline: default generator
    state_d, _ = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=500,
        proposal_generator=DefaultProposalGenerator(),
        observers=[obs_default],
        rng=np.random.default_rng(1),
    )

    # Custom: mixed generator
    state_p, _ = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=500,
        proposal_generator=mixed_gen,
        observers=[obs_priority],
        rng=np.random.default_rng(1),
    )

    print(f"Default  acceptance rate: {obs_default.ce_acceptance_rate:.1%}")
    print(f"Priority acceptance rate: {obs_priority.ce_acceptance_rate:.1%}")

Verifying that the Top Agent Proposes More
-------------------------------------------

With the priority generator active, you can count CE proposals per agent
from the diagnostics:

.. code-block:: python

    _, _, diag = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=500,
        proposal_generator=mixed_gen,
        collect_diagnostics=True,
        rng=np.random.default_rng(2),
    )

    from collections import Counter
    proposer_counts = Counter(
        r.ce_proposer for r in diag.records
        if r.ce_accepted and r.ce_proposer is not None
    )
    print("\nProposals by agent (accepted CEs only):")
    for agent_id, count in proposer_counts.most_common():
        print(f"  {agent_id}: {count}")

The top-authority agent should appear disproportionately in the accepted CE
count when the priority generator is blended in.

Lux Integration: Granting Capabilities
----------------------------------------

Lux is the single authorization gate for all CE execution. Before an agent
can run a task that requires a specific capability, that capability must be
granted via ``lux.grant_capability``:

.. code-block:: python

    from emergo import Lux

    lux = Lux()
    lux.grant_capability("a0", "execute_task")
    lux.grant_capability("a1", "execute_task")

    print(f"a0 has execute_task: {lux.check_capability('a0', 'execute_task')}")
    print(f"a2 has execute_task: {lux.check_capability('a2', 'execute_task')}")

Capability-Gated Task Execution with the Executor
---------------------------------------------------

The ``Executor`` bridges goal-directed behaviour and the four atomic
Emergo operations. It enforces:

- **INV-5**: capability changes flow through ``update_capabilities`` CEs only.
- **INV-6**: resources are pre-deducted; refunded on failure.
- **INV-7**: every attempt is audited.
- **INV-8**: depth and pending-CE quotas are enforced.

.. code-block:: python

    from emergo import Executor, Goal, mock_task_runner

    lux = Lux()
    lux.grant_capability("a0", "analyse")

    executor = Executor(lux=lux, task_runner=mock_task_runner)

    goal = Goal(
        goal_id="demo_goal",
        description="Analyse the coordination graph",
        required_capability="analyse",
        initiating_agent="a0",
        resource_budget=5.0,
        max_depth=3,
    )

    result = executor.execute(goal, state_p)

    print(f"\nGoal: {result.goal_id}")
    print(f"  Success     : {result.success}")
    print(f"  Tasks tried : {result.tasks_attempted}")
    print(f"  Tasks ok    : {result.tasks_succeeded}")
    print(f"  Resources   : {result.resources_spent:.1f}")
    print(f"  Audit IDs   : {result.audit_ids[:3]} ...")

Combining Custom Proposals with Capability-Gated Execution
-----------------------------------------------------------

The full pipeline: custom proposals feed the kernel loop while the executor
runs capability-gated tasks after convergence:

.. code-block:: python

    import numpy as np
    from emergo import (
        DefaultProposalGenerator,
        Executor,
        Goal,
        Graph,
        HistoryObserver,
        Lux,
        WeightedMixGenerator,
        emergo_kernel,
        make_initial_authority,
        make_initial_phi,
        mock_task_runner,
    )

    # --- 1. Setup ---
    rng = np.random.default_rng(99)
    n   = 5
    ids = tuple(f"bot_{i}" for i in range(n))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.4:
                adj[i, j] = rng.uniform(0.3, 0.7)

    G0   = Graph(agent_ids=ids, adjacency=adj, capabilities=np.ones((n, 4)) * 0.5)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=99)
    A0   = make_initial_authority(ids, baseline=0.5)

    # --- 2. Grant capabilities ---
    lux = Lux()
    for agent_id in ids:
        lux.grant_capability(agent_id, "coordinate")

    # --- 3. Build custom generator ---
    priority_gen = PriorityProposalGenerator()
    mixed_gen    = WeightedMixGenerator([
        (DefaultProposalGenerator(), 0.7),
        (priority_gen,              0.3),
    ])

    # --- 4. Run kernel ---
    obs = HistoryObserver()
    final_state, reason = emergo_kernel(
        (G0, phi0, A0, []),
        max_iterations=500,
        proposal_generator=mixed_gen,
        observers=[obs],
        lux=lux,
    )

    _, _, A_final, _ = final_state
    top_agent = max(A_final.scores, key=A_final.scores.get)
    print(f"Kernel done: {reason}")
    print(f"Top agent  : {top_agent} (authority={A_final.get(top_agent):.3f})")

    # --- 5. Execute a goal using the top agent ---
    executor = Executor(lux=lux, task_runner=mock_task_runner)
    goal = Goal(
        goal_id="post_kernel_goal",
        description="Post-convergence coordination task",
        required_capability="coordinate",
        initiating_agent=top_agent,
        resource_budget=3.0,
    )

    result = executor.execute(goal, final_state)
    print(f"\nExecution: success={result.success}, "
          f"tasks={result.tasks_succeeded}/{result.tasks_attempted}")

Further Reading
---------------

- :doc:`first_swarm` — kernel basics and the HistoryObserver.
- ``emergo.coordinator.MultiAgentCoordinator`` — parallel CE proposal rounds.
- ``emergo.proposal.SequenceProposalGenerator`` — replay pre-built CE lists.
- ``emergo.planner.DependencyPlanner`` — decompose goals with task dependencies.
