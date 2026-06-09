Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install emergo                       # core (numpy only)
   pip install "emergo[viz]"               # + matplotlib / networkx
   pip install "emergo[dev,viz]"           # everything for development

From source::

   git clone https://github.com/markhamcarter220-sketch/Emergo.git
   cd Emergo
   pip install -e ".[dev,viz]"
   emergo demo convergence   # verify installation

Your First Kernel Run
---------------------

.. code-block:: python

   import numpy as np
   from emergo import (
       Graph, emergo_kernel,
       make_initial_phi, make_initial_authority,
       HistoryObserver,
   )

   # Build a 5-agent ring graph
   n = 5
   ids = tuple(f"agent_{i}" for i in range(n))
   adj = np.zeros((n, n))
   for i in range(n):
       adj[i, (i + 1) % n] = 0.5

   G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=np.ones((n, 4)) * 0.5)
   phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4)
   A0 = make_initial_authority(ids, baseline=0.5)

   obs = HistoryObserver()
   final_state, reason = emergo_kernel(
       initial_state=(G0, phi0, A0, []),
       max_iterations=500,
       observers=[obs],
   )

   _, _, A_final, _ = final_state
   print(f"Termination: {reason}")
   print(f"CE acceptance: {obs.ce_acceptance_rate:.1%}")

Using the CLI
-------------

.. code-block:: bash

   emergo run                        # 8 agents, 2000 iterations
   emergo run --agents 12 --optimizer adam
   emergo health                     # 8 failure detector health check
   emergo demo convergence           # convergence showcase
   emergo demo stress --agents 20    # large graph stress demo
