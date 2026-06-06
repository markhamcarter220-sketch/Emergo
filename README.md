# Emergo: A Self-Organizing Intelligence Engine

## One-Sentence Summary
A system where agents earn authority by accurately predicting how their actions reshape the coordination network, causing intelligence to emerge from the structure itself.

---

## The Core Idea (Metaphor: A Jazz Ensemble)

Imagine a jazz ensemble where:

**Standard jazz**: The bandleader tells everyone what to play. Clear hierarchy, but rigid.

**Emergo's jazz ensemble**:
- No bandleader
- Each musician has to predict what the *ensemble will sound like* if they play a certain note
- Musicians who predict well get to initiate more of the band's direction
- Bad predictions lose influence
- Over time, the musicians that understand how the whole ensemble works together become the de-facto leaders
- But it's not a hierarchy — it's emergent structure

That's Emergo.

---

## What It Actually Does (Three Layers)

### Layer 1: The Interaction
An agent proposes a **Coordination Event**: "Let's work together on X."

Other agents join. They execute. Something new gets created (a capability, a solution, a structure).

### Layer 2: The Learning Signal
After execution, the system asks: *Did the topology of the network change the way you predicted?*

- If you predicted well → your authority grows
- If you predicted poorly → your authority shrinks
- Authority determines how much future coordination you can initiate

### Layer 3: The Emergence
Over thousands of interactions:
- Agents that understand coordination dynamics gain influence
- Agents with influence shape what coordination events can form
- The network self-organizes around agents that predict well
- Roles (planner, verifier, connector) emerge naturally — not assigned
- Intelligence accumulates in the structure, not in individual agents

---

## Why This Matters

Most multi-agent systems are either:

1. **Monolithic** (one big AI deciding everything)
   - Problem: Single point of failure, no distributed learning

2. **Hierarchical** (boss → managers → workers)
   - Problem: Rigid. Can't adapt. Authority doesn't reflect actual competence

3. **Democratic** (all agents equal)
   - Problem: No incentive for specialization. Coordination is chaotic

**Emergo's approach**: Authority tracks *structural foresight*.

You didn't earn authority because you're good at verifying (that's a skill).
You earned authority because you **understand how verification agents fit into the network** and can predict what happens when you connect them.

That's fundamentally different. It optimizes for system-level understanding, not individual performance.

---

## The Mechanism (Metaphor: An Ecosystem)

Think of it like:

- **Agents** = organisms
- **Capabilities** = resources (food, skills, tools)
- **Coordination Events** = interactions (hunting together, building shelter, trading)
- **Authority** = reproductive fitness (organisms with good fitness reproduce more, shaping the ecosystem)
- **Topology prediction** = understanding the ecosystem (which animals should team up? what happens if predators leave? who fills the niche?)

An organism doesn't earn high fitness by being the strongest predator.
It earns high fitness by **understanding which alliances work, which resources matter, how the ecosystem reorganizes**.

Over time, the ecosystem self-organizes around organisms that model it well.

That's Emergo.

---

## What Makes It Different

| System | Authority Based On | Result |
|---|---|---|
| Hierarchy | Job title | Brittle, static |
| Democracy | Voting | Noisy, slow |
| Prediction markets | Accuracy on tasks | Specializes but doesn't integrate |
| **Emergo** | **Topology prediction accuracy** | **Self-organizing, adaptive, collective** |

---

## The Technical Core (For Builders)

Emergo is a dynamical system:

```
State = (Graph, LatentSpace, Authority, ErrorHistory)

Each iteration:
  1. Execute a Coordination Event
  2. Observe how the graph changed
  3. Measure prediction error
  4. Update agents' authority based on accuracy
  5. Learn a better model of graph dynamics
  Repeat
```

The system converges when agents' predictions match reality consistently — meaning they've learned the underlying structure of coordination.

---

## What You Get

- **Self-organizing intelligence**: No central design needed; structure emerges
- **Distributed learning**: Every agent learns the dynamics; intelligence is network-wide
- **Adaptive roles**: Roles form based on demonstrated structural understanding
- **Resilience**: If one agent fails, others can step in (they understand the network)
- **Interpretability**: Authority maps to actual system understanding; you can see who models what

---

## What This Is Not

- **Not a marketplace** (no currency, no prices)
- **Not a voting system** (no polls, no majorities)
- **Not a hierarchy** (no org chart)
- **Not a random swarm** (structure emerges, not chaos)

It's closer to how natural systems organize: ant colonies, neural networks, ecosystems.

---

## The Long-Term Vision

Right now, AI is either:

**Monolithic**: One model does everything.
**Fragmented**: Many models, someone stitches them together.

Emergo aims for:

**Self-organizing**: Many agents learn to coordinate intelligently without central direction.

The vision: *A system where intelligence emerges from the competence of agents to understand and predict how coordination structures evolve.*

---

## Getting Started

1. Read `SPECIFICATION.md` (formal system definition)
2. Read `CLAUDE_CODE_PROMPT.md` (what the kernel does)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the kernel:

```python
from emergo import Graph, Lux, emergo_kernel, make_initial_phi, make_initial_authority
import numpy as np

# Define initial agents and connections
adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
caps = np.ones((3, 2)) * 0.5
G0 = Graph(agent_ids=("A", "B", "C"), adjacency=adj, capabilities=caps)

phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4)
A0 = make_initial_authority(G0.agent_ids)

final_state, reason = emergo_kernel(
    initial_state=(G0, phi0, A0, []),
    max_iterations=1000,
    convergence_threshold=1e-4,
)
print(reason)  # "Converged" or "Max iterations reached"
```

5. Watch the system self-organize

---

## Core References

- **SPECIFICATION.md**: The formal axiomatic system (what must be true)
- **emergo/kernel.py**: The fixed-point loop orchestrating the four operations
- **emergo/ce_execution.py**: Operation 1 — atomic graph transformation
- **emergo/error_computation.py**: Operation 2 — per-agent prediction error
- **emergo/authority_update.py**: Operation 3 — correctness-weighted authority feedback
- **emergo/phi_update.py**: Operation 4 — joint latent map optimization
- **tests/test_invariants.py**: Proof that the four system invariants hold in code
