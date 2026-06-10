"""Runtime configuration and feature flags.

All values can be overridden via environment variables.
No logic lives here — only constants and env reads.
"""

from __future__ import annotations

import os

# --- Lux integration mode ---
LUX_MODE: str = os.getenv("EMERGO_LUX_MODE", "simulated")
"""
"simulated" (default): use SimulatedLuxBridge (in-memory, no external deps).
"real":                 use RealLuxBridge wrapping actual Lux Python bindings.
"""

# --- INV-8: Bounded Speculation ---
MAX_DECOMPOSITION_DEPTH: int = int(os.getenv("EMERGO_MAX_DEPTH", "5"))
"""Maximum recursive task decomposition depth before a CE is rejected."""

MAX_PENDING_CES_PER_AGENT: int = int(os.getenv("EMERGO_MAX_PENDING", "10"))
"""Maximum unexecuted CE proposals an agent may hold simultaneously."""

# --- Learning ---
DEFAULT_ETA: float = float(os.getenv("EMERGO_ETA", "0.05"))
"""Authority update learning rate η."""

PHI_UPDATE_INTERVAL: int = int(os.getenv("EMERGO_PHI_INTERVAL", "10"))
"""Run PhiUpdate every N successful CE executions."""

# --- Resources ---
DEFAULT_INITIAL_BUDGET: float = float(os.getenv("EMERGO_INITIAL_BUDGET", "100.0"))
"""Starting resource balance per agent in SimulatedLuxBridge."""

DEFAULT_TASK_COST: float = float(os.getenv("EMERGO_TASK_COST", "1.0"))
"""Default resource cost charged per task execution CE."""

# --- PhiUpdate optimizer ---
PHI_OPTIMIZER: str = os.getenv("EMERGO_PHI_OPTIMIZER", "sgd")
"""PhiUpdate optimizer: "sgd" (default) or "adam"."""

PHI_LEARNING_RATE: float = float(os.getenv("EMERGO_PHI_LR", "0.001"))
"""Learning rate for phi_update gradient steps."""

PHI_GRAD_CLIP: float = float(os.getenv("EMERGO_PHI_GRAD_CLIP", "1.0"))
"""Gradient clipping magnitude for phi_update."""

PHI_EARLY_STOP_PATIENCE: int = int(os.getenv("EMERGO_PHI_EARLY_STOP", "5"))
"""Early-stopping patience steps for phi_update (0 = disabled)."""

RANK_PENALTY_LAMBDA: float = float(os.getenv("EMERGO_RANK_PENALTY", "0.1"))
"""Rank-regularization strength for phi_update nuclear-norm penalty.
   0.0 = disabled.  Default 0.1.  See phi_update._rank_penalty_and_grad().
"""

# --- Multi-Agent Coordinator ---
MAX_COORDINATOR_AGENTS: int = int(os.getenv("EMERGO_MAX_COORDINATOR_AGENTS", "4"))
"""Maximum number of agents that may propose simultaneously in one coordination round (INV-9)."""

COORDINATOR_CONFLICT_STRATEGY: str = os.getenv("EMERGO_CONFLICT_STRATEGY", "priority")
"""Conflict resolution strategy for MultiAgentCoordinator.
   'priority': proposals sorted by authority; first on an edge wins.
"""

# --- R5: add_agent authority gate ---
ADD_AGENT_AUTHORITY_THRESHOLD: float = float(os.getenv("EMERGO_ADD_AGENT_THRESHOLD", "0.6"))
"""Minimum proposer authority required to authorize an add_agent CE."""

# --- R1: phi sliding window ---
PHI_WINDOW: int = int(os.getenv("EMERGO_PHI_WINDOW", "200"))
"""Maximum number of recent transitions used by phi_update. 0 = no limit."""

# --- R3: rank regularizer hinge ---
RANK_THRESHOLD: float = float(os.getenv("EMERGO_RANK_THRESHOLD", "0.1"))
"""sigma_min threshold below which the rank regularizer fires. 0.0 = always fire."""
