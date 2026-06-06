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

# --- Multi-Agent Coordinator ---
MAX_COORDINATOR_AGENTS: int = int(os.getenv("EMERGO_MAX_COORDINATOR_AGENTS", "4"))
"""Maximum number of agents that may propose simultaneously in one coordination round (INV-9)."""

COORDINATOR_CONFLICT_STRATEGY: str = os.getenv("EMERGO_CONFLICT_STRATEGY", "priority")
"""Conflict resolution strategy for MultiAgentCoordinator.
   'priority': proposals sorted by authority; first on an edge wins.
"""
