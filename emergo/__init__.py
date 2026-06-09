"""Emergo Kernel — self-referential graph dynamical system.

Public surface:

  Core types:
    Graph, CoordinationEvent, PhiMap, Authority, Errors, State
    Task, Goal

  Governance:
    Lux
    LuxBridge, SimulatedLuxBridge, AuthResult, make_lux_bridge

  Four atomic operations:
    ce_execute, error_computation, authority_update, phi_update

  Kernel (fixed-point loop):
    emergo_kernel, make_initial_phi, make_initial_authority

  Execution runtime:
    Executor, Planner, SequentialPlanner, DependencyPlanner, ExecutionResult, TaskOutcome
    mock_task_runner, failing_task_runner

  Multi-agent coordination:
    MultiAgentCoordinator, ProposedCE, CoordinationRound

  Observers (INV-10):
    KernelObserver, LoggingObserver, HistoryObserver, fire_observers

  Proposal generators:
    ProposalGenerator, DefaultProposalGenerator,
    SequenceProposalGenerator, WeightedMixGenerator

  Metrics / telemetry:
    MetricsObserver

  Diagnostics:
    KernelDiagnostics, IterationRecord, DetectorResult, run_health_check
    detect_authority_collapse, detect_topology_lock_in, detect_phi_gaming,
    detect_speculative_cascades, detect_lux_bottleneck, detect_clique_formation,
    detect_credit_assignment_ambiguity, detect_emergent_conservatism

  Persistence:
    save_state, load_state
"""

from emergo.authority_update import authority_update
from emergo.ce_execution import ce_execute
from emergo.coordinator import CoordinationRound, MultiAgentCoordinator, ProposedCE
from emergo.diagnostics import (
    DetectorResult,
    IterationRecord,
    KernelDiagnostics,
    detect_authority_collapse,
    detect_clique_formation,
    detect_credit_assignment_ambiguity,
    detect_emergent_conservatism,
    detect_lux_bottleneck,
    detect_phi_gaming,
    detect_speculative_cascades,
    detect_topology_lock_in,
    run_health_check,
)
from emergo.error_computation import error_computation
from emergo.executor import (
    ExecutionResult,
    Executor,
    TaskOutcome,
    failing_task_runner,
    mock_task_runner,
)
from emergo.kernel import emergo_kernel, make_initial_authority, make_initial_phi
from emergo.lux import Lux
from emergo.lux_bridge import AuthResult, LuxBridge, SimulatedLuxBridge, make_lux_bridge
from emergo.metrics import MetricsObserver
from emergo.observer import HistoryObserver, KernelObserver, LoggingObserver, fire_observers
from emergo.persistence import load_state, save_state
from emergo.phi_update import phi_update
from emergo.planner import DependencyPlanner, Planner, SequentialPlanner
from emergo.proposal import (
    DefaultProposalGenerator,
    ProposalGenerator,
    SequenceProposalGenerator,
    WeightedMixGenerator,
)
from emergo.types import (
    Authority,
    CoordinationEvent,
    Errors,
    Goal,
    Graph,
    PhiMap,
    State,
    Task,
)

__all__ = [
    "AuthResult",
    "Authority",
    "CoordinationEvent",
    "CoordinationRound",
    "DefaultProposalGenerator",
    "DependencyPlanner",
    "DetectorResult",
    "Errors",
    "ExecutionResult",
    # executor
    "Executor",
    "Goal",
    # types
    "Graph",
    "HistoryObserver",
    "IterationRecord",
    # diagnostics
    "KernelDiagnostics",
    # observer (INV-10)
    "KernelObserver",
    "LoggingObserver",
    # governance
    "Lux",
    "LuxBridge",
    # metrics / telemetry
    "MetricsObserver",
    # coordinator
    "MultiAgentCoordinator",
    "PhiMap",
    # planner
    "Planner",
    # proposal generators
    "ProposalGenerator",
    "ProposedCE",
    "SequenceProposalGenerator",
    "SequentialPlanner",
    "SimulatedLuxBridge",
    "State",
    "Task",
    "TaskOutcome",
    "WeightedMixGenerator",
    "authority_update",
    # four operations
    "ce_execute",
    "detect_authority_collapse",
    "detect_clique_formation",
    "detect_credit_assignment_ambiguity",
    "detect_emergent_conservatism",
    "detect_lux_bottleneck",
    "detect_phi_gaming",
    "detect_speculative_cascades",
    "detect_topology_lock_in",
    # kernel
    "emergo_kernel",
    "error_computation",
    "failing_task_runner",
    "fire_observers",
    "load_state",
    "make_initial_authority",
    "make_initial_phi",
    "make_lux_bridge",
    "mock_task_runner",
    "phi_update",
    "run_health_check",
    "save_state",
]
