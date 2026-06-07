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
"""

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
from emergo.lux_bridge import AuthResult, LuxBridge, SimulatedLuxBridge, make_lux_bridge
from emergo.lux import Lux
from emergo.ce_execution import ce_execute
from emergo.error_computation import error_computation
from emergo.authority_update import authority_update
from emergo.phi_update import phi_update
from emergo.kernel import emergo_kernel, make_initial_phi, make_initial_authority
from emergo.executor import (
    Executor,
    ExecutionResult,
    TaskOutcome,
    mock_task_runner,
    failing_task_runner,
)
from emergo.planner import Planner, SequentialPlanner, DependencyPlanner
from emergo.coordinator import MultiAgentCoordinator, ProposedCE, CoordinationRound
from emergo.observer import KernelObserver, LoggingObserver, HistoryObserver, fire_observers
from emergo.proposal import (
    ProposalGenerator,
    DefaultProposalGenerator,
    SequenceProposalGenerator,
    WeightedMixGenerator,
)
from emergo.metrics import MetricsObserver
from emergo.diagnostics import (
    KernelDiagnostics,
    IterationRecord,
    DetectorResult,
    run_health_check,
    detect_authority_collapse,
    detect_topology_lock_in,
    detect_phi_gaming,
    detect_speculative_cascades,
    detect_lux_bottleneck,
    detect_clique_formation,
    detect_credit_assignment_ambiguity,
    detect_emergent_conservatism,
)

__all__ = [
    # types
    "Graph", "CoordinationEvent", "PhiMap", "Authority", "Errors", "State",
    "Task", "Goal",
    # governance
    "Lux", "LuxBridge", "SimulatedLuxBridge", "AuthResult", "make_lux_bridge",
    # four operations
    "ce_execute", "error_computation", "authority_update", "phi_update",
    # kernel
    "emergo_kernel", "make_initial_phi", "make_initial_authority",
    # executor
    "Executor", "ExecutionResult", "TaskOutcome",
    "mock_task_runner", "failing_task_runner",
    # planner
    "Planner", "SequentialPlanner", "DependencyPlanner",
    # coordinator
    "MultiAgentCoordinator", "ProposedCE", "CoordinationRound",
    # observer (INV-10)
    "KernelObserver", "LoggingObserver", "HistoryObserver", "fire_observers",
    # proposal generators
    "ProposalGenerator", "DefaultProposalGenerator",
    "SequenceProposalGenerator", "WeightedMixGenerator",
    # metrics / telemetry
    "MetricsObserver",
    # diagnostics
    "KernelDiagnostics", "IterationRecord", "DetectorResult", "run_health_check",
    "detect_authority_collapse", "detect_topology_lock_in", "detect_phi_gaming",
    "detect_speculative_cascades", "detect_lux_bottleneck", "detect_clique_formation",
    "detect_credit_assignment_ambiguity", "detect_emergent_conservatism",
]
