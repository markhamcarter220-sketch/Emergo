"""Emergo Kernel — self-referential graph dynamical system.

Public surface:

  Core types:
    Graph, CoordinationEvent, PhiMap, Authority, Errors, State
    Task, Goal

  Governance:
    Lux
    LuxBridge, SimulatedLuxBridge, AuthResult, make_lux_bridge
    RateLimitedLuxBridge, RateLimitConfig

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
    CheckpointKernelObserver, CheckpointMeta, StateStore, SqliteStore, PickleStore, RunInfo

  Sparse graph utilities:
    is_sparse_beneficial, to_sparse_adjacency, from_sparse_adjacency,
    sparse_graph_features, estimate_memory_bytes

  Distributed execution:
    run_parallel_kernels, KernelConfig, KernelResult
    best_converged, all_converged, summarize_results

  Alerting:
    AlertManager, AlertLevel, AlertEvent
    authority_monopoly_rule, convergence_stall_rule
    high_rejection_rate_rule, phi_loss_spike_rule

  Structured logging:
    configure_structured_logging, get_emergo_logger, JsonFormatter
"""

from emergo.alerting import (
    AlertEvent,
    AlertLevel,
    AlertManager,
    AlertRule,
    authority_monopoly_rule,
    convergence_stall_rule,
    high_rejection_rate_rule,
    phi_loss_spike_rule,
)
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
from emergo.distributed import (
    KernelConfig,
    KernelResult,
    all_converged,
    best_converged,
    run_parallel_kernels,
    summarize_results,
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
from emergo.logging_config import JsonFormatter, configure_structured_logging, get_emergo_logger
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
from emergo.rate_limiting import RateLimitConfig, RateLimitedLuxBridge
from emergo.serde import load_checkpoint, save_checkpoint
from emergo.sparse import (
    estimate_memory_bytes,
    from_sparse_adjacency,
    is_sparse_beneficial,
    sparse_graph_features,
    to_sparse_adjacency,
)
from emergo.store import (
    CheckpointKernelObserver,
    CheckpointMeta,
    PickleStore,
    RunInfo,
    SqliteStore,
    StateStore,
    make_store,
)
from emergo.types import (
    Authority,
    CoordinationEvent,
    EligibilityTraces,
    Errors,
    ErrorScales,
    Goal,
    Graph,
    PhiMap,
    State,
    Task,
)

__all__ = [  # noqa: RUF022
    # alerting
    "AlertEvent",
    "AlertLevel",
    "AlertManager",
    "AlertRule",
    "AuthResult",
    "Authority",
    "CoordinationEvent",
    "CoordinationRound",
    "DefaultProposalGenerator",
    "DependencyPlanner",
    "DetectorResult",
    "EligibilityTraces",
    "ErrorScales",
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
    # distributed
    "KernelConfig",
    "KernelResult",
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
    # rate limiting
    "RateLimitConfig",
    "RateLimitedLuxBridge",
    "SequenceProposalGenerator",
    "SequentialPlanner",
    "SimulatedLuxBridge",
    "State",
    "Task",
    "TaskOutcome",
    "WeightedMixGenerator",
    # alerting rules
    "all_converged",
    "authority_monopoly_rule",
    "authority_update",
    "best_converged",
    # four operations
    "ce_execute",
    # structured logging
    "configure_structured_logging",
    "convergence_stall_rule",
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
    "get_emergo_logger",
    "high_rejection_rate_rule",
    "JsonFormatter",
    "load_checkpoint",
    "load_state",
    "make_initial_authority",
    "make_initial_phi",
    "make_lux_bridge",
    "mock_task_runner",
    "phi_loss_spike_rule",
    "phi_update",
    "run_health_check",
    "run_parallel_kernels",
    "save_checkpoint",
    "save_state",
    # store / persistence
    "CheckpointKernelObserver",
    "CheckpointMeta",
    "PickleStore",
    "RunInfo",
    "SqliteStore",
    "StateStore",
    "make_store",
    "summarize_results",
    # sparse graph support
    "estimate_memory_bytes",
    "from_sparse_adjacency",
    "is_sparse_beneficial",
    "sparse_graph_features",
    "to_sparse_adjacency",
]
