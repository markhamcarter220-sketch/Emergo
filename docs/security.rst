Security Model & Best Practices
================================

Emergo's security model is built around three principles: **fail-closed authorization**,
**immutable audit trails**, and **per-agent resource governance**.  This page describes the
production security features available in v0.3.0 and best practices for deploying Emergo.

.. contents::
   :local:
   :depth: 2

----

Lux Authorization Layer
-----------------------

All CE execution — graph mutations, task delegation, capability updates — flows through a
single authorization gate: the ``Lux`` class backed by a ``LuxBridge`` implementation.

There are two authorization paths:

1. **Check-only** (``lux.authorize``): used by :func:`~emergo.ce_execution.ce_execute` for
   graph mutation CEs.  No resource is deducted.
2. **Full authorization** (``lux.authorize_full``): used by :class:`~emergo.executor.Executor`
   before task execution.  Pre-deducts the task resource cost from the agent's ledger;
   caller **must** call ``refund_resource()`` if the task subsequently fails.

Both paths are **fail-closed**: any internal error returns ``authorized=False`` rather than
raising an exception.  This guarantees Emergo never executes an unauthorized CE even under
partial failures.

Selecting a Bridge
~~~~~~~~~~~~~~~~~~

The bridge implementation is selected at construction time via the ``bridge=`` kwarg to
:class:`~emergo.lux.Lux`:

.. code-block:: python

   from emergo.lux import Lux
   from emergo.lux_bridge import SimulatedLuxBridge

   # Default: SimulatedLuxBridge (in-memory, no external deps)
   lux = Lux()

   # Explicit bridge (e.g. with custom initial_budget)
   bridge = SimulatedLuxBridge(initial_budget=500.0)
   lux = Lux(bridge=bridge)

   # Production: RealLuxBridge (requires 'lux' package + live service)
   from emergo.lux_bridge import RealLuxBridge
   lux = Lux(bridge=RealLuxBridge(max_retries=3))

   # Rate-limited bridge (recommended for production)
   from emergo.rate_limiting import RateLimitedLuxBridge, RateLimitConfig
   config = RateLimitConfig(max_proposals_per_second=20, burst_allowance=40,
                            max_in_flight_per_agent=10)
   lux = Lux(bridge=RateLimitedLuxBridge(SimulatedLuxBridge(), config=config))

The ``EMERGO_LUX_MODE`` environment variable (``simulated`` or ``real``) controls the
default bridge used when ``bridge=None`` is passed to ``Lux()``.

----

Rate Limiting and Blast-Radius Controls
-----------------------------------------

The :class:`~emergo.rate_limiting.RateLimitedLuxBridge` wraps any bridge implementation
and adds two complementary controls:

Token-Bucket Rate Limiter
~~~~~~~~~~~~~~~~~~~~~~~~~

Each agent gets its own **token bucket** that refills at ``max_proposals_per_second`` tokens
per second up to ``burst_allowance``.  When an agent's bucket is empty, ``authorize_ce``
returns ``authorized=False`` with a descriptive reason — the CE is cleanly rejected without
touching the inner bridge or the graph.

This prevents a single agent from flooding the kernel with proposals and monopolizing
compute resources.

Blast-Radius Guard
~~~~~~~~~~~~~~~~~~

``max_in_flight_per_agent`` limits the number of simultaneously **pending** CEs per agent
(accepted but not yet resolved by audit or refund).  Once the limit is reached, additional
proposals are denied until in-flight CEs are resolved.

Tracking in-flight CEs requires the caller to:

- Call ``authorize_ce`` (which increments the counter on success)
- Call ``audit(...)`` when a CE completes successfully *or* ``refund_resource(...)`` when it
  rolls back (both decrement the counter)

The :class:`~emergo.executor.Executor` already follows this protocol automatically.

.. code-block:: python

   from emergo.rate_limiting import RateLimitedLuxBridge, RateLimitConfig

   config = RateLimitConfig(
       max_proposals_per_second=10.0,    # token refill rate
       burst_allowance=20.0,             # burst capacity
       max_in_flight_per_agent=5,        # simultaneous pending CEs
   )
   bridge = RateLimitedLuxBridge(inner_bridge, config=config)

   # Inspect stats
   stats = bridge.rate_limit_stats()
   # {"rate_limit_denials": 3, "blast_radius_denials": 0, "in_flight": {"agent_0": 2}, ...}

   # Reset all buckets (e.g. after a kernel reset)
   bridge.reset_rate_limits()

   # Reset one agent's bucket only
   bridge.reset_rate_limits("agent_0")

----

Audit Trail
-----------

Every CE execution writes an immutable audit record via :meth:`~emergo.lux_bridge.LuxBridge.audit`.
Audit records include:

- ``audit_id``: UUID (opaque, globally unique)
- ``timestamp``: Unix epoch (float)
- ``ce_type``: e.g. ``"add_edge"``
- ``agent_ids``: all participating agents
- ``success``: whether the CE completed
- ``resource_deducted``: resource cost charged to the agent

To inspect the audit log (SimulatedLuxBridge):

.. code-block:: python

   from emergo.lux_bridge import SimulatedLuxBridge

   bridge = SimulatedLuxBridge()
   lux = Lux(bridge=bridge)

   # ... run some CEs ...

   log = bridge.get_audit_log()  # returns a deep copy (immutable)
   for record in log:
       print(record["ce_type"], record["success"], record["timestamp"])

.. warning::
   In production, the in-memory audit log grows without bound.  For long-running swarms,
   forward audit events to an external log store (e.g. write a ``KernelObserver`` that
   ships audit records to a database or SIEM).

----

Authority Invariants
--------------------

Emergo enforces hard authority bounds that cannot be bypassed through normal operation:

- **INV-11** (Monopolization cap): ``∀ agent, authority ≤ 0.8`` — hard-clipped in
  :func:`~emergo.authority_update.authority_update`.
- **INV-12** (No self-loops): ``add_edge(A, A)`` is rejected before any graph mutation.
- **INV-18** (Boundary preservation): authority clips to [0, 0.8] — never NaN or Inf.

These are not configurable.  Any attempt to bypass them (e.g. by directly calling
``Authority.set()`` with a value > 0.8) sets the value in the dict but it will be
overwritten on the next ``authority_update`` call.

See :doc:`/specification` and ``emergo/SAFETY_SPEC.md`` for the complete invariant proofs.

----

Capability Governance
---------------------

Capabilities are granted only via ``Lux.grant_capability()`` (or
``SimulatedLuxBridge.grant_capability()``).  No CE type can grant capabilities to
agents — this restriction is enforced by INV-5 (Proposal-Only Authority).

.. code-block:: python

   lux = Lux()
   # System initializer grants initial capabilities:
   lux.grant_capability("agent_0", "compute")
   lux.grant_capability("agent_1", "search")

   # Agents may NOT grant capabilities to themselves or others via CEs.
   # Any CE that tries to do so will be rejected (executor-only types).

----

Network Isolation
-----------------

Emergo is a library — it makes no outbound network calls unless you explicitly configure
OpenTelemetry export or connect a ``RealLuxBridge``.  The default ``SimulatedLuxBridge``
is entirely in-memory.

For OpenTelemetry export, see :class:`~emergo.metrics.MetricsObserver`.

----

Alerting
--------

The :class:`~emergo.alerting.AlertManager` implements the ``KernelObserver`` protocol
and fires configurable alerts when thresholds are exceeded during kernel execution:

.. code-block:: python

   from emergo.alerting import AlertManager, authority_monopoly_rule, convergence_stall_rule

   mgr = AlertManager(
       rules=[
           authority_monopoly_rule(threshold=0.75),   # warns before 0.8 hard cap
           convergence_stall_rule(stall_iterations=100),
       ],
       on_alert=lambda evt: print(f"ALERT: {evt.message}"),
   )

   final_state, reason = emergo_kernel(initial_state, observers=[mgr])

   # Review all alerts
   for alert in mgr.alert_history:
       print(f"[{alert.level.name}] t={alert.iteration}: {alert.message}")

Built-in alert rules:

- :func:`~emergo.alerting.authority_monopoly_rule` — fires when authority exceeds threshold
- :func:`~emergo.alerting.convergence_stall_rule` — fires when no CE accepted in N iterations
- :func:`~emergo.alerting.high_rejection_rate_rule` — fires when Lux rejection rate > threshold
- :func:`~emergo.alerting.phi_loss_spike_rule` — fires when φ loss spikes > N× baseline

----

Structured Logging
------------------

For log aggregation systems (Datadog, Loki, Splunk), use structured JSON logging:

.. code-block:: python

   from emergo.logging_config import configure_structured_logging

   configure_structured_logging(level="INFO")
   # All emergo.* loggers now emit JSON lines to stderr:
   # {"timestamp": "2026-06-09T12:34:56.789Z", "level": "INFO",
   #  "logger": "emergo.kernel", "message": "..."}

   # Or with a per-run context:
   from emergo.logging_config import get_emergo_logger
   log = get_emergo_logger("emergo.myapp", run_id="exp_001", n_agents=50)
   log.info("Kernel started")

----

Checklist: Production Deployment
----------------------------------

Before deploying Emergo in a production environment, verify:

.. raw:: html

   <ul>
   <li>☑ <strong>Use RateLimitedLuxBridge</strong> — prevents CE flooding and blast-radius attacks</li>
   <li>☑ <strong>Set appropriate min_authority</strong> — default 0.1 is conservative; lower for high-autonomy agents</li>
   <li>☑ <strong>Enable structured logging</strong> — ship logs to an aggregation system</li>
   <li>☑ <strong>Wire AlertManager</strong> — catch authority anomalies before they cause issues</li>
   <li>☑ <strong>Use SqliteStore checkpointing</strong> — resume long runs after failures</li>
   <li>☑ <strong>Grant minimal capabilities</strong> — principle of least privilege</li>
   <li>☑ <strong>Monitor with MetricsObserver</strong> — expose Prometheus metrics for your SRE team</li>
   <li>☑ <strong>Review audit log periodically</strong> — forward to a SIEM for compliance</li>
   </ul>
