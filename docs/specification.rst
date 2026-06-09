Specification & Invariants
==========================

See the formal specification in :doc:`SPECIFICATION` and the Tier 1 Safety Contract
in ``emergo/SAFETY_SPEC.md``.

System Invariants
-----------------

Ten core invariants (INV-1 through INV-10) plus eight safety red-lines (INV-11 through
INV-18) are formally specified and tested.

INV-1 through INV-10 are the operational invariants enforced by the kernel loop.
INV-11 through INV-18 are the Tier 1 safety red-lines defined in ``SAFETY_SPEC.md``.

.. list-table:: Core Invariants
   :header-rows: 1
   :widths: 10 90

   * - ID
     - Description
   * - INV-1
     - State Ownership: ``(G, φ, A, E)`` is the only mutable state; all ops are pure functions
   * - INV-2
     - Feedback Loop: ``error → authority → topology`` is closed and observable
   * - INV-3
     - Blast Radius: failures degrade gracefully; regularization prevents φ collapse
   * - INV-4
     - Timing: sequential, atomic, deterministic; no race conditions
   * - INV-5
     - Proposal-Only Authority: Emergo never mints capabilities; only Lux grants them
   * - INV-6
     - Resource Conservation: every task pre-charges Lux ledger; failures trigger refund
   * - INV-7
     - Observable + Fail-Closed: every CE attempt is audited
   * - INV-8
     - Bounded Speculation: depth limit + per-agent pending CE quota enforced
   * - INV-9
     - Coordinator Serialization: parallel proposals sorted by authority
   * - INV-10
     - Observer Isolation: observer exceptions never propagate to the kernel
