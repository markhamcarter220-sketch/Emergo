# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-09

### Added
- **Typer + Rich CLI** (`emergo run`, `emergo demo`, `emergo viz`, `emergo health`) with
  env-var config layering (CLI > env > defaults), Rich progress bars, and coloured tables.
- **`emergo demo`** subcommands: `convergence`, `multi-agent`, `executor`, `stress`.
- **`emergo viz`** command: generate matplotlib plots from saved diagnostics or fresh run.
- **`emergo health`** command: run all 8 failure detectors and print coloured Rich table.
- **Tier 1 Formal Safety Contract** (`emergo/SAFETY_SPEC.md`): Φ_safe manifold definition,
  authority conservation law, 8 red-line invariants (INV-11 through INV-18), gap analysis.
- **Lean 4 theorem statements** (`lean/EmergoConvergence.lean`): 4 main theorems, 4
  supporting lemmas, 2 corollaries targeting Mathlib4 v4.14.0.
- **9 adversarial tests** (`tests/test_adversarial_tier1.py`) covering all 8 red-line
  invariants plus a composite regression guard.
- **Long-horizon benchmark harness** (`examples/tier1_validation.py`): tracks INV-11 through
  INV-18 across up to 1M iterations in 5k-iteration batches with 4 diagnostic plots and a
  structured `[TIER1_VALIDATION]` JSON report; `--quick` flag for 2k-iteration smoke test.
- **Differentiated per-agent error computation**: proposer receives global φ-prediction error;
  non-proposing participants receive local structural-change error — breaking the root cause
  of authority collapse.
- **`DependencyPlanner`**: DAG-based task sequencer with cycle detection.
- **`WeightedMixGenerator`**: probabilistic mixture of `ProposalGenerator` instances.
- **`MetricsObserver`**: Prometheus text-format + optional OpenTelemetry OTLP export.
- **Rank regularization** for φ: nuclear-norm gradient keeps `rank(W_phi) ≥ d_latent // 2`.
- **Adam optimizer** path in `phi_update` alongside SGD.
- **Dynamic graph feature extraction**: fixed-dim spectral + structural features for any `n_agents`.
- **`emergo/__main__.py`**: enables `python -m emergo` invocation.

### Changed
- `error_computation` redesigned: proposer gets global φ-prediction error, participants get
  local adjacency-delta error; breaks the authority-collapse loop that affected multi-participant CEs.
- `PhiMap.embed` and `PhiMap.transition` promoted to public API.
- Authority scores clipped to `[0.0, 1.0]` (hard bounds enforced in `authority_update`).
- `emergo_kernel` `observers` parameter renamed from positional to keyword-only.
- `DefaultProposalGenerator` extracted from `kernel.py` into `proposal.py`.
- `pyproject.toml` updated: hatchling build backend, PEP 621 metadata, optional `[viz]` / `[otel]` / `[dev]` groups.

### Fixed
- Authority collapse when multiple agents participate in a single CE (all received identical
  `correctness = 0` because `max_error = error`).
- `phi_update` accumulating unbounded `g_history` — kernel now passes fresh `E_history=[]`
  each batch in long-horizon runs.
- Feature extractor returning `NaN`/`Inf` on graphs with zero-degree nodes.

### Security / Governance
- INV-11 through INV-18 formally specified and tested (see `emergo/SAFETY_SPEC.md`).
- Lux fail-closed governance enforced on every CE execution path.

## [0.1.0] - 2025-12-01

### Added
- Initial release of the Emergo kernel.
- Core state machine: `Graph`, `PhiMap`, `Authority`, `Errors` types.
- Four atomic operations: `ce_execute`, `error_computation`, `authority_update`, `phi_update`.
- `emergo_kernel` fixed-point loop with convergence detection.
- `Lux` governance layer with capability gating and resource ledger.
- `Executor`, `SequentialPlanner` execution runtime.
- `MultiAgentCoordinator` with authority-serialized conflict resolution (INV-9).
- `KernelObserver` protocol with `LoggingObserver` and `HistoryObserver` implementations.
- `DefaultProposalGenerator` (softmax authority-weighted edge-flip).
- `SequenceProposalGenerator` for deterministic CE replay.
- 8 failure-mode detectors in `emergo.diagnostics`.
- Matplotlib/networkx visualizations in `emergo.visualize`.
- 10 system invariants (INV-1 through INV-10) with full test coverage.
- Convergence analysis (`CONVERGENCE_ANALYSIS.md`) and formal specification (`SPECIFICATION.md`).

[Unreleased]: https://github.com/markhamcarter220-sketch/Emergo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/markhamcarter220-sketch/Emergo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/markhamcarter220-sketch/Emergo/releases/tag/v0.1.0
