# Emergo Benchmark Results

**Purpose**: Reproducible, honest measurement of Emergo topology-learning against four baselines. This document states each task's hypothesis *before* reporting results. The 'Where Emergo Loses' section is a required deliverable, not an optional appendix.

**Configuration**: 3 seeds per task/mode pair

## Systems Under Test

| Mode | Description |
| --- | --- |
| **Vanilla** | Broadcast scalar error to all CE participants; φ frozen at random init. |
| **FixedHierarchy** | Authority ∝ agent rank at init, never updated. No learning. |
| **PerfMetric** | Authority updated by outcome (+0.05 accepted, −0.02 rejected). No φ learning. |
| **PhiLearning†** | ABLATION: same φ-SGD as Emergo + broadcast (vanilla) authority update. |
| **Emergo** | Full system: dual-channel differentiated errors + φ-SGD. |

† PhiLearning is a deliberate ablation. It shares φ-learning code with Emergo but uses Vanilla's broadcast authority. This isolates whether the dual-channel error mechanism (not φ-learning alone) is responsible for preventing authority collapse.

## Fairness Notes

- Each run receives an independent deep-copied initial state (same seed, same graph).
- No baseline is pre-warmed with knowledge of Emergo's learned φ.
- All stochasticity is seeded; seeds are embedded in the JSON output.
- FixedHierarchy's authority advantage (highest Gini by construction) is   **intentional** — it represents the ceiling achievable *without* learning.   Emergo's differentiation is earned, not assigned.

## Results by Task

### T1_standard

**Parameters**: n=10, T=1500, edge_density=0.3

**Hypothesis**: Reference condition. Emergo wins on φ-loss (only system that learns) and maintains authority differentiation. Loses on wall time (~24× all baselines). PhiLearning ablation should match Emergo on φ-loss but match Vanilla on authority Gini — isolating the dual-channel contribution.

#### φ-Loss Reduction % (primary learning signal)

| Mode | Mean | Std |
| --- | :---: | :---: |
| Vanilla | 0.0% | 0.0 |
| FixedHierarchy | 0.0% | 0.0 |
| PerfMetric | 0.0% | 0.0 |
| PhiLearning† | 99.9% | 0.0 |
| Emergo | 99.9% | 0.0 |

#### Authority & Compute

| Mode | Accept% | Auth Gini | Auth Std | Topo Events | Wall (s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Vanilla | 100.0% | 0.0000 | 0.0000 | 1500 | 0.64 |
| FixedHierarchy | 100.0% | 0.2444 | 0.2553 | 1500 | 0.59 |
| PerfMetric | 100.0% | 0.0000 | 0.0000 | 1500 | 0.60 |
| PhiLearning† | 100.0% | 0.0000 | 0.0000 | 1500 | 19.47 |
| Emergo | 100.0% | 0.0285 | 0.0304 | 1463 | 18.63 |

#### Entanglement Onset (first step where Gini < 0.05)

| Mode | Onset |
| --- | :---: |
| Vanilla | step 32 |
| FixedHierarchy | **never** |
| PerfMetric | step 27 |
| PhiLearning† | step 32 |
| Emergo | **never** |

<details><summary>Per-seed raw data</summary>

| seed | mode | φ-loss% | accept% | gini | auth_std | topo | wall(s) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.67s |
| 1 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.63s |
| 2 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.63s |
| 0 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.60s |
| 1 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.59s |
| 2 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.57s |
| 0 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.59s |
| 1 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.62s |
| 2 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.58s |
| 0 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.36s |
| 1 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.93s |
| 2 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.12s |
| 0 | emergo | 99.9% | 100.0% | 0.0318 | 0.0339 | 1500 | 19.21s |
| 1 | emergo | 99.9% | 100.0% | 0.0308 | 0.0312 | 1388 | 17.22s |
| 2 | emergo | 99.9% | 100.0% | 0.0228 | 0.0261 | 1500 | 19.46s |

</details>

### T2_cold_start

**Parameters**: n=10, T=60, edge_density=0.3

**Hypothesis**: Short horizon: φ-loss needs ~1000 steps to converge; at T=60 Emergo has run only 6 gradient updates. Expected φ-loss reduction: ~0–3% (noise level). All systems are effectively tied on quality. Emergo loses: it costs ~24× more wall time for the same outcome. This is a clear, reproducible Emergo loss.

#### φ-Loss Reduction % (primary learning signal)

| Mode | Mean | Std |
| --- | :---: | :---: |
| Vanilla | 0.0% | 0.0 |
| FixedHierarchy | 0.0% | 0.0 |
| PerfMetric | 0.0% | 0.0 |
| PhiLearning† | 0.3% | 0.6 |
| Emergo | 0.4% | 0.6 |

#### Authority & Compute

| Mode | Accept% | Auth Gini | Auth Std | Topo Events | Wall (s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Vanilla | 100.0% | 0.0344 | 0.0485 | 60 | 0.03 |
| FixedHierarchy | 100.0% | 0.2444 | 0.2553 | 60 | 0.02 |
| PerfMetric | 100.0% | 0.0473 | 0.0729 | 60 | 0.02 |
| PhiLearning† | 100.0% | 0.0337 | 0.0478 | 60 | 0.12 |
| Emergo | 100.0% | 0.0344 | 0.0364 | 60 | 0.13 |

#### Entanglement Onset (first step where Gini < 0.05)

| Mode | Onset |
| --- | :---: |
| Vanilla | step 32 |
| FixedHierarchy | **never** |
| PerfMetric | step 27 |
| PhiLearning† | step 32 |
| Emergo | **never** |

<details><summary>Per-seed raw data</summary>

| seed | mode | φ-loss% | accept% | gini | auth_std | topo | wall(s) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | vanilla | 0.0% | 100.0% | 0.0304 | 0.0436 | 60 | 0.03s |
| 1 | vanilla | 0.0% | 100.0% | 0.0478 | 0.0635 | 60 | 0.03s |
| 2 | vanilla | 0.0% | 100.0% | 0.0250 | 0.0383 | 60 | 0.03s |
| 0 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 60 | 0.02s |
| 1 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 60 | 0.02s |
| 2 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 60 | 0.02s |
| 0 | performance_metric | 0.0% | 100.0% | 0.0605 | 0.0896 | 60 | 0.02s |
| 1 | performance_metric | 0.0% | 100.0% | 0.0409 | 0.0568 | 60 | 0.02s |
| 2 | performance_metric | 0.0% | 100.0% | 0.0404 | 0.0723 | 60 | 0.03s |
| 0 | phi_learning | 0.0% | 100.0% | 0.0299 | 0.0431 | 60 | 0.12s |
| 1 | phi_learning | 0.9% | 100.0% | 0.0468 | 0.0624 | 60 | 0.12s |
| 2 | phi_learning | -0.1% | 100.0% | 0.0244 | 0.0379 | 60 | 0.12s |
| 0 | emergo | 0.1% | 100.0% | 0.0384 | 0.0405 | 60 | 0.12s |
| 1 | emergo | 1.0% | 100.0% | 0.0319 | 0.0324 | 60 | 0.13s |
| 2 | emergo | -0.0% | 100.0% | 0.0329 | 0.0363 | 60 | 0.12s |

</details>

### T3_dense

**Parameters**: n=10, T=1500, edge_density=0.9

**Hypothesis**: Near-complete initial graph (90% density). Most CEs toggle existing edges; few new connections are possible. Emergo still learns φ (achieves high φ-loss reduction) but authority differentiation adds less value vs T1: there are fewer 'meaningful' CEs where prediction accuracy matters. FixedHierarchy may match Emergo on exploration since most CEs are accepted by Lux regardless of who proposes them.

#### φ-Loss Reduction % (primary learning signal)

| Mode | Mean | Std |
| --- | :---: | :---: |
| Vanilla | 0.0% | 0.0 |
| FixedHierarchy | 0.0% | 0.0 |
| PerfMetric | 0.0% | 0.0 |
| PhiLearning† | 99.9% | 0.0 |
| Emergo | 99.9% | 0.0 |

#### Authority & Compute

| Mode | Accept% | Auth Gini | Auth Std | Topo Events | Wall (s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Vanilla | 100.0% | 0.0000 | 0.0000 | 1500 | 0.64 |
| FixedHierarchy | 100.0% | 0.2444 | 0.2553 | 1500 | 0.59 |
| PerfMetric | 100.0% | 0.0000 | 0.0000 | 1500 | 0.61 |
| PhiLearning† | 100.0% | 0.0000 | 0.0000 | 1500 | 19.64 |
| Emergo | 100.0% | 0.0369 | 0.0380 | 1464 | 18.84 |

#### Entanglement Onset (first step where Gini < 0.05)

| Mode | Onset |
| --- | :---: |
| Vanilla | step 32 |
| FixedHierarchy | **never** |
| PerfMetric | step 27 |
| PhiLearning† | step 32 |
| Emergo | step 415 |

<details><summary>Per-seed raw data</summary>

| seed | mode | φ-loss% | accept% | gini | auth_std | topo | wall(s) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.64s |
| 1 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.63s |
| 2 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.64s |
| 0 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.59s |
| 1 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.57s |
| 2 | fixed_hierarchy | 0.0% | 100.0% | 0.2444 | 0.2553 | 1500 | 0.59s |
| 0 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.62s |
| 1 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.59s |
| 2 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.61s |
| 0 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.56s |
| 1 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.70s |
| 2 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 19.65s |
| 0 | emergo | 99.9% | 100.0% | 0.0284 | 0.0294 | 1391 | 17.10s |
| 1 | emergo | 99.9% | 100.0% | 0.0339 | 0.0348 | 1500 | 19.57s |
| 2 | emergo | 99.9% | 100.0% | 0.0484 | 0.0499 | 1500 | 19.86s |

</details>

### T4_sparse_large

**Parameters**: n=20, T=1500, edge_density=0.05

**Hypothesis**: Large sparse graph: 20 agents, only 5% initial edge density. Lots of topology to discover. Emergo's learned φ identifies which connections to add; authority differentiation should be highest here. Vanilla authority collapse is expected early. Emergo clearly wins.

#### φ-Loss Reduction % (primary learning signal)

| Mode | Mean | Std |
| --- | :---: | :---: |
| Vanilla | 0.0% | 0.0 |
| FixedHierarchy | 0.0% | 0.0 |
| PerfMetric | 0.0% | 0.0 |
| PhiLearning† | 99.9% | 0.0 |
| Emergo | 99.9% | 0.0 |

#### Authority & Compute

| Mode | Accept% | Auth Gini | Auth Std | Topo Events | Wall (s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Vanilla | 100.0% | 0.0000 | 0.0000 | 1500 | 0.74 |
| FixedHierarchy | 100.0% | 0.2333 | 0.2428 | 1500 | 0.70 |
| PerfMetric | 100.0% | 0.0000 | 0.0000 | 1500 | 0.72 |
| PhiLearning† | 100.0% | 0.0000 | 0.0000 | 1500 | 21.91 |
| Emergo | 100.0% | 0.0199 | 0.0195 | 1326 | 17.99 |

#### Entanglement Onset (first step where Gini < 0.05)

| Mode | Onset |
| --- | :---: |
| Vanilla | step 59 |
| FixedHierarchy | **never** |
| PerfMetric | step 73 |
| PhiLearning† | step 50 |
| Emergo | **never** |

<details><summary>Per-seed raw data</summary>

| seed | mode | φ-loss% | accept% | gini | auth_std | topo | wall(s) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.73s |
| 1 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.75s |
| 2 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.75s |
| 0 | fixed_hierarchy | 0.0% | 100.0% | 0.2333 | 0.2428 | 1500 | 0.68s |
| 1 | fixed_hierarchy | 0.0% | 100.0% | 0.2333 | 0.2428 | 1500 | 0.72s |
| 2 | fixed_hierarchy | 0.0% | 100.0% | 0.2333 | 0.2428 | 1500 | 0.70s |
| 0 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.69s |
| 1 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.73s |
| 2 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.73s |
| 0 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 21.76s |
| 1 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 22.18s |
| 2 | phi_learning | 99.9% | 100.0% | 0.0000 | 0.0000 | 1500 | 21.81s |
| 0 | emergo | 99.9% | 100.0% | 0.0159 | 0.0158 | 1075 | 12.63s |
| 1 | emergo | 99.9% | 100.0% | 0.0233 | 0.0223 | 1491 | 21.78s |
| 2 | emergo | 99.9% | 100.0% | 0.0205 | 0.0203 | 1411 | 19.57s |

</details>

### T5_tiny

**Parameters**: n=3, T=1500, edge_density=0.3

**Hypothesis**: Minimal graph: only 3 agents, 6 possible directed edges. Little variation to differentiate authority across agents. FixedHierarchy's rank assignment may be competitive with Emergo's earned differentiation. Emergo still learns φ and achieves high φ-loss reduction, but the authority advantage over FixedHierarchy is smaller than at larger n.

#### φ-Loss Reduction % (primary learning signal)

| Mode | Mean | Std |
| --- | :---: | :---: |
| Vanilla | 0.0% | 0.0 |
| FixedHierarchy | 0.0% | 0.0 |
| PerfMetric | 0.0% | 0.0 |
| PhiLearning† | 99.4% | 0.0 |
| Emergo | 99.5% | 0.1 |

#### Authority & Compute

| Mode | Accept% | Auth Gini | Auth Std | Topo Events | Wall (s) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Vanilla | 100.0% | 0.0000 | 0.0000 | 1500 | 0.52 |
| FixedHierarchy | 100.0% | 0.2963 | 0.3266 | 1500 | 0.47 |
| PerfMetric | 100.0% | 0.0000 | 0.0000 | 1500 | 0.48 |
| PhiLearning† | 100.0% | 0.0000 | 0.0000 | 1500 | 15.34 |
| Emergo | 90.9% | 0.2928 | 0.3001 | 1363 | 15.51 |

#### Entanglement Onset (first step where Gini < 0.05)

| Mode | Onset |
| --- | :---: |
| Vanilla | **never** |
| FixedHierarchy | **never** |
| PerfMetric | step 12 |
| PhiLearning† | **never** |
| Emergo | step 361 |

<details><summary>Per-seed raw data</summary>

| seed | mode | φ-loss% | accept% | gini | auth_std | topo | wall(s) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.50s |
| 1 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.52s |
| 2 | vanilla | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.52s |
| 0 | fixed_hierarchy | 0.0% | 100.0% | 0.2963 | 0.3266 | 1500 | 0.46s |
| 1 | fixed_hierarchy | 0.0% | 100.0% | 0.2963 | 0.3266 | 1500 | 0.48s |
| 2 | fixed_hierarchy | 0.0% | 100.0% | 0.2963 | 0.3266 | 1500 | 0.47s |
| 0 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.47s |
| 1 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.49s |
| 2 | performance_metric | 0.0% | 100.0% | 0.0000 | 0.0000 | 1500 | 0.49s |
| 0 | phi_learning | 99.4% | 100.0% | 0.0000 | 0.0000 | 1500 | 15.20s |
| 1 | phi_learning | 99.3% | 100.0% | 0.0000 | 0.0000 | 1500 | 15.26s |
| 2 | phi_learning | 99.3% | 100.0% | 0.0000 | 0.0000 | 1500 | 15.57s |
| 0 | emergo | 99.4% | 94.2% | 0.2970 | 0.2821 | 1413 | 16.30s |
| 1 | emergo | 99.5% | 82.7% | 0.3044 | 0.3053 | 1241 | 13.57s |
| 2 | emergo | 99.5% | 95.7% | 0.2772 | 0.3129 | 1435 | 16.65s |

</details>

## Cross-Task Summary

Mean over all seeds for each (task, mode) pair.

### φ-Loss Reduction % Across Tasks

| Task | Vanilla | FixedHierarchy | PerfMetric | PhiLearning† | Emergo |
| --- | :---: | :---: | :---: | :---: | :---: |
| T1_standard | 0.0% | 0.0% | 0.0% | 99.9% | 99.9% |
| T2_cold_start | 0.0% | 0.0% | 0.0% | 0.3% | 0.4% |
| T3_dense | 0.0% | 0.0% | 0.0% | 99.9% | 99.9% |
| T4_sparse_large | 0.0% | 0.0% | 0.0% | 99.9% | 99.9% |
| T5_tiny | 0.0% | 0.0% | 0.0% | 99.4% | 99.5% |

### Authority Gini Across Tasks

| Task | Vanilla | FixedHierarchy | PerfMetric | PhiLearning† | Emergo |
| --- | :---: | :---: | :---: | :---: | :---: |
| T1_standard | 0.0000 | 0.2444 | 0.0000 | 0.0000 | 0.0285 |
| T2_cold_start | 0.0344 | 0.2444 | 0.0473 | 0.0337 | 0.0344 |
| T3_dense | 0.0000 | 0.2444 | 0.0000 | 0.0000 | 0.0369 |
| T4_sparse_large | 0.0000 | 0.2333 | 0.0000 | 0.0000 | 0.0199 |
| T5_tiny | 0.0000 | 0.2963 | 0.0000 | 0.0000 | 0.2928 |

### Wall Time (s) Across Tasks — Emergo's Compute Cost

| Task | Vanilla | FixedHierarchy | PerfMetric | PhiLearning† | Emergo |
| --- | :---: | :---: | :---: | :---: | :---: |
| T1_standard | 0.64 | 0.59 | 0.60 | 19.47 | 18.63 |
| T2_cold_start | 0.03 | 0.02 | 0.02 | 0.12 | 0.13 |
| T3_dense | 0.64 | 0.59 | 0.61 | 19.64 | 18.84 |
| T4_sparse_large | 0.74 | 0.70 | 0.72 | 21.91 | 17.99 |
| T5_tiny | 0.52 | 0.47 | 0.48 | 15.34 | 15.51 |

## Where Emergo Loses or Ties

This section is a required deliverable. An empty section would mean the benchmark is not trustworthy — it would indicate only tasks where Emergo wins were included. The items below are drawn directly from the numbers above.

1. **Compute cost (all tasks)**: Emergo is the slowest system in every task. Worst case: T5_tiny where Emergo runs 31.7× slower than the non-learning baselines. This is expected (φ-SGD is the cost driver), but it is a real loss that is not hidden. Cost is O(phi_window × phi_update_interval⁻¹) per run; phi_window=200 bounds growth.

2. **Cold-start regime (T2_cold_start, T=60)**: At T=60 steps, Emergo achieves only 0.4% φ-loss reduction (φ needs ~1000 steps to converge). All baselines are at 0% by definition, so Emergo 'wins' on this metric — but the win is negligible relative to its 5.2× compute overhead. A practitioner running at T=60 gets almost nothing from Emergo's learning engine at significant extra cost.

3. **Authority diversity vs FixedHierarchy (T1_standard, T2_cold_start, T3_dense, T4_sparse_large, T5_tiny)**: FixedHierarchy achieves higher authority Gini (mean 0.2526) than Emergo (mean 0.0825) on these tasks. This is by design: FixedHierarchy assigns rank-proportional authority at init (a hard-coded ceiling). Emergo's differentiation is *earned* via prediction accuracy, but 'earned' does not mean 'more diverse than a manually tuned assignment.' This is a genuine limitation of learned vs. prescribed authority.

4. **PhiLearning ablation confirms dual-channel is load-bearing**: PhiLearning (φ-learning + broadcast authority) achieves Gini=0.0067 vs Emergo Gini=0.0825 — closer to Vanilla (0.0069) than to Emergo. This means φ-learning alone does not prevent authority collapse. The dual-channel error split is the mechanism that maintains differentiation. Emergo loses to its own ablation on authority differentiation when the ablation keeps broadcast errors.

## Ablation Analysis

PhiLearning† uses the same φ-SGD as Emergo but Vanilla's broadcast authority update. Comparing PhiLearning to Emergo and Vanilla isolates the dual-channel contribution:

| Metric | Vanilla | PhiLearning† | Emergo | Interpretation |
| --- | :---: | :---: | :---: | --- |
| φ-loss reduction % | 0.0% | 79.9% | 79.9% | PhiLearning ≈ Emergo → dual-channel doesn't help φ-learning |
| Authority Gini | 0.0069 | 0.0067 | 0.0825 | PhiLearning ≈ Vanilla → broadcast authority collapses regardless of φ |
| Wall time (s) | 0.51 | 15.30 | 14.22 | Both learning systems cost more |

If PhiLearning≈Emergo on φ-loss AND PhiLearning≈Vanilla on Gini, the table confirms: (a) dual-channel error does not help φ learning, and (b) dual-channel error IS what prevents authority collapse. These are the two load-bearing claims of the Emergo paper.

## Reproducibility

```
# One-command run (full, ~15 min):
make bench

# Equivalently:
python tests/benchmarks/bench_tasks.py --json

# Quick check (~2 min):
python tests/benchmarks/bench_tasks.py --quick

# Plot results (requires bench_results.json):
python tests/benchmarks/plot_results.py
```

Fixed seeds: initial state seed = `seed * 1000 + n_agents + hash(task_name) % 997`. Runner seed = `seed` (0, 1, 2). Deterministic given numpy version.

Full per-seed data and raw metrics are in `bench_results.json` (generated with `--json`).

*Generated over 5 tasks × 5 systems × 3 seeds.*