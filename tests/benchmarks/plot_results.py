"""
Plot benchmark results from bench_results.json.

Usage:
  python tests/benchmarks/plot_results.py                    # reads bench_results.json
  python tests/benchmarks/plot_results.py --input path.json  # custom input
  python tests/benchmarks/plot_results.py --out-dir plots/   # custom output dir

Produces:
  phi_loss_reduction.png   — φ-loss reduction % by task and system
  authority_gini.png       — Final authority Gini by task and system
  wall_time.png            — Wall time (s) by task and system (compute cost)
  ablation_comparison.png  — Side-by-side: Vanilla vs PhiLearning vs Emergo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# Consistent color palette — colorblind-friendly
_COLORS: dict[str, str] = {
    "vanilla": "#4878cf",        # blue
    "fixed_hierarchy": "#6acc65",  # green
    "performance_metric": "#d65f5f",  # red
    "phi_learning": "#b47cc7",   # purple
    "emergo": "#ee854a",         # orange
}
_LABELS: dict[str, str] = {
    "vanilla": "Vanilla",
    "fixed_hierarchy": "FixedHierarchy",
    "performance_metric": "PerfMetric",
    "phi_learning": "PhiLearning†",
    "emergo": "Emergo",
}
_MODE_ORDER = ["vanilla", "fixed_hierarchy", "performance_metric", "phi_learning", "emergo"]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _collect(data: dict, metric: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Returns {task_name: {mode: (mean, std)}}"""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for task_name, by_mode in data["results"].items():
        out[task_name] = {}
        for mode, runs in by_mode.items():
            vals = [r[metric] for r in runs]
            out[task_name][mode] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _bar_chart(
    data: dict[str, dict[str, tuple[float, float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: str,
    pct: bool = False,
) -> None:
    tasks = list(data.keys())
    modes = [m for m in _MODE_ORDER if any(m in data[t] for t in tasks)]
    x = np.arange(len(tasks))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.6), 5))

    for i, mode in enumerate(modes):
        means = []
        stds = []
        for task in tasks:
            if mode in data[task]:
                m, s = data[task][mode]
                means.append(m)
                stds.append(s)
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=_LABELS.get(mode, mode),
            color=_COLORS.get(mode, "#888888"),
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1, "alpha": 0.7},
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if pct:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


def _ablation_chart(data: dict, out_path: str) -> None:
    """Side-by-side Vanilla / PhiLearning / Emergo on φ-loss and Gini."""
    tasks = list(data["results"].keys())
    ablation_modes = ["vanilla", "phi_learning", "emergo"]

    phi_data = _collect(data, "phi_loss_reduction_pct")
    gini_data = _collect(data, "final_authority_gini")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric_data, ylabel, pct in [
        (axes[0], phi_data, "φ-Loss Reduction %", True),
        (axes[1], gini_data, "Final Authority Gini", False),
    ]:
        x = np.arange(len(tasks))
        width = 0.25
        for i, mode in enumerate(ablation_modes):
            means = [metric_data[t].get(mode, (0.0, 0.0))[0] for t in tasks]
            stds = [metric_data[t].get(mode, (0.0, 0.0))[1] for t in tasks]
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                means,
                width,
                label=_LABELS.get(mode, mode),
                color=_COLORS.get(mode, "#888"),
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 1, "alpha": 0.7},
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if pct:
            ax.set_ylim(bottom=0)

    fig.suptitle("Ablation: Vanilla vs PhiLearning† vs Emergo\n"
                 "(PhiLearning has φ-SGD but broadcast authority — isolates dual-channel effect)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="bench_results.json")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(
            f"Error: {args.input} not found. "
            "Run: python tests/benchmarks/bench_tasks.py --json",
            file=sys.stderr,
        )
        sys.exit(1)

    data = _load(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    phi_data = _collect(data, "phi_loss_reduction_pct")
    gini_data = _collect(data, "final_authority_gini")
    wall_data = _collect(data, "wall_seconds")

    _bar_chart(
        phi_data,
        "phi_loss_reduction_pct",
        "φ-Loss Reduction %",
        "φ-Loss Reduction by Task and System",
        os.path.join(args.out_dir, "phi_loss_reduction.png"),
        pct=True,
    )
    _bar_chart(
        gini_data,
        "final_authority_gini",
        "Final Authority Gini",
        "Final Authority Gini by Task and System\n(higher = more differentiated)",
        os.path.join(args.out_dir, "authority_gini.png"),
    )
    _bar_chart(
        wall_data,
        "wall_seconds",
        "Wall Time (s)",
        "Compute Cost: Wall Time by Task and System\n(Emergo's overhead is a real loss)",
        os.path.join(args.out_dir, "wall_time.png"),
    )
    _ablation_chart(data, os.path.join(args.out_dir, "ablation_comparison.png"))


if __name__ == "__main__":
    main()
