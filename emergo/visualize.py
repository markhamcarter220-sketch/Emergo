"""Optional visualization utilities for Emergo diagnostics.

All plot functions return a matplotlib Figure, or None when matplotlib
or networkx is not installed — never raise ImportError.  The core emergo
package has no mandatory GUI dependency; install extras to enable plots::

    pip install matplotlib networkx

ASCII / JSON utilities (no optional deps):
  print_health_report(results)  — tabular health check to stdout / file
  export_diagnostics_json(diag, path)  — serialize KernelDiagnostics to JSON

Usage::

    from emergo.visualize import plot_authority_history, render_health_dashboard
    fig = plot_authority_history(diag)
    if fig:
        fig.savefig("authority.png")
    render_health_dashboard(diag, final_state, output_dir="./plots")
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emergo.diagnostics import DetectorResult, KernelDiagnostics
    from emergo.types import Graph, State

logger = logging.getLogger(__name__)

_MATPLOTLIB_WARNING_SENT = False
_NETWORKX_WARNING_SENT = False


def _try_matplotlib() -> Any:
    global _MATPLOTLIB_WARNING_SENT
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe in headless environments
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        if not _MATPLOTLIB_WARNING_SENT:
            logger.warning(
                "emergo.visualize: matplotlib not installed — plots disabled. "
                "Install with: pip install matplotlib"
            )
            _MATPLOTLIB_WARNING_SENT = True
        return None


def _try_networkx() -> Any:
    global _NETWORKX_WARNING_SENT
    try:
        import networkx as nx

        return nx
    except ImportError:
        if not _NETWORKX_WARNING_SENT:
            logger.warning(
                "emergo.visualize: networkx not installed — graph layouts disabled. "
                "Install with: pip install networkx"
            )
            _NETWORKX_WARNING_SENT = True
        return None


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_authority_history(
    diag: KernelDiagnostics,
    *,
    title: str = "Authority Score History",
    figsize: tuple = (10, 5),
    alpha: float = 0.85,
) -> Any | None:
    """Per-agent authority scores over accepted iterations.

    Returns a matplotlib Figure, or None if matplotlib is not installed or
    the diagnostics contain no accepted iterations.
    """
    plt = _try_matplotlib()
    if plt is None:
        return None

    accepted = [r for r in diag.records if r.ce_accepted]
    if not accepted:
        logger.warning("plot_authority_history: no accepted iterations to plot.")
        return None

    iterations = [r.iteration for r in accepted]
    fig, ax = plt.subplots(figsize=figsize)

    for agent_id in diag.agent_ids:
        scores = [r.authority_scores.get(agent_id, float("nan")) for r in accepted]
        ax.plot(iterations, scores, label=agent_id, alpha=alpha)

    ax.axhline(0.1, color="crimson", linestyle="--", linewidth=0.9, label="min_authority=0.1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Authority Score")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_phi_loss(
    diag: KernelDiagnostics,
    *,
    title: str = "φ Optimization Loss",
    figsize: tuple = (10, 4),
) -> Any | None:
    """φ training loss over iterations where phi_update ran.

    Uses log scale on the y-axis.  Returns None when matplotlib is absent
    or no phi_loss entries exist in the diagnostics.
    """
    plt = _try_matplotlib()
    if plt is None:
        return None

    loss_records = [(r.iteration, r.phi_loss) for r in diag.records if r.phi_loss is not None]
    if not loss_records:
        logger.warning("plot_phi_loss: no phi_loss data in diagnostics.")
        return None

    iters, losses = zip(*loss_records)
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(iters, losses, color="steelblue", linewidth=1.2, marker=".", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_edge_count(
    diag: KernelDiagnostics,
    *,
    title: str = "Active Edge Count",
    figsize: tuple = (10, 4),
) -> Any | None:
    """Active edge count over accepted iterations (step plot).

    Returns a matplotlib Figure, or None when matplotlib is absent.
    """
    plt = _try_matplotlib()
    if plt is None:
        return None

    accepted = [r for r in diag.records if r.ce_accepted]
    if not accepted:
        return None

    iters = [r.iteration for r in accepted]
    counts = [r.edge_count for r in accepted]

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(iters, counts, where="post", color="darkorange", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Active Edges")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_graph_evolution(
    graphs: Sequence[Graph],
    *,
    title: str = "Graph Evolution",
    max_snapshots: int = 6,
    figsize_per_panel: tuple = (3.5, 3.5),
) -> Any | None:
    """Grid of networkx snapshots showing how the topology changes over time.

    Samples up to max_snapshots evenly from the provided sequence.
    Requires both matplotlib and networkx.  Returns None if either is absent.
    """
    plt = _try_matplotlib()
    if plt is None:
        return None
    nx = _try_networkx()
    if nx is None:
        return None

    if not graphs:
        return None

    n = len(graphs)
    k = min(max_snapshots, n)
    indices = sorted(set(round(i * (n - 1) / max(k - 1, 1)) for i in range(k)))
    selected = [graphs[i] for i in indices]

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

    # Hide unused panels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for panel_idx, (snap_idx, G) in enumerate(zip(indices, selected)):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]
        ax.set_visible(True)

        DG = nx.DiGraph()
        DG.add_nodes_from(G.agent_ids)
        for i, from_id in enumerate(G.agent_ids):
            for j, to_id in enumerate(G.agent_ids):
                w = float(G.adjacency[i, j])
                if w > 0:
                    DG.add_edge(from_id, to_id, weight=w)

        pos = nx.spring_layout(DG, seed=42)
        edge_widths = [DG[u][v]["weight"] * 2 for u, v in DG.edges()] or [1.0]
        nx.draw_networkx(
            DG,
            pos=pos,
            ax=ax,
            node_size=400,
            font_size=8,
            arrows=True,
            node_color="skyblue",
            edge_color="steelblue",
            width=edge_widths,
            with_labels=True,
        )
        ax.set_title(f"t={snap_idx}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def render_health_dashboard(
    diag: KernelDiagnostics,
    final_state: State,
    output_dir: str = ".",
    *,
    diag_strict: KernelDiagnostics | None = None,
    graphs: Sequence[Graph] | None = None,
    prefix: str = "emergo",
) -> list[str]:
    """Run all visualizations and save PNGs to output_dir.

    Returns the list of file paths successfully saved.
    Plots that fail (missing deps, empty data) are silently skipped.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def _save(fig: Any, name: str) -> None:
        if fig is None:
            return
        path = out / f"{prefix}_{name}.png"
        try:
            fig.savefig(str(path), dpi=120, bbox_inches="tight")
            saved.append(str(path))
            logger.info("Saved visualization: %s", path)
        except Exception as exc:
            logger.warning("Failed to save %s: %s", path, exc)
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass

    _save(plot_authority_history(diag), "authority_history")
    _save(plot_phi_loss(diag), "phi_loss")
    _save(plot_edge_count(diag), "edge_count")

    if graphs:
        _save(plot_graph_evolution(list(graphs)), "graph_evolution")

    return saved


# ---------------------------------------------------------------------------
# ASCII / plain-text utilities (no optional deps)
# ---------------------------------------------------------------------------


def print_health_report(
    results: list[DetectorResult],
    out: IO[str] | None = None,
) -> None:
    """Print a formatted health-check table to a file-like object (default: stdout).

    Each row shows: status (OK / FAIL), detector name, severity, and evidence.
    Requires no external dependencies — always available.

    Args:
        results: List of DetectorResult objects from run_health_check().
        out:     Output stream.  Defaults to sys.stdout.
    """
    if out is None:
        out = sys.stdout

    col_name = 36
    col_sev = 8

    header = f"{'Detector':<{col_name}} {'Severity':>{col_sev}}  Status"
    separator = "-" * (len(header) + 2)
    print(header, file=out)
    print(separator, file=out)

    for r in results:
        status = "FAIL" if r.failure_detected else " OK "
        print(
            f"[{status}] {r.name:<{col_name}} {r.severity:>{col_sev}.2f}  {r.evidence}",
            file=out,
        )
        if r.failure_detected and r.recommendation:
            print(f"       Recommendation: {r.recommendation}", file=out)

    n_fail = sum(1 for r in results if r.failure_detected)
    print(separator, file=out)
    print(f"Total: {len(results)} detectors, {n_fail} failure(s) detected", file=out)


def export_diagnostics_json(
    diag: KernelDiagnostics,
    path: str,
    *,
    indent: int = 2,
) -> str:
    """Serialize KernelDiagnostics to a JSON file at path.

    Returns the absolute path written.  Raises OSError on write failure.
    Requires no external dependencies — always available.

    Fields serialized per IterationRecord:
        iteration, ce_attempted, ce_accepted, ce_type, ce_proposer,
        ce_participants, authority_scores, authority_delta,
        topology_entropy, edge_count, error_mean, phi_loss.
    """

    def _record_to_dict(r: Any) -> dict[str, Any]:
        return {
            "iteration": r.iteration,
            "ce_attempted": r.ce_attempted,
            "ce_accepted": r.ce_accepted,
            "ce_type": r.ce_type,
            "ce_proposer": r.ce_proposer,
            "ce_participants": list(r.ce_participants) if r.ce_participants else [],
            "authority_scores": r.authority_scores,
            "authority_delta": r.authority_delta,
            "topology_entropy": r.topology_entropy,
            "edge_count": r.edge_count,
            "error_mean": r.error_mean,
            "phi_loss": r.phi_loss,
        }

    payload: dict[str, Any] = {
        "agent_ids": list(diag.agent_ids),
        "n_records": len(diag.records),
        "records": [_record_to_dict(r) for r in diag.records],
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)

    logger.info("export_diagnostics_json: wrote %d records to %s", len(diag.records), out_path)
    return str(out_path.resolve())
