"""emergo CLI — Typer + Rich command-line interface for the Emergo kernel.

Commands:
  emergo run       — Run the kernel loop (ring graph, configurable)
  emergo demo      — Showcase sub-commands: convergence | multi-agent | executor | stress
  emergo viz       — Render visualizations from saved diagnostics or a fresh run
  emergo health    — Run the 8-detector health check and print a rich report

Config precedence: CLI flags  >  env vars  >  compiled defaults
  EMERGO_AGENTS, EMERGO_ITERATIONS, EMERGO_SEED, EMERGO_OUTPUT_DIR, EMERGO_OPTIMIZER
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
import typer

console = Console()

# ---------------------------------------------------------------------------
# App & demo sub-app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="emergo",
    help=(
        "[bold cyan]Emergo[/bold cyan] — Self-Organizing Intelligence Engine\n\n"
        "Agents earn authority by accurately predicting how their actions reshape\n"
        "the coordination network, causing intelligence to emerge from structure."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)

demo_app = typer.Typer(
    name="demo",
    help="Interactive showcase demos illustrating Emergo's core behaviors.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(demo_app, name="demo")

checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Manage kernel run checkpoints (requires emergo[store]).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(checkpoint_app, name="checkpoint")


# ---------------------------------------------------------------------------
# Rich-powered progress observer  (INV-10 safe — exceptions never propagate)
# ---------------------------------------------------------------------------


class _RichProgressObserver:
    """Drives a Rich progress bar from inside the kernel loop."""

    def __init__(self, progress: Progress, task_id: TaskID, max_iters: int) -> None:
        self._progress = progress
        self._task_id = task_id
        self._max_iters = max_iters

    def on_iteration_start(self, t: int, state: object) -> None:
        self._progress.update(self._task_id, completed=t)

    def on_ce_result(self, t: int, ce: object, accepted: bool, errors: object) -> None:
        pass

    def on_phi_updated(self, t: int, loss: float) -> None:
        self._progress.update(
            self._task_id,
            description=f"[cyan]Running[/cyan]  φ_loss=[yellow]{loss:.4f}[/yellow]",
        )

    def on_kernel_done(self, reason: str, state: object, n_iterations: int) -> None:
        self._progress.update(self._task_id, completed=n_iterations)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_ring(n: int) -> tuple:
    """Return (G0, phi0, A0) for an n-agent directed ring graph."""
    from emergo import Graph, make_initial_authority, make_initial_phi

    ids = tuple(f"agent_{i}" for i in range(n))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        adj[i, (i + 1) % n] = 0.5
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=0)
    A0 = make_initial_authority(ids, baseline=0.5)
    return G0, phi0, A0


def _authority_table(A: object, title: str = "Final Authority") -> Table:
    from emergo import Authority

    assert isinstance(A, Authority)
    table = Table(title=title, header_style="bold magenta", show_header=True)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Bar", min_width=22)
    for aid in sorted(A.scores):
        score = A.get(aid)
        filled = int(score * 20)
        bar = "[green]" + "█" * filled + "[/green]" + "░" * (20 - filled)
        table.add_row(aid, f"{score:.4f}", bar)
    return table


def _summary_panel(reason: str, n_agents: int, n_iters: int, obs: object) -> Panel:
    from emergo import HistoryObserver

    assert isinstance(obs, HistoryObserver)
    rows: list[tuple[str, str]] = [
        ("Termination", reason),
        ("Agents", str(n_agents)),
        ("Max iterations", str(n_iters)),
        ("CE acceptance", f"{obs.ce_acceptance_rate:.1%}"),
    ]
    if obs.phi_losses:
        losses = [loss for _, loss in obs.phi_losses]
        rows.append(("phi loss (first -> last)", f"{losses[0]:.5f} -> {losses[-1]:.5f}"))
    inner = Table(show_header=False, box=None, padding=(0, 2))
    inner.add_column(style="bold")
    inner.add_column()
    for k, v in rows:
        inner.add_row(k, v)
    return Panel(inner, title="[bold cyan]Run Summary[/bold cyan]", border_style="cyan")


def _make_progress(total: int) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# emergo run
# ---------------------------------------------------------------------------


@app.command()
def run(
    agents: int = typer.Option(
        8, "--agents", "-n", envvar="EMERGO_AGENTS", help="Number of agents"
    ),
    iterations: int = typer.Option(
        2000, "--iterations", "-i", envvar="EMERGO_ITERATIONS", help="Max kernel iterations"
    ),
    seed: int = typer.Option(0, "--seed", envvar="EMERGO_SEED", help="RNG seed"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        envvar="EMERGO_OUTPUT_DIR",
        help="Save diagnostics pickle and plots here",
    ),
    collect_diagnostics: bool = typer.Option(
        False,
        "--collect-diagnostics/--no-collect-diagnostics",
        help="Record per-iteration diagnostics (required for --viz)",
    ),
    do_viz: bool = typer.Option(
        False,
        "--viz/--no-viz",
        help="Auto-generate plots after the run (requires [viz] extra)",
    ),
    optimizer: str = typer.Option(
        "sgd", "--optimizer", envvar="EMERGO_OPTIMIZER", help="φ optimizer: sgd or adam"
    ),
    threshold: float = typer.Option(1e-4, "--threshold", help="Convergence plateau threshold"),
    phi_lr: float | None = typer.Option(None, "--phi-lr", help="φ learning rate override"),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel kernel runs (>1 enables distributed mode)",
    ),
    checkpoint_db: Path | None = typer.Option(
        None,
        "--checkpoint-db",
        help="SQLite path for auto-checkpointing (e.g. runs.db)",
    ),
    checkpoint_every: int = typer.Option(
        100,
        "--checkpoint-every",
        help="Auto-checkpoint every N accepted CEs (requires --checkpoint-db)",
    ),
) -> None:
    """Run the Emergo kernel loop on a ring graph.

    Config precedence: CLI flags > env vars
    (EMERGO_AGENTS, EMERGO_ITERATIONS, EMERGO_SEED, EMERGO_OUTPUT_DIR, EMERGO_OPTIMIZER)
    > defaults.

    Use [bold]--workers N[/bold] to run N independent parallel kernels (different seeds)
    and report the best result.  Use [bold]--checkpoint-db[/bold] to save checkpoints to
    SQLite for long runs.
    """
    from emergo import HistoryObserver, emergo_kernel

    if workers > 1:
        _run_distributed(
            agents=agents,
            iterations=iterations,
            seed=seed,
            optimizer=optimizer,
            threshold=threshold,
            phi_lr=phi_lr,
            workers=workers,
        )
        return

    G0, phi0, A0 = _build_ring(agents)
    obs = HistoryObserver()
    need_diag = collect_diagnostics or do_viz
    run_observers: list[object] = [obs]

    # Auto-checkpointing
    if checkpoint_db is not None:
        try:
            import uuid

            from emergo.store import CheckpointKernelObserver, make_store

            run_id = f"run_{uuid.uuid4().hex[:8]}"
            store = make_store("sqlite", db_path=str(checkpoint_db))
            ckpt_obs = CheckpointKernelObserver(
                store, run_id=run_id, every_n_accepted=checkpoint_every
            )
            run_observers.append(ckpt_obs)
            console.print(
                f"[dim]Checkpointing to {checkpoint_db} every {checkpoint_every} "
                f"accepted CEs (run_id={run_id})[/dim]"
            )
        except ImportError:
            console.print(
                "[yellow]Warning: store module unavailable — skipping checkpointing[/yellow]"
            )

    with _make_progress(iterations) as prog:
        tid = prog.add_task(f"[cyan]Running[/cyan]  {agents} agents", total=iterations)
        prog_obs = _RichProgressObserver(prog, tid, iterations)
        run_observers.append(prog_obs)

        result = emergo_kernel(
            initial_state=(G0, phi0, A0, []),
            max_iterations=iterations,
            convergence_threshold=threshold,
            phi_optimizer=optimizer,
            phi_lr=phi_lr,
            observers=run_observers,  # type: ignore[arg-type]
            collect_diagnostics=need_diag,
            rng=np.random.default_rng(seed),
        )

    if need_diag:
        final_state, reason, diag = result  # type: ignore[misc]
    else:
        final_state, reason = result  # type: ignore[misc]
        diag = None

    _, _, A_final, _ = final_state
    console.print(_summary_panel(reason, agents, iterations, obs))
    console.print(_authority_table(A_final))

    if output_dir or do_viz:
        out = Path(output_dir) if output_dir else Path("emergo_output")
        out.mkdir(parents=True, exist_ok=True)

        if diag is not None:
            pkl_path = out / "emergo_run.pkl"
            with open(pkl_path, "wb") as fh:
                pickle.dump((final_state, reason, diag), fh)
            console.print(f"[dim]Diagnostics saved → {pkl_path}[/dim]")

        if do_viz and diag is not None:
            try:
                from emergo.visualize import render_health_dashboard

                saved = render_health_dashboard(diag, final_state, output_dir=str(out))
                for p in saved:
                    console.print(f"[green]Plot → {p}[/green]")
            except ImportError:
                console.print(
                    "[yellow]Warning: viz extra not installed. "
                    "Run: pip install 'emergo[viz]'[/yellow]"
                )


# ---------------------------------------------------------------------------
# emergo run --workers helper (distributed mode)
# ---------------------------------------------------------------------------


def _run_distributed(
    agents: int,
    iterations: int,
    seed: int,
    optimizer: str,
    threshold: float,
    phi_lr: float | None,
    workers: int,
) -> None:
    """Run N parallel kernel instances and report results."""
    from emergo.distributed import KernelConfig, run_parallel_kernels

    console.print(
        f"[bold cyan]Distributed mode:[/bold cyan] {workers} workers x "
        f"{agents} agents x {iterations} iterations"
    )
    G0, phi0, A0 = _build_ring(agents)
    initial_state = (G0, phi0, A0, [])

    configs = [
        KernelConfig(
            initial_state=initial_state,
            max_iterations=iterations,
            convergence_threshold=threshold,
            phi_optimizer=optimizer,
            phi_lr=phi_lr,
            seed=seed + i,
        )
        for i in range(workers)
    ]

    with console.status("[cyan]Running parallel kernels…[/cyan]"):
        results = run_parallel_kernels(configs, n_workers=workers)

    table = Table(title=f"Parallel Results ({workers} workers)", header_style="bold magenta")
    table.add_column("Seed", justify="right", style="dim")
    table.add_column("Reason", style="cyan")
    table.add_column("φ loss", justify="right")
    table.add_column("Wall time", justify="right")
    table.add_column("Error", style="red")

    for r in results:
        phi_str = f"{r.final_phi_loss:.5f}" if r.final_phi_loss is not None else "—"
        wall_str = f"{r.wall_time_seconds:.2f}s"
        err_str = r.error or ""
        table.add_row(str(r.config.seed), r.reason, phi_str, wall_str, err_str)

    console.print(table)

    from emergo.distributed import best_converged, summarize_results

    summary = summarize_results(results)
    console.print(
        f"\n  Converged: {summary['n_converged']}/{summary['n_total']}  |  "
        f"Failed: {summary['n_failed']}  |  "
        f"Best φ loss: {summary['best_phi_loss']:.5f if summary['best_phi_loss'] is not None else '—'}  |  "
        f"Avg wall time: {summary['avg_wall_time']:.2f}s"
    )
    best = best_converged(results)
    if best is not None:
        _, _, A_best, _ = best.final_state
        console.print(_authority_table(A_best, title=f"Best Run (seed={best.config.seed})"))


# ---------------------------------------------------------------------------
# emergo checkpoint list / load
# ---------------------------------------------------------------------------


@checkpoint_app.command("list")
def checkpoint_list(
    db: Path = typer.Option(
        Path("runs.db"),
        "--db",
        help="SQLite database path",
    ),
) -> None:
    """List all runs and checkpoint counts in a SQLite store."""
    try:
        from emergo.store import make_store
    except ImportError as exc:
        console.print("[red]Error: emergo.store not available[/red]")
        raise typer.Exit(code=1) from exc

    if not db.exists():
        console.print(f"[yellow]No database found at {db}[/yellow]")
        raise typer.Exit(code=0)

    store = make_store("sqlite", db_path=str(db))
    runs = store.list_runs()
    store.close()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title=f"Runs in {db}", header_style="bold magenta")
    table.add_column("Run ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    table.add_column("Checkpoints", justify="right")

    for r in runs:
        table.add_row(r.run_id, r.created_at[:19], r.updated_at[:19], str(r.n_checkpoints))
    console.print(table)


@checkpoint_app.command("load")
def checkpoint_load(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to load"),
    db: Path = typer.Option(Path("runs.db"), "--db", help="SQLite database path"),
    iterations: int = typer.Option(
        0,
        "--continue-iterations",
        "-i",
        help="Continue kernel for N more iterations after loading (0 = just inspect)",
    ),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Load a checkpoint and optionally continue the kernel run."""
    try:
        from emergo.store import make_store
    except ImportError as exc:
        console.print("[red]Error: emergo.store not available[/red]")
        raise typer.Exit(code=1) from exc

    store = make_store("sqlite", db_path=str(db))
    try:
        state, meta = store.load(checkpoint_id)
    except KeyError as exc:
        console.print(f"[red]Checkpoint {checkpoint_id!r} not found in {db}[/red]")
        raise typer.Exit(code=1) from exc
    finally:
        store.close()

    _, _, A_loaded, _ = state
    console.print(
        f"[bold]Loaded checkpoint[/bold] {meta.checkpoint_id[:12]}…  "
        f"(run={meta.run_id}, iteration={meta.iteration}, reason={meta.reason or '—'})"
    )
    console.print(_authority_table(A_loaded, title="Authority at Checkpoint"))

    if iterations > 0:
        from emergo import HistoryObserver, emergo_kernel

        obs = HistoryObserver()
        with _make_progress(iterations) as prog:
            tid = prog.add_task("[cyan]Continuing[/cyan]", total=iterations)
            result = emergo_kernel(
                initial_state=state,
                max_iterations=iterations,
                observers=[obs, _RichProgressObserver(prog, tid, iterations)],
                rng=np.random.default_rng(seed),
            )
        final_state, reason = result  # type: ignore[misc]
        _, _, A_final, _ = final_state
        console.print(_summary_panel(reason, A_final.scores.__len__(), iterations, obs))
        console.print(_authority_table(A_final, title="Authority After Continuation"))


# ---------------------------------------------------------------------------
# emergo viz
# ---------------------------------------------------------------------------


@app.command("viz")
def viz_cmd(
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Diagnostics pickle saved by `emergo run --collect-diagnostics -o <dir>`",
    ),
    output_dir: Path = typer.Option(
        Path("emergo_viz_output"),
        "--output-dir",
        "-o",
        help="Directory to write PNG plots",
    ),
    agents: int = typer.Option(
        5,
        "--agents",
        "-n",
        envvar="EMERGO_AGENTS",
        help="Agents for fresh run (ignored with --input)",
    ),
    iterations: int = typer.Option(
        300,
        "--iterations",
        envvar="EMERGO_ITERATIONS",
        help="Iterations for fresh run (ignored with --input)",
    ),
    seed: int = typer.Option(0, "--seed", envvar="EMERGO_SEED"),
) -> None:
    """Generate visualizations from saved diagnostics or a fresh kernel run.

    Without [bold]--input[/bold], runs a quick kernel pass to collect data.
    """
    try:
        from emergo.visualize import render_health_dashboard
    except ImportError as err:
        console.print("[red]Error: viz extra not installed. Run: pip install 'emergo[viz]'[/red]")
        raise typer.Exit(code=1) from err

    if input_file is not None:
        console.print(f"[cyan]Loading diagnostics from {input_file}[/cyan]")
        with open(input_file, "rb") as fh:
            payload = pickle.load(fh)
        if not (isinstance(payload, tuple) and len(payload) == 3):
            console.print(
                "[red]Unrecognised format. Regenerate with: "
                "emergo run --collect-diagnostics -o <dir>[/red]"
            )
            raise typer.Exit(code=1)
        final_state, _reason, diag = payload
    else:
        console.print(
            f"[cyan]No --input; running fresh {agents}-agent kernel "
            f"for {iterations} iterations …[/cyan]"
        )
        from emergo import emergo_kernel

        G0, phi0, A0 = _build_ring(agents)
        with _make_progress(iterations) as prog:
            tid = prog.add_task("[cyan]Collecting data[/cyan]", total=iterations)
            final_state, _reason, diag = emergo_kernel(  # type: ignore[misc]
                initial_state=(G0, phi0, A0, []),
                max_iterations=iterations,
                collect_diagnostics=True,
                observers=[_RichProgressObserver(prog, tid, iterations)],
                rng=np.random.default_rng(seed),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Rendering → {output_dir}[/cyan]")
    saved = render_health_dashboard(diag, final_state, output_dir=str(output_dir))
    for p in saved:
        console.print(f"[green]  {p}[/green]")
    console.print(f"[bold green]{len(saved)} plot(s) saved.[/bold green]")


# ---------------------------------------------------------------------------
# emergo health
# ---------------------------------------------------------------------------


@app.command()
def health(
    agents: int = typer.Option(
        5, "--agents", "-n", envvar="EMERGO_AGENTS", help="Number of agents"
    ),
    iterations: int = typer.Option(
        200, "--iterations", "-i", envvar="EMERGO_ITERATIONS", help="Max iterations"
    ),
    seed: int = typer.Option(0, "--seed", envvar="EMERGO_SEED"),
) -> None:
    """Run the 8-detector health check and print a rich diagnostic report."""
    from emergo import emergo_kernel, run_health_check

    G0, phi0, A0 = _build_ring(agents)

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Running health check …[/cyan]"),
        console=console,
        transient=True,
    ) as prog:
        prog.add_task("health", total=None)
        final_state, reason, diag = emergo_kernel(  # type: ignore[misc]
            initial_state=(G0, phi0, A0, []),
            max_iterations=iterations,
            collect_diagnostics=True,
            rng=np.random.default_rng(seed),
        )

    console.print(f"[bold]Termination:[/bold] {reason}\n")

    results = run_health_check(diag, final_state)

    table = Table(title="Health Check — 8 Detectors", header_style="bold magenta")
    table.add_column("Detector", style="cyan", min_width=30)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Severity", justify="right", min_width=8)
    table.add_column("Evidence")

    for r in results:
        status = "[red]FAIL[/red]" if r.failure_detected else "[green] OK [/green]"
        sev = f"{r.severity:.2f}"
        if r.severity > 0.7:
            sev = f"[red]{sev}[/red]"
        elif r.severity > 0.3:
            sev = f"[yellow]{sev}[/yellow]"
        else:
            sev = f"[green]{sev}[/green]"
        evidence = (r.evidence[:68] + "…") if len(r.evidence) > 70 else r.evidence
        table.add_row(r.name, status, sev, evidence)

    console.print(table)

    failures = [r for r in results if r.failure_detected]
    if failures:
        body = "\n".join(f"[bold]{r.name}:[/bold] {r.recommendation}" for r in failures)
        console.print(Panel(body, title="[red]Recommendations[/red]", border_style="red"))
    else:
        console.print(
            Panel(
                "[green]All 8 detectors passed — system is healthy.[/green]",
                border_style="green",
            )
        )


# ---------------------------------------------------------------------------
# emergo demo convergence
# ---------------------------------------------------------------------------


@demo_app.command("convergence")
def demo_convergence(
    agents: int = typer.Option(5, "--agents", "-n", help="Number of agents"),
    iterations: int = typer.Option(500, "--iterations", "-i", help="Max iterations"),
    seed: int = typer.Option(0, "--seed"),
    optimizer: str = typer.Option("sgd", "--optimizer", help="sgd or adam"),
) -> None:
    """Classic convergence: watch φ loss and authority evolve on a ring graph."""
    from emergo import HistoryObserver, emergo_kernel

    console.print(
        Panel(
            f"[bold]Convergence Demo[/bold]\n"
            f"{agents} agents · {iterations} max iterations · "
            f"optimizer=[cyan]{optimizer}[/cyan]",
            border_style="cyan",
        )
    )
    G0, phi0, A0 = _build_ring(agents)
    obs = HistoryObserver()

    with _make_progress(iterations) as prog:
        tid = prog.add_task("[cyan]Converging[/cyan]", total=iterations)
        final_state, reason = emergo_kernel(  # type: ignore[misc]
            initial_state=(G0, phi0, A0, []),
            max_iterations=iterations,
            phi_optimizer=optimizer,
            observers=[obs, _RichProgressObserver(prog, tid, iterations)],
            rng=np.random.default_rng(seed),
        )

    _, _, A_final, _ = final_state
    console.print(_summary_panel(reason, agents, iterations, obs))
    console.print(_authority_table(A_final))


# ---------------------------------------------------------------------------
# emergo demo multi-agent
# ---------------------------------------------------------------------------


@demo_app.command("multi-agent")
def demo_multi_agent(
    agents: int = typer.Option(6, "--agents", "-n", help="Number of agents"),
    rounds: int = typer.Option(4, "--rounds", "-r", help="Coordination rounds"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Multi-agent coordination: agents submit CE proposals serialized by Lux in authority order."""
    from emergo import MultiAgentCoordinator, emergo_kernel
    from emergo.coordinator import ProposedCE
    from emergo.types import CoordinationEvent

    console.print(
        Panel(
            f"[bold]Multi-Agent Coordination Demo[/bold]\n"
            f"{agents} agents · {rounds} coordination rounds",
            border_style="magenta",
        )
    )

    G0, phi0, A0 = _build_ring(agents)

    # Brief warm-up to build a non-uniform authority distribution before coordinating
    final_state, _ = emergo_kernel(  # type: ignore[misc]
        initial_state=(G0, phi0, A0, []),
        max_iterations=50,
        rng=np.random.default_rng(seed),
    )

    rng = np.random.default_rng(seed + 1)
    coordinator = MultiAgentCoordinator()

    table = Table(title="Coordination Rounds", header_style="bold magenta")
    table.add_column("Round", justify="right", style="dim")
    table.add_column("Top proposer", style="cyan")
    table.add_column("CE type")
    table.add_column("Accepted", justify="center")
    table.add_column("Conflicts", justify="right")

    for round_num in range(1, rounds + 1):
        G_curr, _, _, _ = final_state
        ids_list = list(G_curr.agent_ids)
        rng.shuffle(ids_list)

        proposals: list[ProposedCE] = []
        n_proposers = min(len(ids_list), 3)
        for k in range(n_proposers):
            proposer = ids_list[k]
            target = ids_list[(k + 1) % len(ids_list)]
            ce = CoordinationEvent(
                event_type="add_edge",
                participants=(proposer, target),
                params=frozenset([("weight", 0.4)]),
            )
            proposals.append(ProposedCE(agent_id=proposer, ce=ce))

        result = coordinator.coordinate(proposals, final_state)
        final_state = result.final_state

        top = result.accepted[0].agent_id if result.accepted else "—"
        ce_type = result.accepted[0].ce.event_type if result.accepted else "—"
        acc_label = (
            f"[green]{len(result.accepted)} ✓[/green]" if result.accepted else "[red]0 ✗[/red]"
        )
        table.add_row(str(round_num), top, ce_type, acc_label, str(result.n_conflicts_detected))

    console.print(table)
    _, _, A_final, _ = final_state
    console.print(_authority_table(A_final, title="Authority After Coordination"))


# ---------------------------------------------------------------------------
# emergo demo executor
# ---------------------------------------------------------------------------


@demo_app.command("executor")
def demo_executor(
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Task execution: DependencyPlanner + Executor running a 4-step pipeline Goal."""
    from emergo import DependencyPlanner, Executor, Goal, Lux, emergo_kernel, mock_task_runner

    console.print(Panel("[bold]Executor + DependencyPlanner Demo[/bold]", border_style="blue"))

    steps = [
        ("fetch", "Retrieve source documents", []),
        ("extract", "Extract key entities", ["fetch"]),
        ("summarize", "Write summary draft", ["extract"]),
        ("validate", "Validate draft for accuracy", ["summarize", "extract"]),
    ]
    planner = DependencyPlanner(steps=steps)
    goal = Goal(
        goal_id="pipeline",
        description="Document analysis pipeline",
        required_capability="compute",
        initiating_agent="agent_0",
        resource_budget=20.0,
        max_depth=5,
    )

    G0, phi0, A0 = _build_ring(5)
    final_state, _ = emergo_kernel(  # type: ignore[misc]
        initial_state=(G0, phi0, A0, []),
        max_iterations=100,
        rng=np.random.default_rng(seed),
    )

    # Grant the required capability so Lux authorizes execute_task CEs
    lux = Lux()
    lux.grant_capability("agent_0", "compute")

    executor = Executor(lux=lux, planner=planner, task_runner=mock_task_runner)
    result = executor.execute(goal=goal, state=final_state)

    table = Table(title=f"Goal: [cyan]{goal.description}[/cyan]", header_style="bold")
    table.add_column("Step", style="cyan")
    table.add_column("Description")
    table.add_column("Outcome", justify="center")

    icon = "[green]✓ success[/green]" if result.success else "[red]✗ failed[/red]"
    for name, desc, _ in steps:
        table.add_row(name, desc, icon)

    console.print(table)
    status = "[green]SUCCESS[/green]" if result.success else "[red]FAILED[/red]"
    console.print(
        f"[bold]Goal:[/bold] {status}  |  "
        f"tasks {result.tasks_succeeded}/{result.tasks_attempted}  |  "
        f"resources {result.resources_spent:.1f}/{goal.resource_budget:.1f}"
    )


# ---------------------------------------------------------------------------
# emergo demo stress
# ---------------------------------------------------------------------------


@demo_app.command("stress")
def demo_stress(
    agents: int = typer.Option(20, "--agents", "-n", help="Number of agents"),
    iterations: int = typer.Option(500, "--iterations", "-i", help="Max iterations"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    """Stress test: large graph, default CE proposals, 8-detector failure-mode report."""
    from emergo import HistoryObserver, emergo_kernel, run_health_check

    console.print(
        Panel(
            f"[bold red]Stress Test Demo[/bold red]\n" f"{agents} agents · {iterations} iterations",
            border_style="red",
        )
    )

    G0, phi0, A0 = _build_ring(agents)
    obs = HistoryObserver()

    with _make_progress(iterations) as prog:
        tid = prog.add_task("[red]Stress testing[/red]", total=iterations)
        final_state, reason, diag = emergo_kernel(  # type: ignore[misc]
            initial_state=(G0, phi0, A0, []),
            max_iterations=iterations,
            observers=[obs, _RichProgressObserver(prog, tid, iterations)],
            collect_diagnostics=True,
            rng=np.random.default_rng(seed),
        )

    _, _, A_final, _ = final_state
    console.print(_summary_panel(reason, agents, iterations, obs))

    results = run_health_check(diag, final_state)
    failures = [r for r in results if r.failure_detected]

    if failures:
        ftable = Table(title="[red]Failure Modes Detected[/red]", header_style="bold red")
        ftable.add_column("Detector", style="cyan")
        ftable.add_column("Severity", justify="right")
        ftable.add_column("Recommendation")
        for r in failures:
            sev = (
                f"[red]{r.severity:.2f}[/red]"
                if r.severity > 0.5
                else f"[yellow]{r.severity:.2f}[/yellow]"
            )
            ftable.add_row(r.name, sev, r.recommendation[:60])
        console.print(ftable)
    else:
        console.print(
            Panel(
                "[green]No failure modes detected — system is resilient.[/green]",
                border_style="green",
            )
        )

    console.print(_authority_table(A_final))


# ---------------------------------------------------------------------------
# Backwards-compatible entry points (emergo-demo / emergo-kernel / emergo-health)
# These wrappers preserve the old CLI interface after the Typer upgrade.
# ---------------------------------------------------------------------------


def _legacy_demo() -> None:
    """emergo-demo: maps to `emergo demo convergence --agents 5 --iterations 500`."""
    sys.argv = ["emergo", "demo", "convergence", "--agents", "5", "--iterations", "500"]
    app()


def _legacy_kernel() -> None:
    """emergo-kernel [flags]: maps to `emergo run [flags]`."""
    sys.argv = ["emergo", "run", *sys.argv[1:]]
    app()


def _legacy_health() -> None:
    """emergo-health [flags]: maps to `emergo health [flags]`."""
    sys.argv = ["emergo", "health", *sys.argv[1:]]
    app()


# ---------------------------------------------------------------------------
# Entry point for `python -m emergo`
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
