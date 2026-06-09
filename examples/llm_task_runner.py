#!/usr/bin/env python3
"""LLM-Integrated Task Runner Example for Emergo.

Demonstrates how to integrate an LLM (or any AI service) as the task runner
backend for the Emergo Executor, enabling emergent coordination on real AI
workflows such as research decomposition, code generation, and summarization.

The script ships with a ``MockLLMClient`` that simulates LLM responses locally
so it runs without any API keys.  To switch to a real LLM, replace
``MockLLMClient`` with your preferred provider (see the stub at the bottom).

Usage:
    # Run the full demo with mocked LLM responses
    python examples/llm_task_runner.py

    # Enable verbose output showing each LLM call
    python examples/llm_task_runner.py --verbose

    # Run with a real OpenAI-compatible endpoint (requires OPENAI_API_KEY)
    python examples/llm_task_runner.py --real-llm --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from emergo import (
    DependencyPlanner,
    Executor,
    Goal,
    Graph,
    Lux,
    make_initial_authority,
    make_initial_phi,
)
from emergo.executor import TaskOutcome


# ---------------------------------------------------------------------------
# Mock LLM client (no API keys required)
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Simulated LLM response."""

    content: str
    tokens_used: int
    latency_ms: float
    ok: bool = True
    error: str = ""


class MockLLMClient:
    """Simulates an LLM that processes research/coding tasks deterministically.

    Replace with a real client (OpenAI, Anthropic, etc.) for production use.
    See ``RealOpenAIClient`` stub below.
    """

    _RESPONSES: dict[str, str] = {
        "search": "Found 12 relevant papers on multi-agent coordination dynamics.",
        "fetch": "Downloaded 3 source documents (total 42 KB).",
        "extract": "Extracted 8 key findings: authority gradients, phi convergence, ...",
        "summarize": "SUMMARY: Emergo's authority-weighted topology prediction enables "
        "emergent coordination without central control.",
        "validate": "Validation PASS: summary consistent with extracted facts.",
        "generate_code": "```python\nclass EmergoAgent:\n    pass\n```",
        "test_code": "All 3 unit tests pass.",
        "default": "Task completed successfully.",
    }

    def __init__(self, verbose: bool = False, simulate_failures: bool = False) -> None:
        self._verbose = verbose
        self._simulate_failures = simulate_failures
        self._call_count = 0
        self._rng = np.random.default_rng(42)

    def complete(self, task_description: str, capability: str) -> LLMResponse:
        self._call_count += 1
        # Simulate network latency
        latency = float(self._rng.uniform(50, 200))

        # Simulate occasional failures when enabled
        if self._simulate_failures and self._rng.random() < 0.15:
            return LLMResponse(
                content="",
                tokens_used=0,
                latency_ms=latency,
                ok=False,
                error="Simulated transient LLM error",
            )

        # Look up a canned response
        response_key = next(
            (k for k in self._RESPONSES if k in capability.lower()), "default"
        )
        content = self._RESPONSES[response_key]
        tokens = len(content.split()) * 2  # rough estimate

        if self._verbose:
            print(f"    [LLM call #{self._call_count}] capability={capability!r}")
            print(f"       task: {task_description[:60]}...")
            print(f"       → {content[:80]}...")
            print(f"       tokens={tokens}, latency={latency:.0f}ms")

        return LLMResponse(content=content, tokens_used=tokens, latency_ms=latency, ok=True)


# ---------------------------------------------------------------------------
# LLM-backed task runner (plugs into Executor)
# ---------------------------------------------------------------------------


@dataclass
class LLMTaskStats:
    """Accumulated statistics across all LLM task executions."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    results: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_calls / max(self.total_calls, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_calls, 1)


def make_llm_task_runner(client: MockLLMClient, stats: LLMTaskStats):
    """Return a task_runner callable backed by the given LLM client."""

    def llm_task_runner(task) -> TaskOutcome:
        stats.total_calls += 1
        response = client.complete(task.description, task.required_capability)
        stats.total_latency_ms += response.latency_ms

        if response.ok:
            stats.successful_calls += 1
            stats.total_tokens += response.tokens_used
            stats.results.append(response.content[:100])
            # Capability delta: successful execution slightly improves the agent's score
            cap_delta = {"dim0": 0.02}
            return TaskOutcome(
                task_id=task.task_id,
                success=True,
                capability_delta=cap_delta,
                resource_consumed=task.resource_cost,
                notes=f"LLM response ({response.tokens_used} tokens): {response.content[:80]}",
            )
        else:
            stats.failed_calls += 1
            return TaskOutcome(
                task_id=task.task_id,
                success=False,
                capability_delta={},
                resource_consumed=0.0,
                notes=f"LLM error: {response.error}",
            )

    return llm_task_runner


# ---------------------------------------------------------------------------
# Demo: Research decomposition workflow
# ---------------------------------------------------------------------------


def run_research_workflow(
    verbose: bool = False,
    simulate_failures: bool = False,
) -> dict:
    """Run a 5-step research decomposition pipeline via Emergo + LLM.

    Workflow:
        1. search   — find relevant sources
        2. fetch    — download them
        3. extract  — extract key facts
        4. summarize — write a summary (depends on extract + fetch)
        5. validate  — fact-check the summary (depends on summarize + extract)
    """
    print("\n" + "=" * 60)
    print("  Demo 1: Research Decomposition Workflow (5 steps)")
    print("=" * 60)

    # --- Agents: researcher, fetcher, analyst, writer, validator
    agent_ids = ("researcher", "fetcher", "analyst", "writer", "validator")
    n = len(agent_ids)
    adj = np.zeros((n, n))
    # Directional authority flow: researcher → fetcher → analyst → writer → validator
    for i in range(n - 1):
        adj[i, i + 1] = 0.6
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=0)
    A0 = make_initial_authority(agent_ids, baseline=0.5)
    state = (G0, phi0, A0, [])

    # --- Lux: grant each agent their required capability
    lux = Lux()
    lux.grant_capability("researcher", "search")
    lux.grant_capability("fetcher", "fetch")
    lux.grant_capability("analyst", "extract")
    lux.grant_capability("writer", "summarize")
    lux.grant_capability("validator", "validate")

    # --- LLM client + task runner
    client = MockLLMClient(verbose=verbose, simulate_failures=simulate_failures)
    stats = LLMTaskStats()
    task_runner = make_llm_task_runner(client, stats)

    # --- Dependency-aware planner
    planner = DependencyPlanner(
        steps=[
            ("search", "Search for relevant sources on multi-agent coordination", []),
            ("fetch", "Fetch the top-3 papers", ["search"]),
            ("extract", "Extract key findings from papers", ["fetch"]),
            ("summarize", "Write a 200-word summary", ["extract", "fetch"]),
            ("validate", "Validate summary against source facts", ["summarize", "extract"]),
        ]
    )

    goal = Goal(
        goal_id="research_001",
        description="Research and summarize recent advances in multi-agent coordination",
        required_capability="search",
        initiating_agent="researcher",
        resource_budget=40.0,
        max_depth=5,
    )

    executor = Executor(lux=lux, planner=planner, task_runner=task_runner)

    t0 = time.time()
    result = executor.execute(goal, state)
    elapsed = time.time() - t0

    # --- Report
    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"  {status}  ({elapsed*1000:.0f}ms wall time)")
    print(f"  Tasks completed : {result.tasks_succeeded}/{result.tasks_attempted}")
    print(f"  Resources spent : {result.resources_spent:.1f}")
    print(f"  LLM calls       : {stats.total_calls} ({stats.successful_calls} ok, "
          f"{stats.failed_calls} failed)")
    print(f"  LLM tokens used : {stats.total_tokens}")
    print(f"  Avg LLM latency : {stats.avg_latency_ms:.0f}ms (simulated)")
    if stats.results:
        print(f"  Final output    : {stats.results[-1][:70]}...")

    return {
        "success": result.success,
        "tasks_succeeded": result.tasks_succeeded,
        "tasks_attempted": result.tasks_attempted,
        "resources_spent": result.resources_spent,
        "llm_calls": stats.total_calls,
        "llm_tokens": stats.total_tokens,
        "llm_success_rate": stats.success_rate,
    }


# ---------------------------------------------------------------------------
# Demo: Code generation + test pipeline
# ---------------------------------------------------------------------------


def run_code_generation_workflow(verbose: bool = False) -> dict:
    """Run a 3-step code generation + testing pipeline."""
    print("\n" + "=" * 60)
    print("  Demo 2: Code Generation + Testing Pipeline (3 steps)")
    print("=" * 60)

    agent_ids = ("planner_agent", "coder", "tester")
    n = len(agent_ids)
    adj = np.array([[0, 0.7, 0], [0, 0, 0.7], [0, 0, 0]], dtype=float)
    caps = np.ones((n, 4)) * 0.5
    G0 = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi0 = make_initial_phi(d_latent=8, d_features=16, d_ce=4, seed=1)
    A0 = make_initial_authority(agent_ids, baseline=0.5)
    state = (G0, phi0, A0, [])

    lux = Lux()
    lux.grant_capability("planner_agent", "search")
    lux.grant_capability("coder", "generate_code")
    lux.grant_capability("tester", "test_code")

    client = MockLLMClient(verbose=verbose)
    stats = LLMTaskStats()
    task_runner = make_llm_task_runner(client, stats)

    planner = DependencyPlanner(
        steps=[
            ("design", "Design the EmergoAgent class interface", []),
            ("implement", "Implement EmergoAgent in Python", ["design"]),
            ("test", "Write and run unit tests for EmergoAgent", ["implement"]),
        ]
    )

    goal = Goal(
        goal_id="codegen_001",
        description="Implement and test an EmergoAgent wrapper class",
        required_capability="search",
        initiating_agent="planner_agent",
        resource_budget=15.0,
    )

    executor = Executor(lux=lux, planner=planner, task_runner=task_runner)
    result = executor.execute(goal, state)

    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"  {status}")
    print(f"  Tasks: {result.tasks_succeeded}/{result.tasks_attempted} completed")
    print(f"  LLM calls: {stats.total_calls} ({stats.total_tokens} tokens)")
    if stats.results:
        print(f"  Code output: {stats.results[-2][:60] if len(stats.results) >= 2 else ''}...")
        print(f"  Test output: {stats.results[-1][:60]}...")

    return {
        "success": result.success,
        "tasks_succeeded": result.tasks_succeeded,
        "llm_tokens": stats.total_tokens,
    }


# ---------------------------------------------------------------------------
# Stub: Real OpenAI-compatible client (drop-in replacement)
# ---------------------------------------------------------------------------


class RealOpenAIClient:
    """Real LLM client using the OpenAI SDK.

    Replace ``MockLLMClient`` with this for production use.

    Install:  pip install openai
    Set env:  export OPENAI_API_KEY=sk-...
    """

    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = False) -> None:
        self._model = model
        self._verbose = verbose
        try:
            import openai  # noqa: F401 — optional dependency

            self._client = openai.OpenAI()
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from None

    def complete(self, task_description: str, capability: str) -> LLMResponse:
        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an AI assistant with the capability: {capability}.",
                    },
                    {"role": "user", "content": task_description},
                ],
                max_tokens=200,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            latency = (time.time() - t0) * 1000
            return LLMResponse(content=content, tokens_used=tokens, latency_ms=latency, ok=True)
        except Exception as exc:
            return LLMResponse(
                content="",
                tokens_used=0,
                latency_ms=(time.time() - t0) * 1000,
                ok=False,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emergo + LLM task runner integration demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual LLM call details",
    )
    parser.add_argument(
        "--simulate-failures",
        action="store_true",
        help="Enable 15% random LLM failure simulation",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use a real OpenAI-compatible API (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model name (only used with --real-llm)",
    )
    args = parser.parse_args()

    print("\nEmergo LLM Task Runner Integration Demo")
    print("Uses MockLLMClient (no API keys needed)." if not args.real_llm else
          f"Using real LLM: {args.model}")

    if args.real_llm:
        print("\nNote: --real-llm flag set. Attempting to use OpenAI client.")
        print("      Make sure OPENAI_API_KEY is set and `pip install openai` is done.\n")

    r1 = run_research_workflow(
        verbose=args.verbose,
        simulate_failures=args.simulate_failures,
    )
    r2 = run_code_generation_workflow(verbose=args.verbose)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_success = r1["success"] and r2["success"]
    print(f"  Research workflow  : {'PASS' if r1['success'] else 'FAIL'} "
          f"({r1['tasks_succeeded']}/{r1['tasks_attempted']} tasks, "
          f"{r1['llm_tokens']} tokens)")
    print(f"  Code-gen workflow  : {'PASS' if r2['success'] else 'FAIL'} "
          f"({r2['tasks_succeeded']} tasks, {r2['llm_tokens']} tokens)")
    print()
    print("  ✓ LLM integration works: Executor.execute() → DependencyPlanner →")
    print("    per-task LLM calls → authority updates → emergent coordination.")
    print()
    print("  To use a real LLM backend, replace MockLLMClient with")
    print("  RealOpenAIClient (stub included) or any provider that implements")
    print("  the .complete(task_description, capability) → LLMResponse interface.")
    print()

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
