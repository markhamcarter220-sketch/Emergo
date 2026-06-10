# Contributing to Emergo

## Quick start

```bash
git clone https://github.com/markhamcarter220-sketch/Emergo
cd Emergo
pip install -e ".[dev]"
pre-commit install
```

## Branch strategy

- `main` — stable releases only  
- `claude/emergo-kernel-impl-ZTl1L` — active development (current)  
- Feature branches: `feat/<name>` off the active dev branch

## Making changes

1. Run `python -m pytest tests/ -q` — all tests must pass before submitting.
2. Run `python -m ruff check emergo/ tests/ --fix` and `python -m black emergo/ tests/ --line-length 100`.
3. Run `python -m mypy emergo/ --ignore-missing-imports` — zero new errors.
4. Add or update tests for every changed behaviour.

## Commit style

Use conventional commits: `feat:`, `fix:`, `bench:`, `docs:`, `refactor:`, `chore:`.

## Key invariants — never break these

| ID | Name | Enforcement |
|---|---|---|
| INV-11 | Authority ≤ 0.8 | `authority_update` clips to [0, 0.8] |
| INV-12 | No self-loops | `ce_execute` rejects from==to |
| INV-17 | φ never permanently frozen | `phi_force_adapt_interval` in kernel |

## Running the benchmark

```bash
python tests/benchmarks/bench_vanilla_vs_emergo.py --quick
```

## Reporting issues

Open an issue at https://github.com/markhamcarter220-sketch/Emergo/issues
