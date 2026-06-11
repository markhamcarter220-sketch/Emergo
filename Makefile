# Emergo Makefile
#
# make bench       — full benchmark (all 5 tasks, 3 seeds; ~15 min)
# make bench-quick — fast check (T1+T2, 2 seeds; ~2 min)
# make bench-plot  — generate plots from bench_results.json
# make test        — run pytest

.PHONY: bench bench-quick bench-plot bench-all test

# Full reproducible benchmark — writes RESULTS.md and bench_results.json
bench:
	python tests/benchmarks/bench_tasks.py --json

# Quick sanity check (2 tasks, 2 seeds)
bench-quick:
	python tests/benchmarks/bench_tasks.py --quick --json

# Generate plots from previously-run bench_results.json
bench-plot:
	python tests/benchmarks/plot_results.py

# Full run + plots in one shot
bench-all: bench bench-plot

# Run test suite
test:
	python -m pytest tests/ -q
