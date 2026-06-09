## Summary

<!-- 1–3 bullet points describing what this PR does. -->

- 

## Type of change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking addition)
- [ ] Breaking change (fixes or feature that changes existing API)
- [ ] Documentation / examples
- [ ] Tests / CI

## Checklist

<!-- All boxes must be checked before requesting review. -->

- [ ] `pytest` passes: `pytest tests/ -q` → 0 failures
- [ ] `ruff check emergo/ tests/` → 0 errors
- [ ] `mypy emergo/ --ignore-missing-imports` → 0 errors
- [ ] `black --check emergo/ tests/` → 0 files would reformat
- [ ] New public API exported in `emergo/__init__.py` and `__all__`
- [ ] Tests added for new behaviour (≥ 3 tests per new function)
- [ ] Invariant tests still pass: `pytest tests/test_invariants.py tests/test_adversarial_tier1.py`
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] No secrets or credentials committed

## Related issues

Closes #<!-- issue number -->

## Notes for reviewers

<!-- Anything the reviewer should know about design decisions, tradeoffs, or areas of concern. -->
