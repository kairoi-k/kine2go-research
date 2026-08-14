# Contributing

Contributions are welcome. This repository is also a research record, so changes that affect experimental meaning require stricter provenance than ordinary code maintenance.

## Development

- Use Python 3.12 and the committed `uv.lock` where practical.
- Run the same CPU-safe checks as below before opening a PR:

  ```bash
  uvx ruff@0.12.7 check motion_imitation/amp.py motion_imitation/amp_imitation.py data/extract_disc_data_c.py tests tools
  python -m compileall -q motion_imitation go2_genesis motion_retargeting evaluation tests tools data
  python tools/check_repo_hygiene.py
  python -m unittest discover -s tests -p 'test_*.py'
  ```

- Keep machine-specific absolute paths, credentials, generated checkpoints, and unreviewed bulk artifacts out of commits.
- Prefer small commits with one semantic purpose.
- Treat GPU/Genesis validation separately from the CPU-safe layer; record the exact environment and assets used when a change needs integration testing.

## Research-semantic changes

A change is research-semantic if it can alter training behavior, evaluation meaning, reference/data interpretation, acceptance criteria, or a scientific claim.

For such changes:

1. State the intended semantic change in the PR.
2. Preserve the old protocol/evaluator when it has already produced cited results.
3. Version a semantically changed frozen evaluator instead of editing it in place.
4. Do not modify preregistered gates after observing the result; create a new protocol if the question changes.
5. Identify results or claims invalidated by the change and rerun only what is necessary.
6. Preserve failed experiments when they are part of the evidence trail.

## Tests

Run the CPU-safe checks in the README before opening a PR. Full Genesis simulation/training is environment-specific.

The frozen `evaluation/quant_eval_v5.py` is a provenance artifact. Do not edit it to satisfy style or portability checks; semantic fixes require a new evaluator version and a new result identity.

## Documentation

Update `RESEARCH_MANIFEST.md` or `docs/RESEARCH_INDEX.md` when a change alters the canonical research state. Historical evidence should normally be retained or explicitly superseded rather than rewritten to erase earlier decisions.
