# Reproducibility

## Scope

This repository distinguishes three levels of reproducibility:

1. **Static reproducibility** — a fresh clone can run syntax and repository-hygiene checks.
2. **Utility-level reproducibility** — CPU-safe conditional-AMP regressions can run without Genesis simulation.
3. **Full computational reproduction** — requires a Genesis-compatible GPU environment plus the motion/checkpoint assets used by the experiment.

The third level cannot be achieved from Git alone because the evaluated checkpoint is not distributed in the repository.

To run the frozen v5 evaluator from this checkout, use `python evaluation/run_quant_eval_v5.py` rather than editing `quant_eval_v5.py`. The launcher keeps the recorded file hash intact.

## Frozen software environment

The clean baseline targets:

- Python >= 3.12
- PyTorch 2.10.0
- Genesis 0.3.10

Install the committed environment with:

```bash
uv sync --frozen
```

## Static and CPU-safe checks

```bash
uvx ruff@0.12.7 check motion_imitation/amp.py motion_imitation/amp_imitation.py data/extract_disc_data_c.py tests tools
python -m compileall -q motion_imitation go2_genesis motion_retargeting evaluation tests tools data
python tools/check_repo_hygiene.py
python -m unittest discover -s tests -p 'test_*.py'
```

The hosted `Code Quality` workflow runs repository-wide Ruff lint and format. Hygiene and the unit tests above are the local research gate, not a hosted-runner pass.

## Frozen clean-run evidence

The public release captures development milestone `0ad4343` from 2026-08-13. That identifier belongs to the development history; a clean-history public snapshot may have different Git commit IDs.

Canonical evidence:

- protocol and verdict: `docs/clean_run_preregistration.md`
- implementation audit: `docs/IMPLEMENTATION_AUDIT.md`
- evaluator: `evaluation/quant_eval_v5.py`
- evaluator version: `v5-frozen-20260813`
- evaluator SHA-256 prefix: `7d6f6d4a1649f5ab`
- result: `eval_results/amp_clean_run_5000_v5.json`
- evaluated policy: local `logs/amp_clean_run/model_5000.pt`
- policy SHA-256 prefix: `4dfb111d165275f9`
- training revision recorded by the result: `c64922265fa8cd92529b66b6aaaa52f16fe8eefd`
- motion SHA-256 prefix recorded by the result: `d1367f511b9d1402`
- configuration hash recorded by the result: `a31836d7ae2aaf95`

`evaluation/quant_eval_v5.py` is preserved byte-for-byte because its own hash is part of the result provenance. Known diagnostic limitations are documented rather than silently repaired in place.

## Data

`data/disc_data_c6.npy` is the canonical conditioned discriminator dataset retained for the frozen baseline. Older intermediate discriminator arrays are intentionally omitted from the curated release.

The dataset schema and derivation are documented in `data/README.md`; `data/extract_disc_data_c.py` is the maintained generator for new conditioned datasets.

## Reproducing a new experiment

For every new scientific run, record at minimum:

1. repository revision;
2. complete training configuration and random seed;
3. reference/data version or content hash;
4. starting policy and discriminator state, if applicable;
5. protocol and evaluator version;
6. final checkpoint hash;
7. raw result JSON, including termination and fall status.

Do not reuse `v5-frozen-20260813` for changed evaluator semantics. New semantics require a new evaluator version and new result artifact.

The three-seed seam record is `eval_results/*_v2.json`. Those files use the historical v2 evaluator (not shipped). Print the published RMSE table with `python tools/summarize_seam_results.py`. Do not mix v2 RMSE with the v5 AMP span/fall verdict.

## Interpretation boundary

The frozen experiment is a negative result for one single-reference conditional-AMP configuration. It must not be generalized to AMP as a method class or to a future multi-reference configuration.
