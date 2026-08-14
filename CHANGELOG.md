# Changelog

## 2026-08-13

Frozen conditional-AMP research state at development milestone `0ad4343`.

### Added

- speed-conditioned AMP training and command-support-aware curriculum;
- matched real/fake condition sampling and paired policy/discriminator checkpoint semantics;
- frozen v5 evaluator, portable launcher, canonical AMP result, and three-seed seam JSON record;
- preregistration, implementation audit, claim-to-evidence index, provenance, and reproducibility documentation;
- CPU-safe AMP regression tests and repository-hygiene checks.

### In this tree

- canonical conditioned discriminator dataset (`disc_data_c6.npy`);
- frozen v5 evaluator and canonical v5 result artifact;
- Ruff lint/format on the public tree; `quant_eval_v5.py` is excluded so its recorded hash stays intact.
