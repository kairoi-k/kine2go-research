# Changelog

## Unreleased — initial public-release candidate

This curated snapshot captures the conditional-AMP research state finalized at development milestone `0ad4343`.

### Added

- speed-conditioned AMP training and command-support-aware curriculum;
- matched real/fake condition sampling and paired policy/discriminator checkpoint semantics;
- frozen v5 evaluator, portable launcher, canonical AMP result, and three-seed seam JSON record;
- preregistration, implementation audit, claim-to-evidence index, provenance, and reproducibility documentation;
- CPU-safe AMP regression tests and repository-hygiene CI.

### Curated for public release

- retained the canonical conditioned discriminator dataset (`disc_data_c6.npy`);
- retained the exact frozen v5 evaluator and canonical v5 result artifact;
- Ruff lint/format pass on the public tree; `quant_eval_v5.py` is excluded so its recorded hash stays intact;
- removed obsolete evaluator/result intermediates, superseded discriminator arrays, ad-hoc experiment scripts, model-specific review transcripts, inherited website files, and private-workspace residue.

The development archive remains the complete historical record. The public candidate intentionally exposes a smaller, bounded evidence surface.
