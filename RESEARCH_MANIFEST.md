# Research Manifest

Last updated: 2026-08-13

This file is the compact state of record. Historical evaluator versions stay when they are needed to interpret older evidence; current claims are anchored to the artifacts below.

## Upstream provenance

- Upstream project: `nomagiclab/kine2go-pipeline`
- Upstream base: `65b39f104706c1b4307dfc2a7df2aa8bed8d20aa`
- Upstream paper: Pałucki et al., *Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions* (2026)

Development milestone IDs (e.g. `0ad4343`) refer to the archive, not this Git history.

## Frozen single-reference conditional-AMP baseline

- Development milestone: `0ad4343` (2026-08-13)
- Training configuration: preregistered single-reference conditional AMP
- Evaluated policy: `model_5000.pt`
- Policy SHA-256 prefix: `4dfb111d165275f9`
- Canonical evaluator: `evaluation/quant_eval_v5.py`
- Evaluator version: `v5-frozen-20260813`
- Evaluator SHA-256 prefix: `7d6f6d4a1649f5ab`
- Canonical result: `eval_results/amp_clean_run_5000_v5.json`
- Protocol and verdict: `docs/clean_run_preregistration.md`

### Verdict

The tested configuration failed the preregistered controllability and stability criteria.

- Endpoint speed span, command 0.5 vs 2.5 m/s: 1.059 vs 1.068 m/s → **0.009 m/s**; required ≥ 0.4 m/s.
- Command-1.5 repeat evaluation: **3/5 repeats fell**; required zero falls.
- The sweep contained an additional fall at command 1.0 m/s.
- Reliable command-conditioned locomotion was not established across the evaluated range.

The conclusion is limited to this audited configuration. It does not establish that AMP generally fails. Reviewed implementation defects were corrected before the clean run, but undiscovered implementation issues cannot be excluded.

The failure mechanism remains unresolved. Candidate explanations include insufficient single-reference support across commanded speeds and task/style or discriminator-conditioning effects.

## Evidence rules

- Frozen evaluator semantics are immutable. A semantic change requires a new evaluator version.
- Preregistered thresholds are not changed after observing results.
- Under v5, a sweep trajectory containing a fall is incomplete for primary sweep metrics.
- Superseded evaluators remain historical evidence only; they do not override v5 where semantics conflict.
- Claims must identify the protocol, evaluator, and result artifact that support them.
- The checkpoint is not distributed in Git; its recorded hash identifies the evaluated local artifact.

## Next study — not part of this baseline

The next planned experiment is a controlled multi-reference pilot:

1. characterize the final Go2 body-forward speed and gait of the released Kine2Go motions;
2. select 4–6 homogeneous, approximately straight locomotion references spanning useful speeds;
3. compare the frozen single-reference baseline with a multi-reference design using preregistered evaluation semantics.

No multi-reference training is included in the frozen baseline.

## Reference-seam intervention (v2 JSON record)

Same `v3@17997` start, three paired seeds, original vs repaired cycle seam. RMSE -29% / -23% / -41% (aggregate 0.286 → 0.196, -31%). Artifacts under `eval_results/*_v2.json`. This is not command-conditioned speed controllability.

See `docs/RESEARCH_INDEX.md`, `docs/REPRODUCIBILITY.md`, and `docs/PROVENANCE.md` for the complete evidence map.
