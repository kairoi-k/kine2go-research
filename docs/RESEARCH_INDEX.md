# Research index

This page maps claims to committed artifacts. Two evidence threads are recorded here. They do not substitute for each other.

## 1. Frozen conditional-AMP baseline

Frozen 2026-08-13.

| Purpose | Canonical artifact |
|---|---|
| Compact research state | `RESEARCH_MANIFEST.md` |
| Preregistered protocol and final verdict | `docs/clean_run_preregistration.md` |
| Implementation-correctness audit | `docs/IMPLEMENTATION_AUDIT.md` |
| Frozen evaluator | `evaluation/quant_eval_v5.py` (`v5-frozen-20260813`) |
| Portable launcher for new runs | `evaluation/run_quant_eval_v5.py` |
| Final v5 evaluation record | `eval_results/amp_clean_run_5000_v5.json` |
| Evaluated policy identity | SHA-256 prefix `4dfb111d165275f9` |
| Environment / rerun boundary | `docs/REPRODUCIBILITY.md` |
| Fork and development provenance | `docs/PROVENANCE.md` |
| Conditioned discriminator data | `data/README.md` |

### Supported conclusion

The tested single-reference conditional-AMP configuration is a **negative result**:

- preregistered endpoint speed span: **0.009 m/s**, required ≥ 0.4 m/s;
- command-1.5 repeats: **3/5 fell**, required zero falls;
- one additional sweep trajectory at command 1.0 m/s was incomplete because of a fall.

Reliable command-conditioned locomotion was not established across the evaluated range. This does not show that AMP generally cannot produce controllable locomotion.

## 2. Reference-seam intervention

Same starting policy (`v3@17997`), three paired seeds, +3000 iterations, original vs repaired cycle seam. Evaluator for these files is historical `quant_eval_v2` (not shipped). The committed JSON files are the record.

| Seed | Original RMSE | Repaired RMSE | RMSE change |
|---|---|---|---|
| seed1 | 0.273 | 0.193 | -29% |
| seed2 | 0.275 | 0.211 | -23% |
| seed3 | 0.309 | 0.183 | -41% |
| aggregate | 0.286 | 0.196 | -31% |

Artifacts: `eval_results/seam_orig_s1_v2.json`, `orig_s2_v2.json`, `orig_s3_v2.json`, `fixed_s1_v2.json`, `fixed_s2_v2.json`, `fixed_s3_v2.json`. Recompute the table with `python tools/summarize_seam_results.py`.

**Supported claim:** repairing the tested reference seam improved command-tracking RMSE under this intervention, same direction on all three seeds.

**Outside the claim:** this is not command-conditioned speed controllability. Both arms still run near ~1.4 m/s. Aggregate dip (1.00 vs 1.17) is a secondary descriptor; per-seed dip is mixed. The AMP negative result does not negate this intervention, and this intervention does not rescue AMP controllability.

## Evaluator semantics

`evaluation/quant_eval_v5.py` is frozen because the AMP result records its SHA-256 prefix (`7d6f6d4a1649f5ab`). Do not edit it; use `evaluation/run_quant_eval_v5.py` for new runs. Semantic fixes belong in a new evaluator version.

Seam JSON files must not be compared to v5 AMP numbers without accounting for the protocol change.

## Next study

A controlled single-reference vs multi-reference experiment is planned after measuring released Kine2Go motions. It is not part of this baseline.
