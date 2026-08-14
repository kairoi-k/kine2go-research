# Evaluation results

Two evidence threads are committed. Do not mix v2 seam numbers with the v5 AMP result.

## Conditional AMP (frozen v5)

Canonical artifact: `amp_clean_run_5000_v5.json`.

- evaluator version: `v5-frozen-20260813`
- evaluator SHA-256 prefix: `7d6f6d4a1649f5ab`
- model SHA-256 prefix: `4dfb111d165275f9`
- training commit: `c64922265fa8cd92529b66b6aaaa52f16fe8eefd`
- configuration hash: `a31836d7ae2aaf95`
- motion SHA-256 prefix: `d1367f511b9d1402`

Verdict and claim boundary: `../docs/clean_run_preregistration.md`.

## Reference seam (historical v2)

Paired original vs repaired-reference evaluations after +3000 iterations from the same `v3@17997` start:

| Role | Files |
|---|---|
| original seam | `seam_orig_s1_v2.json`, `orig_s2_v2.json`, `orig_s3_v2.json` |
| repaired seam | `fixed_s1_v2.json`, `fixed_s2_v2.json`, `fixed_s3_v2.json` |

The v2 evaluator is not shipped. These JSON files are the record. Print the published table with `python tools/summarize_seam_results.py`.

Intermediate AMP v1–v4 JSON files remain in the development archive only.
