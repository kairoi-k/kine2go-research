# Evaluation

`quant_eval_v5.py` is the frozen evaluator for the single-reference conditional-AMP baseline.

- evaluator version: `v5-frozen-20260813`
- recorded SHA-256 prefix: `7d6f6d4a1649f5ab`
- canonical result: `../eval_results/amp_clean_run_5000_v5.json`

The file is preserved byte-for-byte because its own content hash is part of the experiment provenance. Known diagnostic limitations are documented in `../docs/IMPLEMENTATION_AUDIT.md`; they must not be silently repaired in this file.

For a new evaluation from this checkout, use `python evaluation/run_quant_eval_v5.py ...`. The launcher substitutes the checkout root for the development-machine prefix without changing `quant_eval_v5.py`.

Any future semantic change requires a new versioned evaluator and a new result artifact.
