# Implementation audit

This document summarizes correctness issues identified during the conditional-AMP implementation review and the controls added before the frozen clean run. It replaces model-specific review transcripts with a neutral technical record.

The audit establishes what was checked and corrected; it does not claim that the implementation is free of undiscovered defects.

## State-feature semantics

The discriminator requires real motion samples and policy-generated samples to have identical physical meaning column by column.

Issues identified during review included a mismatch between Genesis DOF ordering and the policy motor ordering, plus the need to make linear/angular velocity frame conventions explicit. The real-data extraction path was corrected to use the same joint order as policy features; the corrected joint permutation is:

```text
[1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
```

Joint position and velocity features are treated consistently, and the corrected conditioned discriminator dataset (`disc_data_c6.npy`) is the clean-run input.

## Episode-boundary isolation

AMP state pairs must not cross an environment reset. The reward/training path was audited so done/reset state is available at the point where transition validity is decided, and invalid cross-episode pairs are excluded rather than silently entering discriminator data or style reward computation.

## Control timestep and reward scaling

The environment control timestep is distinct from the physics substep. The clean path asserts the expected control-step semantics and applies reward scaling consistently at the control timestep rather than relying on an inferred fallback value.

## Command support and curriculum

The conditional discriminator is meaningful only where real reference data support the command condition. Review identified failure modes in curriculum contraction/expansion and in allowing command distributions that were inconsistent with discriminator support.

The clean configuration:

- uses positive forward commands for the registered run;
- fixes lateral/yaw commands to zero;
- clamps the curriculum to the physical support interval used by the conditioned real data;
- records curriculum behavior explicitly.

## Real/fake condition matching

Balancing real-data buckets alone is insufficient if the discriminator can distinguish real and fake samples from their command marginal. The clean path matches real sampling to the command/speed bins represented by policy samples so the discriminator is trained on the intended conditional comparison rather than an avoidable `p(real c)` vs `p(fake c)` shortcut.

## Checkpoint provenance

Policy and discriminator state form a logical pair for a conditional-AMP run. The checkpoint path was audited to preserve paired iteration semantics, avoid destructive resume behavior, and record enough provenance to associate a final evaluation with the intended policy/discriminator/training revision.

The frozen clean-run result identifies the evaluated policy artifact by path and SHA-256 prefix; the large checkpoint itself is not distributed in Git.

## Evaluator corrections

Evaluation correctness was treated separately from training correctness. Earlier development evaluators produced intermediate records and exposed several semantic defects; those revisions are omitted from the curated public snapshot and remain in the development archive.

Important evaluator corrections included:

- contact state is thresholded as a boolean contact event rather than treating raw force magnitude as a boolean;
- early/short terminated trajectories are handled without producing invalid smoothing/metric behavior;
- a sweep trajectory with any fall is incomplete for primary v5 sweep metrics;
- the preregistered speed span uses the registered endpoint pair rather than a post-hoc substitute;
- evaluator provenance hashes the actual frozen evaluator file.

The canonical evaluator for the clean negative baseline is `evaluation/quant_eval_v5.py` (`v5-frozen-20260813`).

## Known frozen-v5 limitations

The v5 source is intentionally preserved for provenance. Two known limitations remain in diagnostics that do not alter the registered speed-span/stability verdict:

- a machine-local fallback path remains in Git-metadata lookup;
- the push-recovery timing origin has a diagnostic defect.

`evaluation/run_quant_eval_v5.py` substitutes the checkout root for the development-machine prefix without rewriting this file. Use that launcher for new runs.

In addition, the external-force push test was invalid in the recorded environment because the required Genesis ForceField API was unavailable. It is excluded from pass/fail.

Any semantic evaluator fix belongs in a new version rather than an in-place modification of v5.

## Audit boundary

The audit supports the statement that the known reviewed correctness defects were addressed before the clean run. It does not prove the absence of latent bugs, and it does not by itself establish the scientific mechanism behind the negative result. See [`clean_run_preregistration.md`](clean_run_preregistration.md) for the registered question and final verdict.
