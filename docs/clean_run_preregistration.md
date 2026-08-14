# Conditional AMP clean run — preregistration and final verdict

Date: 2026-08-13

This document records the frozen protocol and final verdict for the first post-correctness-fix conditional-AMP run. The research question was whether the audited single-reference configuration could establish command-conditioned forward-speed control over 0.5–2.5 m/s.

## Frozen training configuration

- Starting policy: `v8b@22996` (model hash recorded with the run)
- Discriminator: cold start using corrected `disc_data_c6.npy`
- Style reward weight: 0.3
- Tracking reward weight: 2.0
- Joint-rotation imitation weight: 0.5 → 0.05, linearly decayed to its floor by iteration 3500
- Command curriculum: `(0.9, 1.1)` toward `(0.5, 2.5)` with success threshold 0.6
- Physical conditioning support clamp: `[0.3, 3.5]`
- Lateral and yaw commands fixed to zero
- 16,384 environments
- 5,000 iterations
- Training seed: 1
- Positive forward commands only
- Paired policy/discriminator checkpoints and training Git revision recorded

No acceptance criterion was to be changed after observing the run.

## Preregistered acceptance criteria

The configuration would pass only if all primary criteria passed:

| Criterion | Threshold |
|---|---:|
| Command-response span, cmd 0.5 vs 2.5 m/s | >= 0.4 m/s |
| RMSE at cmd 2.0 m/s | < 0.5 m/s |
| Falls | 0 |
| Dip metric | <= 1.0 |
| Jerk | <= 1500 |
| Blind qualitative review | >= 6/10 |

Duty factor, flight fraction, cost-of-transport proxy, and action-rate measures were diagnostic rather than pass/fail criteria.

## Preregistered stop conditions

Training would stop for any of the following:

- measured `vx` at cmd 1.0 m/s below 0.7 m/s at iteration 1000;
- discriminator loss collapsing to zero or exceeding three times its early-run level;
- any NaN/Inf gradient;
- curriculum failing to expand or contract for 300 iterations, indicating a broken record/update path.

None of these stop conditions triggered before the final checkpoint.

## Intermediate observations

At iteration 1000, evaluated forward speeds were approximately 0.97–0.99 m/s across commands and the response span was about 0.05 m/s. At iteration 2000, evaluated speeds were approximately 1.09–1.14 m/s with a similarly small span. These were recorded as intermediate diagnostics; the preregistered protocol required continuing because no stop condition had fired.

## Evaluator correction and final semantics

The evaluation code underwent correctness fixes after the protocol was written. The final result is therefore reported with the frozen `evaluation/quant_eval_v5.py` (`v5-frozen-20260813`) semantics rather than silently preserving known defects in earlier evaluator versions.

The relevant v5 rules are:

- any sweep trajectory that falls is marked incomplete for primary sweep metrics;
- the preregistered response span is computed only from the specified 0.5 and 2.5 m/s endpoints when both endpoints complete;
- evaluator provenance records the v5 evaluator itself;
- the push diagnostic is excluded from pass/fail when the required external-force API is unavailable.

Earlier v4 reporting used a different endpoint pair for one span summary; that value is superseded and is not the preregistered metric.

## Final v5 evaluation

The same `model_5000.pt` was reevaluated with v5.

### Repeat evaluation at cmd 1.5 m/s

Five repeats were run. Three fell, leaving two complete trajectories. This fails the zero-fall criterion.

### Command sweep

| Command (m/s) | Outcome | Mean forward speed / RMSE |
|---:|---|---|
| 0.5 | complete | 1.059 / 0.587 m/s |
| 1.0 | incomplete | fall at step 235 |
| 1.5 | complete | 1.072 / 0.448 m/s |
| 2.0 | complete | 1.096 / 0.930 m/s |
| 2.5 | complete | 1.068 / 1.441 m/s |

The preregistered endpoint span is **1.068 - 1.059 = 0.009 m/s**, far below the required 0.4 m/s.

The external-force push diagnostic was invalid in this environment because the Genesis ForceField API was unavailable; it is not used in the verdict.

## Verdict

**FAIL.** The tested single-reference conditional-AMP configuration did not establish reliable command-conditioned locomotion across the evaluated speed range.

Primary failures include:

- endpoint speed span: **0.009 m/s** vs required **>= 0.4 m/s**;
- cmd-2.0 RMSE: **0.930 m/s** vs required **< 0.5 m/s**;
- falls: **3 repeat falls + 1 sweep fall** vs required **0**.

The qualitative review was not run because primary quantitative gates had already failed.

## Interpretation and limits

Known implementation defects identified before this run were corrected and covered by the project's regression/audit process. Those known defects are insufficient to explain the final failure. The result supports a method/data-level limitation in this tested configuration, while undiscovered implementation issues cannot be excluded.

The mechanism is unresolved. Two live hypotheses are:

1. one reference motion does not provide adequate style/support across the commanded speed range;
2. the task/style reward balance or discriminator conditioning is insufficient to preserve command response.

The current evidence does not distinguish these mechanisms. The queued follow-up is a final-speed/gait map of the released Kine2Go motions, followed by a controlled pilot using a small bank of homogeneous straight-locomotion references. That follow-up is not part of this frozen baseline.
