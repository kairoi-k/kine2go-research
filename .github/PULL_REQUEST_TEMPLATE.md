## Summary

Describe the change and why it is needed.

## Scope

- [ ] Documentation / repository maintenance only
- [ ] Training or policy behavior
- [ ] Evaluation semantics / metrics
- [ ] Dataset / reference-motion processing
- [ ] Experiment protocol / acceptance criteria
- [ ] Other

## Research-semantic impact

State explicitly whether this PR changes the meaning of an existing experiment, metric, evaluator, preregistered gate, or scientific claim. If yes, identify the new version/protocol and which prior artifacts become superseded or require re-evaluation.

## Validation

List the checks, tests, or experiments run. Separate static/code checks from scientific validation.

## Provenance

Record relevant commit/config/data/checkpoint/evaluator identifiers when the change affects research results.

## Checklist

- [ ] I did not silently modify a frozen evaluator or preregistered acceptance criterion.
- [ ] Failed or superseded results needed for provenance remain traceable.
- [ ] Claims in docs are bounded by the evidence actually produced.
- [ ] Host-specific paths, credentials, and private artifacts are not introduced.
- [ ] New research-semantic behavior has an explicit version or protocol boundary.
