# Provenance

## Upstream

This repository is a research fork of:

- repository: `nomagiclab/kine2go-pipeline`
- upstream base: `65b39f104706c1b4307dfc2a7df2aa8bed8d20aa`
- paper: Pałucki et al., *Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions* (2026)

The upstream BSD 3-Clause license is preserved in `LICENSE`.

## Research-fork changes

The fork adds research infrastructure around the upstream retargeting/imitation pipeline, including:

- speed-conditioned adversarial motion-prior training;
- command-support-aware curriculum logic;
- real/fake condition-matched discriminator sampling;
- transition-boundary and feature-semantics corrections;
- paired policy/discriminator checkpoint handling;
- versioned quantitative evaluation;
- preregistered acceptance criteria and explicit negative-result reporting;
- retained three-seed reference-seam evaluation JSON (historical v2 protocol).

`docs/IMPLEMENTATION_AUDIT.md` records correctness issues relevant to the frozen experiment without reproducing internal review transcripts.

## Development provenance

The research state was finalized at development milestone `0ad4343` on 2026-08-13. The frozen evaluation record contains more specific execution provenance:

- training commit: `c64922265fa8cd92529b66b6aaaa52f16fe8eefd`
- evaluation-time commit recorded by v5: `c57a9a8adc38e27e04cf4729f1ade3ce3cab6531`
- evaluated model SHA-256 prefix: `4dfb111d165275f9`
- motion SHA-256 prefix: `d1367f511b9d1402`
- evaluator SHA-256 prefix: `7d6f6d4a1649f5ab`
- configuration hash: `a31836d7ae2aaf95`

These IDs are from the development archive; they are not commits in this repository.

## Large artifacts

The evaluated `model_5000.pt` is not recoverable. It lived at `logs/amp_clean_run/model_5000.pt` under a `/tmp` working tree and was wiped by a WSL reboot on 2026-08-14. Identity remains the recorded SHA-256 prefix `4dfb111d165275f9` and the committed v5 JSON.

The canonical conditioned discriminator dataset `data/disc_data_c6.npy` is in git because it is consumed by the frozen training path. Superseded intermediate discriminator arrays are not.

## External source attributions

Some environment and imitation code derives from or references third-party projects in comments/docstrings. Those attributions should be preserved when modifying the corresponding code. Third-party assets may have their own licenses and are not automatically relicensed by this repository.
