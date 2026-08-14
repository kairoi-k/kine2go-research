#!/usr/bin/env python3
"""Launch the frozen v5 evaluator with repo-relative paths.

`quant_eval_v5.py` is kept byte-for-byte because its content hash is part of
the recorded provenance. This launcher does not rewrite that file. It loads
the source, replaces the development-machine prefix with this checkout root,
and executes the result. New evaluations should go through this launcher;
do not silently edit the frozen evaluator.
"""

from __future__ import annotations

from pathlib import Path

FROZEN = Path(__file__).with_name("quant_eval_v5.py")
ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = "/home/che/dev/kine2go-research"


def main() -> None:
    source = FROZEN.read_text(encoding="utf-8")
    if LEGACY_ROOT not in source:
        raise SystemExit(
            "frozen evaluator no longer contains the expected legacy path; "
            "update this launcher rather than editing quant_eval_v5.py in place",
        )
    patched = source.replace(LEGACY_ROOT, str(ROOT))
    code = compile(patched, str(FROZEN), "exec")
    namespace = {"__name__": "__main__", "__file__": str(FROZEN)}
    exec(code, namespace)  # noqa: S102


if __name__ == "__main__":
    main()
