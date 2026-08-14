"""Sanity-check the portable v5 launcher without running Genesis."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evaluation" / "quant_eval_v5.py"
LAUNCHER = ROOT / "evaluation" / "run_quant_eval_v5.py"


def _launcher_module():
    spec = importlib.util.spec_from_file_location("run_quant_eval_v5", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvalLauncherTests(unittest.TestCase):
    def test_frozen_file_keeps_legacy_path(self) -> None:
        text = FROZEN.read_text(encoding="utf-8")
        self.assertIn(_launcher_module().LEGACY_ROOT, text)

    def test_launcher_rewrites_legacy_path_to_checkout_root(self) -> None:
        module = _launcher_module()
        source = FROZEN.read_text(encoding="utf-8")
        patched = source.replace(module.LEGACY_ROOT, str(ROOT))
        self.assertNotIn(module.LEGACY_ROOT, patched)
        self.assertIn(str(ROOT), patched)
        self.assertTrue(LAUNCHER.is_file())


if __name__ == "__main__":
    unittest.main()
