"""Check committed seam JSON artifacts against the published RMSE table."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_seam_results import rows


class SeamArtifactTests(unittest.TestCase):
    def test_per_seed_rmse_direction_and_reported_percents(self) -> None:
        data = rows()
        self.assertEqual(len(data), 3)
        percents = []
        for _name, orig_rmse, fixed_rmse, delta, _orig_dip, _fixed_dip in data:
            self.assertLess(fixed_rmse, orig_rmse)
            percents.append(round(delta * 100))
        self.assertEqual(percents, [-29, -23, -41])

    def test_aggregate_rmse_matches_index(self) -> None:
        data = rows()
        orig_mean = sum(item[1] for item in data) / len(data)
        fixed_mean = sum(item[2] for item in data) / len(data)
        self.assertAlmostEqual(orig_mean, 0.286, places=3)
        self.assertAlmostEqual(fixed_mean, 0.196, places=3)
        self.assertEqual(round((fixed_mean - orig_mean) / orig_mean * 100), -31)


if __name__ == "__main__":
    unittest.main()
