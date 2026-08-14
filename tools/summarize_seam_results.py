#!/usr/bin/env python3
"""Print the three-seed seam intervention table from committed JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval_results"
PAIRS = (
    ("seed1", "seam_orig_s1_v2.json", "fixed_s1_v2.json"),
    ("seed2", "orig_s2_v2.json", "fixed_s2_v2.json"),
    ("seed3", "orig_s3_v2.json", "fixed_s3_v2.json"),
)


def _rmse(path: Path) -> float:
    payload = json.loads(path.read_text())
    return float(payload["metrics"]["A_speed"]["rmse"]["mean"])


def _dip(path: Path) -> float:
    payload = json.loads(path.read_text())
    return float(payload["metrics"]["A_speed"]["dip_count_abs"]["mean"])


def rows() -> list[tuple[str, float, float, float, float, float]]:
    out = []
    for name, orig_name, fixed_name in PAIRS:
        orig = RESULTS / orig_name
        fixed = RESULTS / fixed_name
        orig_rmse = _rmse(orig)
        fixed_rmse = _rmse(fixed)
        delta = (fixed_rmse - orig_rmse) / orig_rmse
        out.append(
            (name, orig_rmse, fixed_rmse, delta, _dip(orig), _dip(fixed)),
        )
    return out


def main() -> None:
    data = rows()
    orig_mean = sum(item[1] for item in data) / len(data)
    fixed_mean = sum(item[2] for item in data) / len(data)
    orig_dip = sum(item[4] for item in data) / len(data)
    fixed_dip = sum(item[5] for item in data) / len(data)
    print("seed  orig_rmse  fixed_rmse  rmse_delta  orig_dip  fixed_dip")
    for name, orig_rmse, fixed_rmse, delta, orig_dip_v, fixed_dip_v in data:
        print(
            f"{name}  {orig_rmse:.3f}      {fixed_rmse:.3f}       "
            f"{delta * 100:+.0f}%        {orig_dip_v:.2f}      {fixed_dip_v:.2f}",
        )
    print(
        f"agg   {orig_mean:.3f}      {fixed_mean:.3f}       "
        f"{(fixed_mean - orig_mean) / orig_mean * 100:+.0f}%        "
        f"{orig_dip:.2f}      {fixed_dip:.2f}",
    )


if __name__ == "__main__":
    main()
