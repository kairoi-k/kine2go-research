"""Build speed-conditioned discriminator transition data for conditional AMP.

Usage:
    python data/extract_disc_data_c.py <clips_dir> <out_npy>

Output:
    ``(N, 61) = [s(30), s'(30), c(1)]`` where ``c`` is the smoothed
    horizontal forward-speed label in metres per second.
"""

import glob
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

clips_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])

fs_hz = 60.0

# Kine2Go motion columns use the Genesis-native joint ordering, while the
# policy feature path uses the configured motor ordering. Apply the same
# permutation to position and velocity before constructing real AMP features.
joint_permutation = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

pairs = []
for clip_path_str in sorted(
    glob.glob(str(clips_dir / "*" / "motion.npy")),
):
    clip_path = Path(clip_path_str)
    name = clip_path.parent.name
    motion = np.load(clip_path)
    if len(motion) < 5:
        continue

    joint_pos = motion[:, 6:18][:, joint_permutation]
    joint_vel = motion[:, 24:36][:, joint_permutation]
    lin_vel_world = motion[:, 55:58]
    ang_vel_world = motion[:, 58:61]
    quat = motion[:, 51:55]

    w, x, y, z = (
        quat[:, 0],
        quat[:, 1],
        quat[:, 2],
        quat[:, 3],
    )
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotate world-frame linear velocity into the yaw-aligned frame.
    lin_vel_yaw = np.stack(
        [
            cy * lin_vel_world[:, 0] + sy * lin_vel_world[:, 1],
            -sy * lin_vel_world[:, 0] + cy * lin_vel_world[:, 1],
            lin_vel_world[:, 2],
        ],
        axis=1,
    )

    # Rotate world-frame angular velocity into the full body frame.
    rotation = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    ang_vel_body = np.einsum(
        "nij,nj->ni",
        rotation.transpose(0, 2, 1),
        ang_vel_world,
    )

    features = np.concatenate(
        [
            joint_pos,
            joint_vel,
            lin_vel_yaw,
            ang_vel_body,
        ],
        axis=1,
    )

    # Label each t -> t+1 transition with smoothed horizontal speed.
    pos = motion[:, 0:3]
    vel = np.diff(pos, axis=0) * fs_hz
    speed = np.linalg.norm(vel[:, :2], axis=1)
    speed = uniform_filter1d(speed, size=15)

    state = features[:-1]
    next_state = features[1:]
    valid = speed > 0.1
    if valid.sum() < 20:
        print(f"skip {name}: only {valid.sum()} valid transitions")
        continue

    pairs.append(
        np.concatenate(
            [
                state[valid],
                next_state[valid],
                speed[valid, None],
            ],
            axis=1,
        ),
    )
    print(
        f"{name}: {valid.sum()} transitions, median speed {np.median(speed[valid]):.2f} m/s",
    )

if not pairs:
    raise RuntimeError(f"no valid motion transitions found under {clips_dir}")

data = np.concatenate(pairs, axis=0)
np.save(out_path, data)
print(f"saved {len(data)} transitions -> {out_path}")
print(
    "speed labels: "
    f"min={data[:, -1].min():.2f} "
    f"median={np.median(data[:, -1]):.2f} "
    f"p90={np.percentile(data[:, -1], 90):.2f} "
    f"max={data[:, -1].max():.2f}",
)
