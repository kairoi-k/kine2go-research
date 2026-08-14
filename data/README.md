# Conditioned discriminator data

`disc_data_c6.npy` is the canonical real-transition dataset used by the frozen conditional-AMP training path.

## Schema

Each row has 61 float features:

```text
[s_t (30), s_{t+1} (30), c (1)]
```

Each 30-D state contains, in order:

```text
joint_position(12)
joint_velocity(12)
base_linear_velocity_yaw_frame(3)
base_angular_velocity_body_frame(3)
```

`c` is the smoothed horizontal motion-speed label in metres per second.

The real motion joint columns are permuted into the same motor order used by policy features:

```text
[1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
```

## Generation

Use:

```bash
python data/extract_disc_data_c.py <clips_dir> <output.npy>
```

The generator expects Kine2Go-style `*/motion.npy` clips at 60 Hz. It converts world-frame linear/angular velocities to the policy feature frames, constructs adjacent-frame transitions, smooths horizontal speed labels, and removes near-stationary transitions below 0.1 m/s.

## What's in git

Only `disc_data_c6.npy` is the frozen baseline input. Earlier `disc_data*` variants are not in this tree.

Git blob SHA of `disc_data_c6.npy`: `ada1bd929a3b675735ff54c2e96d3b44b541d886`.

For new experiments, write a new data artifact and record its content hash rather than overwriting the frozen baseline input.
