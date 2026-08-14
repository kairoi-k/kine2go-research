"""Conditional adversarial motion prior utilities.

The discriminator consumes transition pairs ``(s, s', c)`` where each state has
30 features and ``c`` is the forward-speed condition. Real samples are drawn
from fixed physical-speed buckets and matched to the fake-sample command
marginal during discriminator updates.
"""

import numpy as np
import torch
import torch.nn as nn


def build_state_feature(env):
    """Extract the 30-D discriminator state feature from a Genesis environment."""
    dof_pos = env.dof_pos
    dof_vel = env.dof_vel
    lin_vel = env.base_lin_vel
    ang_vel = env.base_ang_vel
    return torch.cat([dof_pos, dof_vel, lin_vel, ang_vel], dim=-1)


class AMPDiscriminator(nn.Module):
    """Conditional discriminator over ``(s, s', c)`` transition features."""

    def __init__(self, state_dim: int = 30, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, sp, c):
        x = torch.cat([s, sp, c[:, None]], dim=-1)
        return self.net(x).squeeze(-1)


def style_reward_from_logit(
    logit: torch.Tensor,
    scale: float = 1.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Map discriminator logits to the timestep-scaled style reward."""
    return torch.sigmoid(logit) * scale * dt


def r1_gradient_penalty(disc, real_s, real_sp, real_c, lambda_gp: float = 10.0):
    """Return the mean per-sample R1 gradient penalty."""
    real_s.requires_grad_(True)  # noqa: FBT003
    real_sp.requires_grad_(True)  # noqa: FBT003
    d_real = disc(real_s, real_sp, real_c)
    grads = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=[real_s, real_sp],
        create_graph=True,
        retain_graph=True,
    )
    grad_norm = sum((g * g).view(g.shape[0], -1).sum(dim=1).mean() for g in grads)
    gp = 0.5 * lambda_gp * grad_norm
    real_s.requires_grad_(False)  # noqa: FBT003
    real_sp.requires_grad_(False)  # noqa: FBT003
    return gp


_BUCKET_RNG = np.random.default_rng(42)

# Fixed physical-speed buckets in m/s. Equal bucket weighting prevents the
# discriminator update from being dominated by densely sampled walking speeds.
PHYSICAL_SPEED_EDGES = np.array([0.3, 0.8, 1.2, 1.6, 2.1, 2.7, 3.5])


def bucket_balanced_sample(
    ref_s,
    ref_sp,
    ref_c,
    batch,
    n_buckets=None,
    rng=None,
    edges=None,
):
    """Sample approximately equal counts from fixed physical-speed buckets."""
    if rng is None:
        rng = _BUCKET_RNG
    if edges is None:
        edges = PHYSICAL_SPEED_EDGES

    ref_c_np = np.asarray(ref_c.detach().cpu().numpy() if isinstance(ref_c, torch.Tensor) else ref_c)
    n_buckets = len(edges) - 1
    per = max(batch // n_buckets, 8)
    idxs = []

    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        in_bucket = np.nonzero((ref_c_np >= lo) & (ref_c_np < hi))[0]
        if len(in_bucket) == 0:
            continue
        if len(in_bucket) >= per:
            chosen = in_bucket[rng.choice(len(in_bucket), per, replace=False)]
        else:
            extra = rng.choice(
                len(in_bucket),
                per - len(in_bucket),
                replace=True,
            )
            chosen = np.concatenate([in_bucket, in_bucket[extra]])
        idxs.append(chosen)

    if len(idxs) == 0:
        return ref_s[:batch], ref_sp[:batch], ref_c[:batch]

    idx = np.concatenate(idxs)
    if len(idx) > batch:
        idx = idx[rng.choice(len(idx), batch, replace=False)]
    elif len(idx) < batch:
        extra = rng.choice(
            len(idx),
            batch - len(idx),
            replace=True,
        )
        idx = np.concatenate([idx, idx[extra]])

    assert len(idx) == batch, f"bucket sample length {len(idx)} != batch {batch}"
    return ref_s[idx], ref_sp[idx], ref_c[idx]


def update_discriminator(
    disc,
    opt,
    ref_pairs,
    policy_pairs,
    lambda_gp: float = 10.0,
    n_buckets: int = 8,
):
    """Update the conditional LS-GAN discriminator with R1 regularization."""
    s_f, sp_f, c_f = policy_pairs
    if s_f.shape[0] < 64:
        return 0.0

    s_r, sp_r, c_r = ref_pairs

    n = min(len(s_f), len(s_r))
    idx_f = torch.randperm(len(s_f))[:n]
    s_f, sp_f, c_f = s_f[idx_f], sp_f[idx_f], c_f[idx_f]

    # Match the real speed-condition marginal to the fake command marginal so
    # the discriminator cannot classify samples from p(real c) vs p(fake c).
    c_f_np = c_f.detach().cpu().numpy()
    edges = PHYSICAL_SPEED_EDGES
    counts = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        counts[b] = int(((c_f_np >= edges[b]) & (c_f_np < edges[b + 1])).sum())

    s_r_np = s_r.detach().cpu().numpy()
    sp_r_np = sp_r.detach().cpu().numpy()
    c_r_np = c_r.detach().cpu().numpy()
    rng = np.random.default_rng(0)
    idx_r = []

    for b in range(len(edges) - 1):
        in_b = np.nonzero((c_r_np >= edges[b]) & (c_r_np < edges[b + 1]))[0]
        want = int(counts[b])
        if len(in_b) == 0 or want <= 0:
            continue
        if len(in_b) >= want:
            idx_r.append(in_b[rng.choice(len(in_b), want, replace=False)])
        else:
            extra = rng.choice(
                len(in_b),
                want - len(in_b),
                replace=True,
            )
            idx_r.append(np.concatenate([in_b, in_b[extra]]))

    if len(idx_r) == 0:
        idx_r = [np.arange(min(n, len(s_r_np)))]

    idx_r = np.concatenate(idx_r)
    if len(idx_r) > n:
        idx_r = idx_r[rng.choice(len(idx_r), n, replace=False)]

    s_r = torch.from_numpy(s_r_np[idx_r]).to(s_f.device)
    sp_r = torch.from_numpy(sp_r_np[idx_r]).to(s_f.device)
    c_r = torch.from_numpy(c_r_np[idx_r]).float().to(s_f.device)

    opt.zero_grad()
    d_real = disc(s_r, sp_r, c_r)
    d_fake = disc(s_f, sp_f, c_f)
    loss = 0.5 * torch.mean((d_real - 1.0) ** 2) + 0.5 * torch.mean(d_fake**2)
    loss = loss + r1_gradient_penalty(
        disc,
        s_r,
        sp_r,
        c_r,
        lambda_gp,
    )
    loss.backward()
    opt.step()
    return float(loss.item())


class BoxAdaptiveCurriculum:
    """Success-rate-driven box curriculum for the forward command range."""

    def __init__(
        self,
        env,
        start_range=(-0.5, 1.0),
        target_range=(-1.0, 2.5),
        step=0.25,
        eval_interval: int = 100,
        success_threshold: float = 0.6,
        window: int = 20,
        track_sigma: float = 0.3,
        support_range=None,
    ):
        self.support_range = support_range
        self.env = env
        self.cur = list(start_range)
        self.target = target_range
        self.step = step
        self.eval_interval = eval_interval
        self.success_threshold = success_threshold
        self.window = window
        self.track_sigma = track_sigma
        self.recent_errors = []
        self._apply()

    def _apply(self):
        self.env.command_cfg["lin_vel_x_range"] = tuple(self.cur)

    def record(
        self,
        cmd_vx: torch.Tensor,
        actual_vx: torch.Tensor,
        mask: torch.Tensor,
    ):
        if mask.sum() == 0:
            return
        err = torch.abs(cmd_vx[mask] - actual_vx[mask])
        self.recent_errors.append(err.detach().cpu())

    def success_rate(self) -> float:
        if not self.recent_errors:
            return 0.0
        errs = torch.cat(self.recent_errors[-self.window :])
        return float((errs < self.track_sigma).float().mean())

    def step_update(self, it: int):
        if it % self.eval_interval != 0:
            return None

        sr = self.success_rate()
        lo, hi = self.cur
        if sr >= self.success_threshold:
            lo = max(self.target[0], lo - self.step)
            hi = min(self.target[1], hi + self.step)
        else:
            lo = min(lo + self.step * 0.5, 0.0)
            hi = max(hi - self.step * 0.5, 1.0)

        if self.support_range is not None:
            lo = max(lo, self.support_range[0])
            hi = min(hi, self.support_range[1])

        self.cur = [lo, hi]
        self._apply()
        self.recent_errors = []
        return (lo, hi, sr)
