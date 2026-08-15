"""Train the conditional adversarial motion-prior (AMP) locomotion baseline.

The training path combines PPO imitation, a speed-conditioned LS-GAN
discriminator, adaptive forward-command curriculum, paired policy/discriminator
checkpoints, and a decaying joint-rotation anchor.
"""

import hashlib
import os
import pickle
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import genesis as gs
import numpy as np
import torch
import tyro
from tyro.conf import Positional

from go2_genesis.logging_utils import LoggingOnPolicyRunner
from motion_imitation.amp import (
    PHYSICAL_SPEED_EDGES,
    AMPDiscriminator,
    BoxAdaptiveCurriculum,
    build_state_feature,
    style_reward_from_logit,
    update_discriminator,
)
from motion_imitation.config import get_imitation_cfgs, get_ppo_cfg
from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class TrainCfg:
    exp_name: Positional[str]
    motion_path: Positional[str]
    motion_start: int = 0
    motion_end: int | None = None
    num_envs: int = 16384
    max_iterations: int = 5000
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    cpu: bool = False
    entity: str = "quadruped-rl"
    group: str | None = None
    project: str = "amp_imitation"
    resume_from: str | None = None
    seed: int = 1

    # Conditional AMP.
    disc_data: str = "data/disc_data_c6.npy"
    style_weight: float = 0.3
    tracking_weight: float = 2.0
    tracking_sigma: float = 0.25
    speed_match: bool = False
    fix_lateral_yaw: bool = True
    joint_rot_weight_init: float = 0.5
    joint_rot_decay_iters: int = 3500
    joint_rot_floor: float = 0.05
    disc_update_interval: int = 50
    disc_lr: float = 1e-5
    disc_lambda_gp: float = 10.0
    disc_batch: int = 4096
    disc_buckets: int = 6

    # Positive-only command curriculum used by the frozen clean run.
    curriculum: bool = True
    cur_start_range: tuple = (0.9, 1.1)
    cur_target_range: tuple = (0.5, 2.5)
    cur_eval_interval: int = 100
    cur_success_threshold: float = 0.6

    save_interval: int = 500


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _save_disc_pair(log_dir, disc, disc_opt, it, args):
    """Atomically save the discriminator paired with ``model_<it>.pt``."""
    tmp = os.path.join(log_dir, f"disc_{it}.pt.tmp")
    final = os.path.join(log_dir, f"disc_{it}.pt")
    torch.save(
        {
            "disc_state_dict": disc.state_dict(),
            "disc_opt_state_dict": disc_opt.state_dict(),
            "iter": it,
            "policy_sha256": _sha256_of(
                os.path.join(log_dir, f"model_{it}.pt"),
            ),
        },
        tmp,
    )
    os.replace(tmp, final)
    print(
        f"[amp] saved paired checkpoints: model_{it}.pt + disc_{it}.pt",
        flush=True,
    )


def _git_head_for_repo() -> str:
    """Return the repository HEAD for provenance, or an empty string on failure."""
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(  # noqa: S603,S607
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def main(args: TrainCfg):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level="warning",
    )
    log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_imitation_cfgs()
    if getattr(args, "fix_lateral_yaw", True):
        command_cfg["lin_vel_y_range"] = (0.0, 0.0)
        command_cfg["ang_vel_range"] = (0.0, 0.0)

    reward_cfg["reward_scales"]["tracking_lin_vel"] = args.tracking_weight
    reward_cfg["tracking_sigma"] = args.tracking_sigma
    reward_cfg["speed_match"] = args.speed_match

    expected_dt = 1.0 / env_cfg["control_freq"]
    reward_cfg["reward_scales"]["joint_rot"] = args.joint_rot_weight_init

    if os.path.exists(log_dir):
        if args.resume_from and os.path.dirname(args.resume_from) == log_dir:
            raise RuntimeError(
                f"resume_from points inside the destination log directory ({log_dir}); choose a different exp_name",
            )
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    retargetted_motions = [
        (args.motion_path, args.motion_start, args.motion_end),
    ]
    training_device = "cpu" if args.cpu else "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = ImitationWrapper(
        retargetted_motions,
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=False,
        debug=False,
        device=training_device,
    )
    assert abs(env.dt - expected_dt) < 1e-9, f"env.dt={env.dt} != expected control dt={expected_dt}"
    assert (
        min(args.cur_target_range) >= PHYSICAL_SPEED_EDGES[0] and max(args.cur_target_range) <= PHYSICAL_SPEED_EDGES[-1]
    ), (
        f"cur_target_range {args.cur_target_range} exceeds discriminator "
        f"support {PHYSICAL_SPEED_EDGES[0]}-{PHYSICAL_SPEED_EDGES[-1]}"
    )

    policy_cfg = get_ppo_cfg(args)
    policy_cfg["motion_range"] = (args.motion_start, args.motion_end)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg],
            f,
        )
    shutil.copyfile(args.motion_path, f"{log_dir}/motion.npy")
    with open(f"{log_dir}/train_git_commit.txt", "w") as f:
        f.write(_git_head_for_repo())

    disc = AMPDiscriminator(state_dim=30).to(training_device)
    disc_opt = torch.optim.Adam(
        disc.parameters(),
        lr=args.disc_lr,
    )

    ref_data = np.load(args.disc_data)
    ref_s = torch.from_numpy(ref_data[:, :30]).float().to(training_device)
    ref_sp = torch.from_numpy(ref_data[:, 30:60]).float().to(training_device)
    ref_c = torch.from_numpy(ref_data[:, 60]).float().to(training_device)
    print(
        "[amp] conditional discriminator: "
        f"{len(ref_data)} transition pairs, "
        f"speed {ref_c.min():.2f}-{ref_c.max():.2f} m/s, "
        f"{args.disc_buckets} buckets",
        flush=True,
    )

    env.style_disc = disc
    env.style_weight = args.style_weight
    env.last_s = None
    env.last_s_valid = torch.zeros(
        args.num_envs,
        dtype=torch.bool,
        device=training_device,
    )
    orig_compute_reward = env.compute_reward

    def compute_reward_with_style():
        orig_compute_reward()
        s = build_state_feature(env).detach()
        if env.last_s is not None:
            with torch.no_grad():
                done_now = env.reset_buf
                valid_s = (env.last_s_valid & (~done_now)).nonzero(as_tuple=False).squeeze(-1)
                if len(valid_s) > 0:
                    cmd_c = getattr(
                        env,
                        "prev_commands",
                        env.commands,
                    )
                    logit = disc(
                        env.last_s[valid_s],
                        s[valid_s],
                        cmd_c[valid_s, 0],
                    )
                    style_rew = torch.zeros_like(env.rew_buf)
                    style_rew[valid_s] = style_reward_from_logit(
                        logit,
                        env.style_weight,
                        env.dt,
                    )
                    env.rew_buf += style_rew

        env.last_s_valid = (~env.reset_buf).bool()
        env.last_s = s

    env.compute_reward = compute_reward_with_style

    if args.curriculum:
        cur = BoxAdaptiveCurriculum(
            env,
            start_range=args.cur_start_range,
            target_range=args.cur_target_range,
            eval_interval=args.cur_eval_interval,
            success_threshold=args.cur_success_threshold,
            support_range=(
                PHYSICAL_SPEED_EDGES[0],
                PHYSICAL_SPEED_EDGES[-1],
            ),
        )
        print(
            f"[amp] curriculum: {args.cur_start_range} -> {args.cur_target_range}",
            flush=True,
        )

    runner = LoggingOnPolicyRunner(
        env,
        policy_cfg,
        log_dir,
        device=training_device,
    )
    if args.resume_from:
        print(f"[amp] resume: {args.resume_from}", flush=True)
        runner.load(args.resume_from, load_optimizer=False)

        base = os.path.basename(args.resume_from)
        iter_tag = base.replace("model_", "").replace(".pt", "")
        disc_path = os.path.join(
            os.path.dirname(args.resume_from),
            f"disc_{iter_tag}.pt",
        )

        if not os.path.exists(disc_path):
            legacy = os.path.join(
                os.path.dirname(args.resume_from),
                "disc.pt",
            )
            if os.path.exists(legacy):
                print(
                    f"[warning] no paired disc_{iter_tag}.pt; falling back to disc.pt, "
                    "which may be from a different iteration",
                    flush=True,
                )
                disc_path = legacy
            else:
                print(
                    "[warning] no discriminator checkpoint found; starting the discriminator from scratch",
                    flush=True,
                )
                disc_path = None

        if disc_path and os.path.exists(disc_path):
            d = torch.load(  # noqa: S614
                disc_path,
                map_location=training_device,
            )
            disc.load_state_dict(d["disc_state_dict"])
            disc_opt.load_state_dict(d["disc_opt_state_dict"])
            got_iter = d.get("iter", -1)
            print(
                f"[amp] discriminator restored from {disc_path} (iter {got_iter})",
                flush=True,
            )
            if iter_tag.isdigit() and got_iter != int(iter_tag):
                print(
                    f"[warning] discriminator/policy checkpoint iteration mismatch: disc={got_iter}, policy={iter_tag}",
                    flush=True,
                )

    alg = runner.alg

    runner._prepare_logging_writer()
    obs = env.get_observations().to(training_device)
    runner.train_mode()
    num_steps = policy_cfg["num_steps_per_env"]
    policy_samples = deque(maxlen=64)
    prev_s = None

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
        args.num_envs,
        device=training_device,
    )
    cur_episode_length = torch.zeros(
        args.num_envs,
        device=training_device,
    )

    for it in range(args.max_iterations):
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(num_steps):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(
                    actions.to(env.device),
                )
                obs = obs.to(training_device)
                rewards = rewards.to(training_device)
                dones = dones.to(training_device)
                alg.process_env_step(
                    obs,
                    rewards,
                    dones,
                    extras,
                )

                s_now = build_state_feature(env).detach()
                if prev_s is not None:
                    valid = (dones <= 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(valid) > 0:
                        cmd_c = getattr(
                            env,
                            "prev_commands",
                            env.commands,
                        )
                        policy_samples.append(
                            (
                                prev_s[valid].clone(),
                                s_now[valid].clone(),
                                cmd_c[valid, 0].clone(),
                            ),
                        )
                prev_s = s_now

                if args.curriculum:
                    cur.record(
                        env.commands[:, 0],
                        env.base_lin_vel[:, 0],
                        (dones <= 0).bool(),
                    )

                cur_reward_sum += rewards
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(
                    cur_reward_sum[new_ids][:, 0].cpu().tolist(),
                )
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
                if "episode" in extras:
                    ep_infos.append(extras["episode"])

            alg.compute_returns(obs)

        collect_t = time.time() - t0

        disc_loss = 0.0
        if it % args.disc_update_interval == 0 and policy_samples:
            all_f = [torch.cat([s, sp, c[:, None]], dim=-1) for s, sp, c in policy_samples]
            pf = torch.cat(all_f, dim=0)
            perm = torch.randperm(len(pf))[: min(args.disc_batch, len(pf))]
            pf = pf[perm]
            s_f = pf[:, :30]
            sp_f = pf[:, 30:60]
            c_f = pf[:, 60]
            disc_loss = update_discriminator(
                disc,
                disc_opt,
                (ref_s, ref_sp, ref_c),
                (s_f, sp_f, c_f),
                args.disc_lambda_gp,
                args.disc_buckets,
            )

        if it < args.joint_rot_decay_iters:
            w = (
                args.joint_rot_weight_init
                - (args.joint_rot_weight_init - args.joint_rot_floor) * it / args.joint_rot_decay_iters
            )
        else:
            w = args.joint_rot_floor

        env.reward_scales["joint_rot"] = max(w, args.joint_rot_floor) * env.dt

        cur_info = None
        if args.curriculum and it % args.cur_eval_interval == 0:
            cur_info = cur.step_update(it)

        alg.update()
        learn_t = time.time() - t0 - collect_t
        _ = learn_t

        try:
            runner.writer.add_scalar(
                "Train/reward",
                float(np.mean(rewbuffer)) if rewbuffer else 0.0,
                it,
            )
            runner.writer.add_scalar("Train/disc_loss", disc_loss, it)
            runner.writer.add_scalar(
                "Train/cmd_range_lo",
                env.command_cfg["lin_vel_x_range"][0],
                it,
            )
            runner.writer.add_scalar(
                "Train/cmd_range_hi",
                env.command_cfg["lin_vel_x_range"][1],
                it,
            )
            runner.writer.add_scalar(
                "Train/joint_rot_weight",
                env.reward_scales["joint_rot"],
                it,
            )
        except Exception:  # noqa: S110
            pass

        if it % 100 == 0:
            mean_rew = float(np.mean(rewbuffer)) if rewbuffer else 0.0
            print(
                f"[{it:5d}] reward={mean_rew:.2f} "
                f"disc_loss={disc_loss:.3f} "
                "range="
                f"{tuple(round(x, 2) for x in env.command_cfg['lin_vel_x_range'])} "
                "steps/s="
                f"{args.num_envs * num_steps / max(collect_t, 1e-6):.0f} "
                "joint_rot_w="
                f"{env.reward_scales['joint_rot']:.3f} "
                f"{cur_info or ''}",
                flush=True,
            )

        if it % args.save_interval == 0:
            runner.save(os.path.join(log_dir, f"model_{it}.pt"))
            _save_disc_pair(log_dir, disc, disc_opt, it, args)

    runner.save(
        os.path.join(log_dir, f"model_{args.max_iterations}.pt"),
    )
    _save_disc_pair(
        log_dir,
        disc,
        disc_opt,
        args.max_iterations,
        args,
    )
    print("[amp] training complete", flush=True)


if __name__ == "__main__":
    main(tyro.cli(TrainCfg))
