import os
import pickle
from dataclasses import dataclass

import genesis as gs
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from motion_imitation.imitation_wrapper import ImitationWrapper


@dataclass
class EvalConfig:
    ckpt_file: tyro.conf.Positional[str]
    headless: bool = False
    cpu: bool = False
    record: bool = False
    camera_res: tuple[int, int] | None = None
    """录制分辨率, 如 (1280,1280). None=Genesis默认(低)"""
    num_episodes: int = 1
    max_steps: int | None = None
    """录制最大步数 (冒烟测试用), None=跑到episode结束"""
    fix_heading: bool = False
    """固定命令 (直线+固定速度, 侧视录制用)"""
    fix_speed: float = 1.5
    """fix_heading 时的固定速度"""
    measure: bool = False
    """测量模式: 输出速度曲线到 measure_log.txt (不录制视频)"""
    debug: bool = False


def main(args):
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning", performance_mode=False)

    log_dir = os.path.dirname(args.ckpt_file)

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(cfg_file)

    motion_path = os.path.join(log_dir, "motion.npy")
    motion_start, motion_end = policy_cfg["motion_range"]
    eval_motion = [(motion_path, motion_start, motion_end)]

    device = "cuda"
    if args.cpu:
        device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    env_cfg["perturb_init_state"] = True
    env = ImitationWrapper(
        eval_motion,
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        eval=True,
        debug=args.debug,
        device=device,
        camera_res=args.camera_res,
    )

    args.max_iterations = 1
    runner = OnPolicyRunner(env, policy_cfg, log_dir, device=device)

    runner.load(args.ckpt_file, map_location=device)

    policy = runner.get_inference_policy(device=device)

    # 固定命令: 直线 + 固定速度 (录制用), 避免周期性变速
    if getattr(args, "fix_heading", False):
        env.commands[:, 0] = getattr(args, "fix_speed", 1.5)  # lin_vel_x 固定
        env.commands[:, 1] = 0.0  # lin_vel_y = 0
        env.commands[:, 2] = 0.0  # ang_vel_yaw = 0
    env.reset()
    obs = env.get_observations()

    with torch.no_grad():
        if args.record and not args.measure:
            env.start_recording(record_internal=True)  # 用侧视机位+高质量编码

        for _ in range(args.num_episodes):
            stop = False

            _step_count = 0
            while not stop:
                # 每步钉住命令 (避免重采样覆盖)
                if getattr(args, "fix_heading", False):
                    env.commands[:, 0] = getattr(args, "fix_speed", 1.5)
                    env.commands[:, 1] = 0.0
                    env.commands[:, 2] = 0.0
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                _step_count += 1
                if args.max_steps is not None and _step_count >= args.max_steps:
                    stop = True
                if dones[0]:
                    stop = True

    if args.record and not args.measure:
        env.stop_recording(os.path.join(log_dir, "recording.mp4"))

    if args.measure:
        # 速度曲线: 每 dt 记录实际速度 (走 20s)
        speeds = []
        prev_pos = None
        with torch.no_grad():
            for _i in range(int(20.0 / env.dt)):
                if getattr(args, "fix_heading", False):
                    env.commands[:, 0] = getattr(args, "fix_speed", 1.5)
                    env.commands[:, 1] = 0.0
                    env.commands[:, 2] = 0.0
                a = policy(obs)
                obs, _, _, _ = env.step(a)
                pos = env.base_pos[0].detach().cpu().numpy().copy()
                if prev_pos is not None:
                    speeds.append(float(np.linalg.norm(pos - prev_pos) / env.dt))
                prev_pos = pos
        # 每 0.25s 平均
        seg = 12  # 0.25s / dt
        agg = [round(float(np.mean(speeds[i : i + seg])), 4) for i in range(0, len(speeds), seg)]
        with open(os.path.join(log_dir, "measure_log.txt"), "w") as f:
            f.write(f"cmd_speed={getattr(args, 'fix_speed', 1.5)} dt={env.dt}\n")
            f.write(
                f"mean={np.mean(speeds):.4f} std={np.std(speeds):.4f} "
                f"min={np.min(speeds):.4f} max={np.max(speeds):.4f}\n"
            )
            f.write("time_s,speed_mps\n")
            for i, v in enumerate(agg):
                f.write(f"{i * 0.25:.2f},{v}\n")
        print(
            f"measure done: mean={np.mean(speeds):.4f} std={np.std(speeds):.4f} "
            f"min={np.min(speeds):.4f} max={np.max(speeds):.4f}"
        )


if __name__ == "__main__":
    args = tyro.cli(EvalConfig)
    main(args)
