"""quant_eval v5 (FROZEN, 2026-08-13) — v4 + provenance/fall 判定收尾
v5 变更 (gpt review 2026-08-13):
1. evaluator_file_sha256 指向本文件 (v4 误 hash quant_eval_v3.py — provenance bug)
2. 统一判定: 任何 sweep 档 fall/truncated -> incomplete (不产出 RMSE/span);
   预注册 span 在端点 incomplete 时 = undefined (不允许换端点口径)
quant_eval v4 (FROZEN, 2026-08-13) — v3 + contact 二值化 P0 修复
v4 变更: foot_contact_log 存 (force > 1N) 布尔 (v3 存 raw force, contact-derived metrics 全部 invalid)
contact-derived: duty_factor / flight_frac / contact_transition_rate_hz / stride_frequency_hz / diagonal_coord

quant_eval v3 (FROZEN, 2026-08-13) — 基于 v2.1 + 双 review 收尾
变更 vs v2.1:
1. meta 拆分: train_git_commit (checkpoint 元数据) / eval_git_commit (评估时代码) + 各 hash
2. fall 分段: fall 后 reset + 重新 warmup, 指标只统计完整连续段 (不再拼接跨 episode)
3. cadence 重定义: contact_transition_rate_hz (原 cadence_hz, 每足开关切换率)
   + stride_frequency_hz (同一足 touchdown 间隔频率, 即步频)
4. true_power_w -> estimated_mechanical_power_w (注明估计量)
5. push: 增加 lateral 响应指标 (y 方向速度峰值/恢复)
6. joint_jerk_filtered: savgol 滤波后位置信号三阶差分 (jerk, rad/s^3), 与代码一致
6. repeats/sweep 统计口径: 所有指标报告 mean±std (repeats>=2)
"""
import sys
sys.path.insert(0, "/home/che/dev/kine2go-research")
import os, pickle, json, argparse, hashlib, subprocess, time
import torch
import numpy as np
import genesis as gs

VERSION = "v5-frozen-20260813"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--cmd-speed", type=float, default=1.5)
parser.add_argument("--sweep", type=str, default=None)
parser.add_argument("--push-test", action="store_true")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

DIP_ABS_THRESHOLD = 0.5
WARMUP_S = 1.0
CONTACT_FORCE_THRESHOLD = 1.0
FALL_ROLL_LIMIT = 1.0
FALL_PITCH_LIMIT = 1.2
FALL_HEIGHT_LIMIT = 0.15
PUSH_ACC_MPS2 = 2.0

gs.init(backend=gs.gpu, logging_level="error")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

log_dir = os.path.dirname(args.ckpt)
with open(f"{log_dir}/cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg = pickle.load(f)

from motion_imitation.imitation_wrapper import ImitationWrapper
from rsl_rl.runners import OnPolicyRunner

def build_env():
    env = ImitationWrapper(
        [(f"{log_dir}/motion.npy", 0, None)], num_envs=1,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=False, eval=True, debug=False, device="cuda")
    runner = OnPolicyRunner(env, policy_cfg, log_dir, device="cuda")
    runner.load(args.ckpt, map_location="cuda")
    policy = runner.get_inference_policy(device="cuda")
    return env, policy

env, policy = build_env()
try:
    from genesis.engine.force_fields import Constant
    _push_ff = Constant(direction=[0.0, 1.0, 0.0], strength=PUSH_ACC_MPS2)
    env.scene.add_force_field(_push_ff)
    force_api = f"ForceField {PUSH_ACC_MPS2} m/s^2 lateral"
    force_ok = True
except Exception as e:
    _push_ff = None
    force_api = "no force field"
    force_ok = False

def run_episode(cmd_speed, duration, record_all=False):
    """完整连续段评估; fall 时终止该段并标记 (不拼接)"""
    env.reset()
    obs = env.get_observations()
    dt = env.dt
    n_steps = int(duration / dt)
    warmup_steps = int(WARMUP_S / dt)
    vx_log, vy_log, roll_log, pitch_log, height_log = [], [], [], [], []
    jp_log, foot_contact_log, torque_log, jv_log = [], [], [], []
    fall_at_step = None
    with torch.no_grad():
        for i in range(n_steps):
            env.commands[:, 0] = cmd_speed
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            vx_log.append(float(env.base_lin_vel[0, 0].item()))
            vy_log.append(float(env.base_lin_vel[0, 1].item()))
            roll_log.append(float(env.base_euler[0, 0].item()))
            pitch_log.append(float(env.base_euler[0, 1].item()))
            height_log.append(float(env.base_pos[0, 2].item()))
            jp_log.append(env.dof_pos[0].detach().cpu().numpy())
            jv_log.append(env.dof_vel[0].detach().cpu().numpy())
            fc = env.link_contact_forces[0, env.feet_link_indices, 2].detach().cpu().numpy()
            # v4 P0 修复 (用户复核 2026-08-13): 必须 > 阈值二值化后再存 —
            # v3 存 raw force 导致 duty/flight/transition/stride/diag 全部失真 (duty=41.98 暴露)
            foot_contact_log.append(fc > CONTACT_FORCE_THRESHOLD)
            torque_log.append(env.torques[0].detach().cpu().numpy())
            # fall 自检: 该段终止 (不 reset 拼接)
            if (abs(roll_log[-1]) > FALL_ROLL_LIMIT or abs(pitch_log[-1]) > FALL_PITCH_LIMIT
                    or height_log[-1] < FALL_HEIGHT_LIMIT):
                fall_at_step = i
                break
            if dones[0]:
                fall_at_step = i
                break
    seg = slice(warmup_steps, None) if not record_all else slice(None)
    if "vx" in locals() and isinstance(vx, dict) and vx.get("_fail_closed"):
        return vx
    vx = np.array(vx_log)[seg]
    # Sol P0: fail-closed — 任何指标源数据含 NaN/Inf 时整段标记失败
    for arr, nm in [(vx_log, "vx"), (roll_log, "roll"), (pitch_log, "pitch")]:
        if not all(np.isfinite(x) for x in arr):
            return {"_fail_closed": True, "reason": f"nonfinite {nm}"}
    vy = np.array(vy_log)[seg]
    roll = np.array(roll_log)[seg]
    pitch = np.array(pitch_log)[seg]
    height = np.array(height_log)[seg]
    jp = np.array(jp_log)[seg]
    jv = np.array(jv_log)[seg]
    contacts = np.array(foot_contact_log)[seg]
    taus = np.array(torque_log)[seg]
    dt_seg = dt
    # 指标 (与 v2.1 相同定义)
    from scipy.signal import savgol_filter
    rmse = float(np.sqrt(np.mean((vx - cmd_speed) ** 2))) if len(vx) else -1.0
    dip = 0
    below = False
    for v in vx:
        if v < DIP_ABS_THRESHOLD:
            if not below:
                dip += 1
            below = True
        else:
            below = False
    jerk_vals = []
    # v4 P0 fix (2026-08-13): fall 过早时截断段可能 < savgol window+阶数 —
    # 段太短 (< 20 步) 时 jerk 无法计算, 该段标记 truncated (falls 已计, 指标不入均值)
    if len(jp) < 20:
        return {"_truncated": True, "fall_at_step": fall_at_step, "reason": "segment too short for jerk"}
    # K3 P0-4 消歧: joint_jerk_filtered = savgol 滤波后位置信号的三阶差分 (jerk)
    # (v=一阶, a=二阶, jrk=三阶); 输入是关节位置 (rad), 输出 rad/s^3
    for j in range(jp.shape[1]):
        q = savgol_filter(jp[:, j], 15, 3)
        v = np.diff(q) / dt_seg
        a = np.diff(v) / dt_seg
        jrk = np.diff(a) / dt_seg
        jerk_vals.append(np.mean(np.abs(jrk)))
    duty = float(np.mean(contacts)) if len(contacts) else -1.0
    all_air = ~np.any(contacts.astype(bool), axis=1)
    flight = float(np.mean(all_air)) if len(all_air) else -1.0
    # v3: cadence 重定义 — contact transition rate vs stride frequency
    if len(contacts) > 10:
        switch_mask = np.abs(np.diff(contacts.astype(int), axis=0)).sum(axis=1) > 0
        contact_transition_rate_hz = float(switch_mask.sum() / 4 / (len(contacts) * dt_seg))
        # stride: 每足 touchdown (0->1) 间隔
        # K3 P0-6: touchdown 事件需去抖 (接触抖动会制造假 touchdown)
        # 定义: 接触 0->1 后持续 >= 2 帧才算一次 touchdown
        touch_events = []
        for leg in range(4):
            c = contacts[:, leg] > 0
            events = []
            i = 0
            while i < len(c) - 2:
                if not c[i] and c[i + 1]:
                    # 潜在 touchdown, 检查后续持续
                    if c[i + 2]:
                        events.append(i + 1)
                        i += 2
                i += 1
            touch_events.append(events)
        stride_interval = []
        for leg in range(4):
            t_idx = np.array(touch_events[leg])
            if len(t_idx) >= 2:
                stride_interval.extend(np.diff(t_idx) * dt_seg)
        stride_frequency_hz = float(1.0 / np.mean(stride_interval)) if stride_interval else -1.0
        d1 = contacts[:, 0] + contacts[:, 2]
        d2 = contacts[:, 1] + contacts[:, 3]
        diag_coord = float(np.corrcoef(d1, d2)[0, 1]) if np.std(d1) > 1e-6 and np.std(d2) > 1e-6 else 0.0
    else:
        contact_transition_rate_hz, stride_frequency_hz, diag_coord = -1.0, -1.0, 0.0
    true_power = None
    if len(taus) and len(jv):
        true_power = float(np.mean(np.sum(np.abs(taus) * np.abs(jv), axis=1)))
    # K3 P1: COT (能耗归一化) — 需质量
    mass = float(env.robot.get_mass()) if hasattr(env.robot, "get_mass") else None
    return {
        "mean_vx": float(np.mean(vx)) if len(vx) else None,
        "mean_vy": float(np.mean(vy)) if len(vy) else None,  # K3 P1: 横向漂移 (sweep 报告)
        "rmse": rmse,
        "std": float(np.std(vx)) if len(vx) else None,
        "dip_count_abs": dip,
        "dip_frac": float(np.mean(vx < DIP_ABS_THRESHOLD)) if len(vx) else None,
        "joint_jerk_filtered": float(np.mean(jerk_vals)) if jerk_vals else None,
        "duty_factor": duty,
        "flight_frac": flight,
        "contact_transition_rate_hz": contact_transition_rate_hz,
        "stride_frequency_hz": stride_frequency_hz,
        "diagonal_coord": diag_coord,
        "fall_at_step": fall_at_step,
        "roll_std": float(np.std(roll)) if len(roll) else None,
        "pitch_std": float(np.std(pitch)) if len(pitch) else None,
        "height_std": float(np.std(height)) if len(height) else None,
        "estimated_mechanical_power_w": true_power,
        "estimated_cost_of_transport": (
            float(true_power / (mass * 9.81 * max(float(np.mean(vx)), 1e-3))) if true_power and mass else None),
    }

def stats(values):
    """repeats 统计: mean±std (口径统一)"""
    if not values:
        return {"mean": None, "std": None, "n": 0}
    v = np.array(values)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}

# 主评估: repeats
print(f"=== {VERSION} ===")
print(f"ckpt: {args.ckpt}")
metrics = {}
runs = []
for r in range(max(args.repeats, 1)):
    ep = run_episode(args.cmd_speed, args.duration)
    if ep.get("_truncated"):
        print(f"  repeat {r}: TRUNCATED (fall_at={ep.get('fall_at_step')}, {ep.get('reason')})")
        runs.append({"fall_at_step": ep.get("fall_at_step")})  # 只计 fall, 无指标
        continue
    runs.append(ep)
    print(f"  repeat {r}: vx={ep['mean_vx']:.3f} fall_at={ep['fall_at_step']}")
if len(runs) > 1 and any(x["fall_at_step"] is not None for x in runs):
    print("  [warn] 存在 fall 段: falls 单独计数, 指标仅统计完整段")

falls_total = sum(1 for x in runs if x["fall_at_step"] is not None)
# K3 P0-2: fall 聚合规则 — fall 段 (截断) 不进入指标均值 (幸存者偏差);
# falls 单独计数, 完整段才贡献 mean±std
clean = [x for x in runs if x["fall_at_step"] is None]
metrics = {
    "A_speed": {k: stats([x[k] for x in clean if x[k] is not None])
                for k in ["mean_vx", "rmse", "std", "dip_count_abs", "dip_frac"]},
    "B_gait": {k: stats([x[k] for x in clean if x[k] is not None])
               for k in ["joint_jerk_filtered", "duty_factor", "flight_frac",
                         "contact_transition_rate_hz", "stride_frequency_hz", "diagonal_coord"]},
    "C_stability": {"falls": falls_total, "n_complete_segments": len(clean),
                    "roll_std": stats([x["roll_std"] for x in clean if x["roll_std"] is not None]),
                    "pitch_std": stats([x["pitch_std"] for x in clean if x["pitch_std"] is not None]),
                    "height_std": stats([x["height_std"] for x in clean if x["height_std"] is not None])},
    "E_energy": {"estimated_mechanical_power_w": stats(
        [x["estimated_mechanical_power_w"] for x in clean if x["estimated_mechanical_power_w"] is not None]),
        "estimated_cost_of_transport": stats(
        [x["estimated_cost_of_transport"] for x in clean if x["estimated_cost_of_transport"] is not None])},
}

# sweep
speed_sweep = {}
if args.sweep:
    for c in [float(x) for x in args.sweep.split(",")]:
        ep = run_episode(c, args.duration)
        if ep.get("_truncated") or ep.get("fall_at_step") is not None:
            # v5: 任何 sweep 档 fall -> incomplete (不产出 RMSE/span 数据)
            speed_sweep[str(c)] = {"fall_at_step": ep.get("fall_at_step"),
                                   "truncated": True, "incomplete": True,
                                   "actual_vx": None, "rmse": None, "mean_vy": None}
            print(f"  sweep cmd={c}: INCOMPLETE fall_at={ep.get('fall_at_step')}")
            continue
        speed_sweep[str(c)] = {"actual_vx": ep["mean_vx"], "mean_vy": ep["mean_vy"],
                               "rmse": ep["rmse"], "fall_at_step": ep["fall_at_step"]}
        print(f"  sweep cmd={c}: vx={ep['mean_vx']:.3f} rmse={ep['rmse']:.3f}")

# push
push = None
if args.push_test:
    print("=== PUSH TEST ===")
    env.reset()
    obs = env.get_observations()
    dt = env.dt
    with torch.no_grad():
        for i in range(int(2.0 / dt)):
            env.commands[:, 0] = args.cmd_speed
            a = policy(obs)
            obs, _, _, _ = env.step(a)
        base_vx, base_vy = [], []
        for i in range(int(1.0 / dt)):
            env.commands[:, 0] = args.cmd_speed
            a = policy(obs)
            obs, _, _, _ = env.step(a)
            base_vx.append(float(env.base_lin_vel[0, 0].item()))
            base_vy.append(float(env.base_lin_vel[0, 1].item()))
        base_mean = float(np.mean(base_vx))
        lateral_peak = 0.0
        # Sol P0: genesis ForceField 是全局加速度场 (非力), 2 m/s^2 横向加速度,
        # 等效于 ~28N (Go2 质量 ~14kg) 仅在质量/质心假设下成立 — 语义: 加速度扰动
        push_log = {"method": "acceleration_field", "acc_mps2": PUSH_ACC_MPS2, "duration_s": 2.0, "axis": "lateral"}
        if force_ok and _push_ff is not None:
            _push_ff.activate()
            for i in range(int(2.0 / dt)):
                env.commands[:, 0] = args.cmd_speed
                a = policy(obs)
                obs, _, _, _ = env.step(a)
                lateral_peak = max(lateral_peak, abs(float(env.base_lin_vel[0, 1].item())))
            _push_ff.deactivate()
        else:
            push_log["invalid"] = True
            for i in range(int(2.0 / dt)):
                env.commands[:, 0] = args.cmd_speed
                a = policy(obs)
                obs, _, _, _ = env.step(a)
        # 恢复: 前向速度回到 80% 基准并保持 0.5s
        recover_time = -1.0
        recovered = False
        above_count = 0
        for i in range(int(5.0 / dt)):
            env.commands[:, 0] = args.cmd_speed
            a = policy(obs)
            obs, _, _, _ = env.step(a)
            v = float(env.base_lin_vel[0, 0].item())
            if v >= 0.8 * base_mean:
                above_count += 1
                if above_count >= int(0.5 / dt):
                    recover_time = (i + 1) * dt - 2.0
                    recovered = True
                    break
            else:
                above_count = 0
        push = {"base_vx": base_mean, "lateral_peak_mps": lateral_peak,
                "recovered": recovered, "recover_time_s": recover_time, **push_log}
        print(f"  push: lateral_peak={lateral_peak:.2f} recovered={recovered} t={recover_time:.2f}s")

# meta (v3: 拆分 train/eval SHA)
model_sha = sha256_file(args.ckpt)
eval_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd="/home/che/dev/kine2go-research",
                             capture_output=True, text=True).stdout.strip()
train_commit = "unknown"
try:
    with open(f"{log_dir}/train_git_commit.txt") as f:
        train_commit = f.read().strip()
except FileNotFoundError:
    pass
meta = {
    "version": VERSION,
    "ckpt": args.ckpt,
    "model_sha256": model_sha[:16],
    "train_git_commit": train_commit,
    "eval_git_commit": eval_commit,
    "cfg_hash": hashlib.sha256(pickle.dumps([env_cfg, obs_cfg, reward_cfg, command_cfg, policy_cfg])).hexdigest()[:16],
    "motion_sha256": sha256_file(f"{log_dir}/motion.npy")[:16],
    "seed": args.seed, "repeats": len(runs), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "force_api": force_api,
    "evaluator_file_sha256": sha256_file("/home/che/dev/kine2go-research/evaluation/quant_eval_v5.py")[:16],
}
result = {"meta": meta, "cmd_speed": args.cmd_speed, "metrics": metrics,
          "speed_sweep": speed_sweep, "push": push}
if args.out:
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"saved -> {args.out}")
else:
    print(json.dumps(result, indent=1))
