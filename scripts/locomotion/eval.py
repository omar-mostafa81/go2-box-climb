"""Evaluate a walking checkpoint: speed tracking and gait quality.

Runs 64 robots (no random pushes, no sensor noise) for 8 s at each fixed command
and measures, after a 2 s settling time (the "walk>stand" cases first walk forward at
0.3 m/s for 3 s, then stand; they check that the robot really stops stepping):

    actual speed      mean forward / sideways / turning velocity vs. the command
    falls             fraction of robots that fell (trunk or head touched the ground)
    height            mean base height [m]
    duty              fraction of time a foot is on the ground (trot ~0.5-0.6; shuffling ~0.8+)
    steps/s           touchdowns per second per foot
    lift              mean peak foot-center height per swing [m] (0.022 m = foot on the ground)
    trot sync         fraction of time diagonal feet (FL+RR, FR+RL) are in the same contact state
    hip dev           mean |hip angle - default| [rad] (legs splayed sideways when large)
    jvel              mean |joint velocity| [rad/s] (standing still: ~0)
    rise              how high each foot rises above its own lowest point, mean over feet [mm]
                      (standing still: ~0; it catches feet tapping or lifting in place)

Results are printed and saved to  <run folder>/eval/<checkpoint name>_gait.json

Example:
    python scripts/locomotion/eval.py --stage a --checkpoint logs/rsl_rl/go2_walk/<run>/model_999.pt
"""

import argparse

from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Evaluate a walking checkpoint (tracking and gait).")
parser.add_argument("--stage", choices=("a", "b"), required=True, help="Which stage's environment settings to use.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pt) to evaluate.")
parser.add_argument("--num_envs", type=int, default=64)
add_launch_args(parser, default_mode="headless")
args = parser.parse_args()
simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402
from go2_box_climb.envs.locomotion.agents import WalkPPORunnerCfg, WalkPushPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.locomotion.walk_env_cfg import WalkEnvCfg, WalkPushEnvCfg  # noqa: E402
from go2_box_climb.skills.locomotion import DEFAULT_BODY_HEIGHT  # noqa: E402

# (name, vx [m/s], vy [m/s], wz [rad/s], body height h [m]; h is used by stage B policies only)
H0 = DEFAULT_BODY_HEIGHT
CASES = [
    ("stand", 0.0, 0.0, 0.0, H0),
    ("fwd 0.3", 0.3, 0.0, 0.0, H0),
    ("fwd 0.6", 0.6, 0.0, 0.0, H0),
    ("fwd 1.0", 1.0, 0.0, 0.0, H0),
    ("back 0.5", -0.5, 0.0, 0.0, H0),
    ("side 0.4", 0.0, 0.4, 0.0, H0),
    ("turn 0.8", 0.3, 0.0, 0.8, H0),
    ("low fwd .3", 0.3, 0.0, 0.0, 0.27),  # pushing posture
    ("low stand", 0.0, 0.0, 0.0, 0.27),
    # walk forward at 0.3 m/s for WALK_FIRST_STEPS, then stand: does the robot really stop stepping?
    ("walk>stand", 0.0, 0.0, 0.0, H0),
    ("walk>stand low", 0.0, 0.0, 0.0, 0.27),
]
WALK_FIRST_STEPS = 150  # 3 s of walking before the "walk>stand" cases (inside their longer settle time)
SETTLE_STEPS, MEASURE_STEPS = 100, 300  # 2 s settle, 6 s measure (policy steps of 0.02 s)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def main():
    env_cfg, agent_cfg = (WalkEnvCfg(), WalkPPORunnerCfg()) if args.stage == "a" else (WalkPushEnvCfg(), WalkPushPPORunnerCfg())
    env_cfg.play_mode()  # no random pushes, no sensor noise
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = 1e6  # no time-outs during a test
    env_cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)  # we set the command ourselves
    # No "standing" robots: Isaac Lab forces their command to zero every step, which
    # would mix robots that stand into every test case and lower the measured speeds.
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.events.reset_base.params["velocity_range"] = {}
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot, sensor = u.scene["robot"], u.scene["contact_forces"]
    cmd_term = u.command_manager.get_term("base_velocity")
    height_term = u.command_manager.get_term("body_height") if args.stage == "b" else None
    feet = [robot.body_names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
    feet_s = [sensor.body_names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
    hips = [robot.joint_names.index(n) for n in C.JOINT_NAMES[:4]]
    q0 = _t(robot.data.default_joint_pos)[:, hips]
    n, dev = args.num_envs, u.device

    results = {}
    print(f"\n{'case':14s} {'cmd (vx,vy,wz)':>16s} {'actual':>18s} {'falls':>6s} {'height':>7s} {'duty':>5s} "
          f"{'steps/s':>7s} {'lift':>6s} {'trot':>5s} {'hipdev':>6s} {'jvel':>5s} {'rise':>6s}")
    for name, vx, vy, wz, hcmd in CASES:
        if height_term is None and hcmd != H0:
            continue  # stage A has no height command
        u.reset()
        obs = env.get_observations()
        command = torch.tensor([vx, vy, wz], device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        vel, height, hipdev, contacts, peaks, jvel = [], [], [], [], [], []
        foot_min = torch.full((n, 4), 1e9, device=dev)
        foot_max = torch.full((n, 4), -1e9, device=dev)
        running_peak = torch.zeros(n, 4, device=dev)
        walk_first = name.startswith("walk>stand")
        settle = WALK_FIRST_STEPS + SETTLE_STEPS if walk_first else SETTLE_STEPS
        for k in range(settle + MEASURE_STEPS):
            walking = walk_first and k < WALK_FIRST_STEPS
            cmd_term.vel_command_b[:] = torch.tensor([0.3, 0.0, 0.0], device=dev) if walking else command
            if height_term is not None:
                height_term.height[:] = hcmd
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            if k < settle:
                continue
            fell |= dones.bool()
            lin_b, ang_b = _t(robot.data.root_lin_vel_b), _t(robot.data.root_ang_vel_b)
            vel.append(torch.stack([lin_b[:, 0], lin_b[:, 1], ang_b[:, 2]], dim=1))
            height.append(_t(robot.data.root_pos_w)[:, 2])
            hipdev.append((_t(robot.data.joint_pos)[:, hips] - q0).abs().mean(dim=1))
            c = _t(sensor.data.net_forces_w)[:, feet_s, 2].abs() > 1.0
            fz = _t(robot.data.body_pos_w)[:, feet, 2]
            if contacts:  # a touchdown ends a swing: record its peak height
                touchdown = c & ~contacts[-1]
                peaks.append(running_peak[touchdown])
                running_peak[touchdown] = 0.0
            running_peak = torch.where(c, running_peak, torch.maximum(running_peak, fz))
            contacts.append(c)
            jvel.append(_t(robot.data.joint_vel).abs().mean(dim=1))
            foot_min, foot_max = torch.minimum(foot_min, fz), torch.maximum(foot_max, fz)
        ok = ~fell
        V = torch.stack(vel)[:, ok].mean(dim=(0, 1))
        Cn = torch.stack(contacts)[:, ok].float()
        touchdowns = (Cn[1:] - Cn[:-1] > 0.5).float().sum(0).mean() / (MEASURE_STEPS * C.POLICY_DT)
        sync = ((Cn[..., 0] == Cn[..., 3]).float().mean() + (Cn[..., 1] == Cn[..., 2]).float().mean()) / 2
        P = torch.cat(peaks) if peaks else torch.zeros(0, device=dev)
        r = {
            "command": [vx, vy, wz],
            "height_command": hcmd if height_term is not None else None,
            "actual": [round(v, 3) for v in V.tolist()],
            "fall_rate": round(fell.float().mean().item(), 3),
            "base_height": round(torch.stack(height)[:, ok].mean().item(), 3),
            "duty": round(Cn.mean().item(), 3),
            "steps_per_s": round(touchdowns.item(), 2),
            "swing_peak_height": round(P.mean().item(), 3) if len(P) else None,
            "trot_sync": round(sync.item(), 3),
            "hip_dev": round(torch.stack(hipdev)[:, ok].mean().item(), 3),
            "joint_speed": round(torch.stack(jvel)[:, ok].mean().item(), 3),
            "foot_rise_mm": round(1000 * (foot_max - foot_min)[ok].mean().item(), 1),
        }
        results[name] = r
        lift = f"{r['swing_peak_height']:.3f}" if r["swing_peak_height"] is not None else "  -  "
        print(f"{name:14s} {str((vx, vy, wz)):>16s} {str(tuple(round(v, 2) for v in r['actual'])):>18s} "
              f"{r['fall_rate']:6.2f} {r['base_height']:7.3f} {r['duty']:5.2f} {r['steps_per_s']:7.2f} "
              f"{lift:>6s} {r['trot_sync']:5.2f} {r['hip_dev']:6.2f} {r['joint_speed']:5.2f} {r['foot_rise_mm']:6.1f}")

    out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(args.checkpoint).replace(".pt", "_gait.json"))
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "stage": args.stage, "cases": results}, f, indent=2)
    print(f"\n[INFO] Saved: {out}")
    env.close()


if __name__ == "__main__":
    main()
    close_app(simulation_app)
