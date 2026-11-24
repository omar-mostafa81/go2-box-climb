"""Run and evaluate the MPPI push skill in Isaac Sim.

The robot starts at the origin, the box at BOX_START_POS. PushMPPISkill plans walking
commands with MPPI and the walking policy turns them into joint targets, exactly as in
the full task. Each episode ends on SUCCESS or FAILURE, with an end reason (success,
box_off_center, no_contact, fell, timeout; see push_skill.py); then the world is reset.

--scenario A|B|C|D|E starts the box at a named pose (scripts/push_mppi/scenarios.py) by
moving it there after each reset. --horizon / --samples / --timeout override the planner
horizon, the rollouts per replan and the episode timeout (defaults: MPPIConfig / PushConfig).

Output (ignored by git): logs/push_mppi/<YYYY-MM-DD_HH-MM-SS>/
    steps.csv      one row per policy step: time, robot/box pose, command, cost terms, and the box
                   side the head touches (front/left/back/right in the box frame, or none; from
                   geometry, for logging only)
    episodes.csv   one row per episode: result, end reason, time, final box gap / y / yaw
    push.png       box gap / y / yaw, commands and the largest cost terms over time (episode 0)

Examples (from the repository root):
    python scripts/push_mppi/run_push.py --mode stream                       # watch one push
    python scripts/push_mppi/run_push.py --mode headless --episodes 10       # evaluate
    python scripts/push_mppi/run_push.py --mode stream --scenario E --horizon 40 --episodes 30
"""

import argparse

from go2_box_climb.envs.scene import SCENES
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Run / evaluate the MPPI push skill.")
parser.add_argument("--scene", choices=SCENES, default="basic", help="World to load.")
parser.add_argument("--episodes", type=int, default=1, help="Number of pushes to run.")
parser.add_argument("--timeout", type=float, default=None,
                    help="Seconds before an episode counts as failed (default: PushConfig.timeout).")  # fmt: skip
parser.add_argument("--contact_margin", type=float, default=0.05, help="Head contact point beyond the head frame [m].")
parser.add_argument("--samples", type=int, default=None, help="MPPI rollouts per replan K (default: MPPIConfig).")
parser.add_argument("--horizon", type=int, default=None, help="MPPI horizon in steps of 0.1 s (default: MPPIConfig).")
parser.add_argument("--scenario", choices=("A", "B", "C", "D", "E"), default=None,
                    help="Box start pose from scripts/push_mppi/scenarios.py (default: the normal start).")  # fmt: skip
add_launch_args(parser, default_mode="headless")
args = parser.parse_args()
simulation_app = launch(args)

import csv  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402
from go2_box_climb.common.skill import SkillStatus  # noqa: E402
from go2_box_climb.common.state import TaskState  # noqa: E402
from go2_box_climb.envs.scene import get_scene_cfg_class  # noqa: E402
from go2_box_climb.skills.push_mppi import MPPIConfig, PushConfig, PushMPPISkill, box_errors_planar, planar_state  # noqa: E402
from go2_box_climb.skills.push_mppi.model import SIDE_NAMES, nearest_side  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402  (scripts/push_mppi/scenarios.py, next to this script)
from go2_box_climb.utils.runs import REPO_ROOT  # noqa: E402

TERMS = ("box_pos", "box_yaw", "aim", "approach", "heading", "box_speed", "effort", "smooth", "corner",
         "platform", "terminal")  # fmt: skip


def _np(x):
    return (x.torch if hasattr(x, "torch") else x)[0].detach().cpu().numpy().astype(float)


def plot_episode(steps_csv, png, episode: int = 0):
    """Plot one episode from steps.csv: box state, commands, cost terms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(steps_csv) as f:
        rows = [r for r in csv.DictReader(f) if int(r["episode"]) == episode]
    if not rows:
        return
    col = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "side"}
    t = col["t"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, 100 * (C.PLATFORM_FRONT_X - C.BOX_SIZE[0] / 2 - col["bx"]), label="box gap to platform [cm]")
    ax[0].plot(t, 100 * col["by"], label="box y [cm]")
    ax[0].plot(t, np.degrees(col["byaw"]), label="box yaw [deg]")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[1].plot(t, col["vx"], label="vx [m/s]")
    ax[1].plot(t, col["vy"], label="vy [m/s]")
    ax[1].plot(t, col["wz"], label="wz [rad/s]")
    ax[1].plot(t, col["h"], label="body height h [m]")
    for k in sorted(TERMS, key=lambda k: -col[k].max())[:6]:
        ax[2].plot(t, col[k], label=k)
    ax[2].set_yscale("symlog", linthresh=1.0)
    ax[0].set_title("Box state"), ax[1].set_title("Command to the walking policy")
    ax[2].set_title("MPPI cost of the followed plan (largest terms)"), ax[2].set_xlabel("time [s]")
    for a in ax:
        a.legend(fontsize=8, loc="upper right"), a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main():
    out_dir = REPO_ROOT / "logs" / "push_mppi" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=C.SIM_DT, device=args.device))
    scene = InteractiveScene(get_scene_cfg_class(args.scene)(num_envs=1))
    sim.reset()
    robot, box = scene["robot"], scene["box"]
    assert list(robot.joint_names) == list(C.JOINT_NAMES), f"Joint order mismatch: {robot.joint_names}"
    origin = scene.env_origins[0].cpu().numpy()
    default_q = torch.tensor(C.DEFAULT_JOINT_POS, device=sim.device, dtype=torch.float32)

    mppi = MPPIConfig()
    mppi.num_samples = args.samples or mppi.num_samples
    mppi.horizon = args.horizon or mppi.horizon
    cfg = PushConfig(mppi=mppi, contact_margin=args.contact_margin)
    cfg.timeout = args.timeout or cfg.timeout
    skill = PushMPPISkill(config=cfg)
    print(f"[INFO] Scenario {args.scenario or 'normal start'}: horizon {mppi.horizon} steps ({mppi.horizon * mppi.dt:.1f} s), "
          f"samples {mppi.num_samples}, timeout {cfg.timeout:.0f} s. Head contact point "
          f"{skill.planner.params.head_x:.3f} m ahead of the base. Output: {out_dir}")  # fmt: skip

    def touched_side(x) -> str:
        """Box side the head touches (box frame), from geometry; for logging only."""
        s = nearest_side(x[None], skill.planner.params)
        return SIDE_NAMES[s["side"][0]] if s["gap"][0] <= cfg.contact_gap else "none"

    def read_state(t: float) -> TaskState:
        return TaskState(
            base_pos=_np(robot.data.root_pos_w) - origin,
            base_quat=_np(robot.data.root_quat_w),
            base_lin_vel=_np(robot.data.root_lin_vel_w),
            base_ang_vel=_np(robot.data.root_ang_vel_w),
            joint_pos=_np(robot.data.joint_pos),
            joint_vel=_np(robot.data.joint_vel),
            box_pos=_np(box.data.root_pos_w) - origin,
            box_quat=_np(box.data.root_quat_w),
            box_lin_vel=_np(box.data.root_lin_vel_w),
            box_ang_vel=_np(box.data.root_ang_vel_w),
            time=t,
        )

    def reset_world():
        for asset in (robot, box):
            pose = (asset.data.default_root_pose.torch if hasattr(asset.data.default_root_pose, "torch")
                    else asset.data.default_root_pose).clone()  # fmt: skip
            pose[:, :3] += scene.env_origins
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=sim.device))
        robot.write_joint_state_to_sim_index(
            position=default_q.unsqueeze(0), velocity=torch.zeros(1, C.NUM_JOINTS, device=sim.device)
        )
        if args.scenario:  # move the box to the scenario pose, resting on the floor (quaternion x, y, z, w)
            bx, by, byaw = SCENARIOS[args.scenario]
            half = np.deg2rad(byaw) / 2
            box_pose = torch.tensor([[bx, by, C.BOX_START_POS[2], 0.0, 0.0, np.sin(half), np.cos(half)]],
                                    device=sim.device, dtype=torch.float32)  # fmt: skip
            box_pose[:, :3] += scene.env_origins
            box.write_root_pose_to_sim_index(root_pose=box_pose)
        scene.reset()
        # let the robot settle on its feet before the skill starts
        for _ in range(C.DECIMATION * 10):
            robot.set_joint_position_target_index(target=default_q.unsqueeze(0))
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(C.SIM_DT)

    step_file = open(out_dir / "steps.csv", "w", newline="")
    steps = csv.writer(step_file)
    steps.writerow(["episode", "t", "rx", "ry", "ryaw", "bx", "by", "byaw", "vx", "vy", "wz", "h", *TERMS, "side"])
    episodes = []

    for ep in range(args.episodes):
        if not simulation_app.is_running():
            break
        reset_world()
        state = read_state(0.0)
        skill.reset(state)
        status, t = SkillStatus.RUNNING, 0.0
        touches = {n: 0 for n in SIDE_NAMES}  # policy steps with the head on each box side
        while simulation_app.is_running():
            t0 = time.time()
            state = read_state(t)
            status = skill.status(state)
            if status != SkillStatus.RUNNING:
                break
            action = skill.act(state)
            target = default_q + C.ACTION_SCALE * torch.tensor(action, device=sim.device, dtype=torch.float32)
            robot.set_joint_position_target_index(target=target.unsqueeze(0))
            for i in range(C.DECIMATION):
                scene.write_data_to_sim()
                sim.step(render=(args.mode != "headless" and i == C.DECIMATION - 1))
                scene.update(C.SIM_DT)
            t += C.POLICY_DT

            x = planar_state(state)
            terms = skill.last_cost_terms
            side = touched_side(x)
            touches[side] = touches.get(side, 0) + 1
            steps.writerow([ep, f"{t:.2f}", *np.round(x, 4), *np.round(skill.command, 3),
                            *(round(terms.get(k, 0.0), 3) for k in TERMS), side])  # fmt: skip
            if round(t / C.POLICY_DT) % 25 == 0:  # status line every 0.5 s
                gap, dy, dyaw = box_errors_planar(x)
                top = sorted(terms.items(), key=lambda kv: -kv[1])[:3]
                print(f"\r[ep {ep}] t {t:5.1f}s cmd {np.round(skill.command, 2).tolist()} | box gap {100 * gap:5.1f} cm "
                      f"y {100 * dy:+5.1f} cm yaw {np.degrees(dyaw):+5.1f} deg | top cost "
                      + ", ".join(f"{k} {v:.1f}" for k, v in top) + "   ", end="", flush=True)  # fmt: skip
            if args.mode != "headless":
                p = state.base_pos + origin
                sim.set_camera_view(eye=(p[0] - 1.2, p[1] - 1.8, 1.2), target=(p[0] + 0.6, p[1], 0.2))
                time.sleep(max(0.0, C.POLICY_DT - (time.time() - t0)))  # real time

        gap, dy, dyaw = box_errors_planar(planar_state(state))  # yaw mod 90 deg
        sides = " ".join(f"{n}:{k}" for n, k in touches.items() if k and n != "none") or "none"
        episodes.append([ep, status.value, skill.end_reason, round(t, 2), round(100 * gap, 1), round(100 * dy, 1),
                         round(float(np.degrees(dyaw)), 1), sides])  # fmt: skip
        print(f"\n[ep {ep}] {status.value.upper()} ({skill.end_reason}) after {t:.1f} s: box gap {100 * gap:.1f} cm, "
              f"y {100 * dy:+.1f} cm, yaw {np.degrees(dyaw):+.1f} deg | sides touched (steps): {sides}")  # fmt: skip

    step_file.close()
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "result", "end_reason", "time_s", "gap_cm", "y_cm", "yaw_deg", "sides_touched"])
        w.writerows(episodes)
    if episodes:
        ok = [e for e in episodes if e[1] == "success"]
        plot_episode(out_dir / "steps.csv", out_dir / "push.png")
        print(f"[INFO] Success {len(ok)}/{len(episodes)}"
              + (f", mean time {np.mean([e[3] for e in ok]):.1f} s" if ok else "") + f". Logs: {out_dir}")  # fmt: skip


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped (Ctrl+C).")
    close_app(simulation_app)
