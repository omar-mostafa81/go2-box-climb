"""Watch the jump-tracking policy (or the bare reference), and export the policy.

Runs 16 robots, each repeating jump attempts from random start conditions, and prints
how the attempts ended. Two ways to drive the robots:

    --checkpoint <run>/model_<N>.pt   the trained policy; it is also exported next to the
                                       checkpoint: <run>/exported/policy.pt (TorchScript) and
                                       policy.onnx
    --open_loop                       no policy: the PD motors simply follow the reference's
                                       joint angles. Shows what the reference alone does (and
                                       checks the whole environment before any training). It
                                       also prints how far the feet computed by kinematics.py
                                       are from the simulator's feet (should be ~0).

Examples (from the repository root):
    python scripts/jump_from_tracking/play.py --jump ground_to_box --open_loop --mode gui \
        --reference outputs/jump_from_tracking/placeholder_jump_ground_to_box.npz
    python scripts/jump_from_tracking/play.py --jump ground_to_box --mode gui \
        --checkpoint logs/rsl_rl/go2_jump_track_ground_to_box/<run>/model_1499.pt

    # No window (e.g. on the cluster): record 6 s to videos/play_jump_<jump>_<date_time>.mp4
    python scripts/jump_from_tracking/play.py --jump ground_to_box --mode headless --video --duration 6 \
        --checkpoint logs/rsl_rl/go2_jump_track_ground_to_box/<run>/model_1499.pt
"""

import argparse

from go2_box_climb.skills.jump.jump_from_tracking.reference import JUMPS, reference_file
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Play (and export) the jump-tracking policy, or replay the reference.")
parser.add_argument("--jump", choices=JUMPS, required=True)
parser.add_argument("--reference", type=str, default=None, help="Reference .npz (default: references/).")
who = parser.add_mutually_exclusive_group(required=True)
who.add_argument("--checkpoint", type=str, help="Checkpoint (.pt) to load.")
who.add_argument("--open_loop", action="store_true", help="No policy: follow the reference's joint angles.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (default: until closed).")
add_launch_args(parser, default_mode="gui")
args = parser.parse_args()
if args.video and args.mode == "headless" and args.duration is None:
    parser.error("--video in headless mode needs --duration (how many seconds to record)")
reference = reference_file(args.jump, args.reference)  # fail fast, before Isaac Sim starts

simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import importlib.metadata  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.agents import JumpTrackPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.jump_env_cfg import COMMAND, JumpTrackEnvCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.mdp import FEET, open_loop_actions  # noqa: E402
from go2_box_climb.skills.jump.jump_from_tracking.kinematics import foot_positions  # noqa: E402
from go2_box_climb.utils.video import VideoRecorder  # noqa: E402

OUTCOMES = ("success", "fall_contact", "fall_tilt", "time_out")


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def main():
    env_cfg = JumpTrackEnvCfg().configure(args.jump, str(reference))
    env_cfg.play_mode()
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    u = env.unwrapped
    print(f"[INFO] Jump {args.jump}, reference {reference}")

    if args.checkpoint:
        agent_cfg = JumpTrackPPORunnerCfg().for_jump(args.jump)
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading: {args.checkpoint}")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=u.device)
        export_dir = os.path.join(os.path.dirname(args.checkpoint), "exported")
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_dir}")
    else:
        print("[INFO] Open loop: the motors follow the reference's joint angles (no policy).")

        def policy(_obs):
            return open_loop_actions(u, COMMAND)

    robot = u.scene["robot"]
    feet_ids, _ = robot.find_bodies(list(FEET), preserve_order=True)
    counts = dict.fromkeys(OUTCOMES, 0)
    fk_error = 0.0
    obs = env.get_observations()
    sim_time = 0.0
    name = f"play_jump_{args.jump}" + ("_open_loop" if args.open_loop else "")
    video = VideoRecorder(name, fps=1 / C.POLICY_DT, enabled=args.video)  # 1 frame per step
    print("[INFO] Playing. Close the window (or press Ctrl+C) to stop.")
    while simulation_app.is_running():
        if args.duration is not None and sim_time >= args.duration:
            break
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        video.capture()
        sim_time += C.POLICY_DT
        done = dones.bool()
        just_reset = done.clone()
        if done.any():
            for k in OUTCOMES:  # one outcome per attempt, in this priority order
                hit = done & u.termination_manager.get_term(k)
                counts[k] += int(hit.sum())
                done &= ~hit
        if args.open_loop and not just_reset.all():  # kinematics.py vs the simulator's own feet
            keep = ~just_reset  # robots placed by a reset this step may not have updated body poses yet
            p = _t(robot.data.root_pos_w)[keep]
            fk = foot_positions(p, _t(robot.data.root_quat_w)[keep], _t(robot.data.joint_pos)[keep])
            sim_feet = _t(robot.data.body_pos_w)[keep][:, feet_ids, :]
            fk_error = max(fk_error, float((fk - sim_feet).norm(dim=-1).max()))
    video.close()

    total = sum(counts.values())
    print(f"\n[INFO] {total} finished attempts in {sim_time:.1f} s: "
          + ", ".join(f"{k} {v}" + (f" ({v / total:.0%})" if total else "") for k, v in counts.items()))
    if args.open_loop:
        print(f"[INFO] Largest distance between kinematics.py feet and simulator feet: {1000 * fk_error:.1f} mm")
    env.close()


if __name__ == "__main__":
    main()
    close_app(simulation_app)
