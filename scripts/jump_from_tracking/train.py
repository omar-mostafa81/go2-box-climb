"""Train the TO-guided jump policy (one per jump) with PPO (rsl_rl).

Each episode is one jump attempt from random start conditions (common/constants.py),
guided by the jump's reference (references/, from the TO part). Before the real
references exist, train on a placeholder to check that everything works:

    python scripts/jump_from_tracking/make_placeholder_reference.py
    python scripts/jump_from_tracking/train.py --jump ground_to_box --max_iterations 20 \
        --reference outputs/jump_from_tracking/placeholder_jump_ground_to_box.npz

Real training (no window), with the final reference in references/:
    python scripts/jump_from_tracking/train.py --jump ground_to_box
    python scripts/jump_from_tracking/train.py --jump box_to_platform

Output: logs/rsl_rl/go2_jump_track_<jump>/<date_time>[_<run_name>]/: checkpoints
model_<iteration>.pt every 50 iterations, TensorBoard files, params/ (the exact settings
used, plus a copy of the reference as params/reference.npz) and plots/training.png,
plots/rewards.png, plots/jump.png (redrawn every 30 s). Ctrl+C stops training and saves
a checkpoint.
"""

import argparse

from go2_box_climb.skills.jump.jump_from_tracking.reference import JUMPS, reference_file
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Train the TO-guided jump policy (PPO).")
parser.add_argument("--jump", choices=JUMPS, required=True)
parser.add_argument(
    "--reference", type=str, default=None, help="Reference .npz (default: the final one in references/)."
)
parser.add_argument("--num_envs", type=int, default=4096, help="Robots simulated in parallel.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default: 1500).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load", type=str, default=None, help="Checkpoint (.pt) to continue from.")
parser.add_argument("--run_name", type=str, default="", help="Optional suffix for the run folder name.")
add_launch_args(parser, default_mode="headless")
args = parser.parse_args()
reference = reference_file(args.jump, args.reference)  # fail fast, before Isaac Sim starts

simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import importlib.metadata  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2_box_climb.envs.jump_from_tracking.agents import JumpTrackPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.jump_env_cfg import JumpTrackEnvCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.plots import JumpLivePlotter  # noqa: E402
from go2_box_climb.utils.runs import RSL_RL_LOGS  # noqa: E402


def main():
    env_cfg = resolve_presets(JumpTrackEnvCfg().configure(args.jump, str(reference)))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    agent_cfg = JumpTrackPPORunnerCfg().for_jump(args.jump)
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.run_name = args.run_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    log_dir = RSL_RL_LOGS / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        log_dir = log_dir.with_name(f"{log_dir.name}_{args.run_name}")
    print(f"[INFO] Jump {args.jump}: {args.num_envs} robots, {agent_cfg.max_iterations} iterations")
    print(f"[INFO] Reference: {reference}")
    print(f"[INFO] Logging to: {log_dir}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args.load:
        print(f"[INFO] Starting from checkpoint: {args.load}")
        runner.load(args.load)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    shutil.copyfile(reference, log_dir / "params" / "reference.npz")  # the exact reference this run used
    with open(log_dir / "params" / "reference.txt", "w") as f:
        f.write(f"{reference}\n")
    if args.load:
        with open(log_dir / "params" / "started_from.txt", "w") as f:
            f.write(f"{args.load}\n")

    plotter = JumpLivePlotter(log_dir).start()
    print(f"[INFO] Training plots (redrawn every {plotter.every_s:.0f} s): {log_dir / 'plots'}")
    start = time.time()
    try:
        # Every episode must start at reference time 0 (the robot is placed in the reference's
        # first pose), so episodes are NOT started at random lengths.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"[INFO] Training finished in {(time.time() - start) / 60:.1f} min. Run folder: {log_dir}")
    except KeyboardInterrupt:
        path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        runner.save(path)
        print(f"\n[INFO] Stopped by user after {(time.time() - start) / 60:.1f} min. Saved: {path}")
    plotter.stop()  # final redraw
    env.close()


if __name__ == "__main__":
    main()
    close_app(simulation_app)
