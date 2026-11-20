"""Train the Go2 walking policy with PPO (rsl_rl).

Stage A trains normal walking from scratch. Stage B continues from a stage A
checkpoint and adds a steady push on the head that ramps up to +-40 N.

Examples (from the repository root):
    # Watch a single robot learn (slow; only to check that everything looks right)
    python scripts/locomotion/train.py --stage a --num_envs 1 --mode stream

    # Real training, no window
    python scripts/locomotion/train.py --stage a --mode headless
    python scripts/locomotion/train.py --stage b --mode headless --load logs/rsl_rl/go2_walk/<run>/model_999.pt

    # Continue an interrupted run of the same stage
    python scripts/locomotion/train.py --stage a --mode headless --load logs/rsl_rl/go2_walk/<run>/model_450.pt

Output: logs/rsl_rl/go2_walk/<date_time>/ (stage A) or logs/rsl_rl/go2_walk_push/<date_time>/
(stage B): checkpoints model_<iteration>.pt every 50 iterations, TensorBoard files,
params/ (the exact settings used) and plots/training.png + plots/rewards.png (redrawn
every 30 s while training, so you can watch them). Ctrl+C stops training and saves a checkpoint.
"""

import argparse

from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Train the Go2 walking policy (PPO).")
parser.add_argument("--stage", choices=("a", "b"), required=True, help="a: normal walking | b: + steady push")
parser.add_argument("--num_envs", type=int, default=4096, help="Robots simulated in parallel.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default: 1000 for a, 1500 for b).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--load", type=str, default=None, help="Checkpoint (.pt) to start from. Required for stage b (a stage A checkpoint)."
)
parser.add_argument("--run_name", type=str, default="", help="Optional suffix for the run folder name.")
add_launch_args(parser, default_mode="headless")
args = parser.parse_args()
if args.stage == "b" and not args.load:
    parser.error("stage b continues from stage A: pass --load logs/rsl_rl/go2_walk/<run>/model_<N>.pt")
simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import importlib.metadata  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2_box_climb.envs.locomotion.agents import WalkPPORunnerCfg, WalkPushPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.locomotion.checkpoint import widen_checkpoint_inputs  # noqa: E402
from go2_box_climb.envs.locomotion.walk_env_cfg import WalkEnvCfg, WalkPushEnvCfg  # noqa: E402
from go2_box_climb.utils.runs import RSL_RL_LOGS  # noqa: E402
from go2_box_climb.utils.training_plots import LivePlotter  # noqa: E402


def main():
    if args.stage == "a":
        env_cfg, agent_cfg = WalkEnvCfg(), WalkPPORunnerCfg()
    else:
        env_cfg, agent_cfg = WalkPushEnvCfg(), WalkPushPPORunnerCfg()
    load_path = args.load

    # Pick the default physics backend (PhysX) for Isaac Lab's multi-backend options
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.run_name = args.run_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    log_dir = RSL_RL_LOGS / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        log_dir = log_dir.with_name(f"{log_dir.name}_{args.run_name}")
    print(f"[INFO] Stage {args.stage.upper()}: {args.num_envs} robots, {agent_cfg.max_iterations} iterations")
    print(f"[INFO] Logging to: {log_dir}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if load_path:
        print(f"[INFO] Starting from checkpoint: {load_path}")
        obs_dim = env.get_observations()["policy"].shape[1]
        widened = log_dir / "params" / "start_checkpoint_widened.pt"
        if widen_checkpoint_inputs(load_path, obs_dim, widened):
            # stage A (48 inputs) -> stage B (49): new input starts with zero weight; fresh optimizer
            print(f"[INFO] Checkpoint widened to {obs_dim} inputs (new inputs start at zero weight)")
            runner.load(str(widened), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True})
        else:
            runner.load(load_path)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if load_path:
        with open(os.path.join(log_dir, "params", "started_from.txt"), "w") as f:
            f.write(f"{load_path}\n")

    plotter = LivePlotter(log_dir).start()
    print(f"[INFO] Training plots (redrawn every {plotter.every_s:.0f} s): {log_dir / 'plots'}")
    start = time.time()
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
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
