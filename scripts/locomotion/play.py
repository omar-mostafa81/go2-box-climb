"""Watch a trained walking policy, and export it for deployment.

Runs the policy in the training world of the given stage (flat floor, 16 robots, random velocity
commands shown as arrows: green = command, blue = actual velocity), without random
pushes. It also exports the network next to the checkpoint:
    <run>/exported/policy.pt    (TorchScript, used by skills/locomotion)
    <run>/exported/policy.onnx

Examples (from the repository root):
    python scripts/locomotion/play.py --stage a --mode gui --checkpoint logs/rsl_rl/go2_walk/<run>/model_999.pt
    python scripts/locomotion/play.py --stage b --mode stream --checkpoint logs/rsl_rl/go2_walk_push/<run>/model_1799.pt

    # No window (e.g. on the cluster): record 10 s to videos/play_stage_a_<date_time>.mp4
    python scripts/locomotion/play.py --stage a --mode headless --video --duration 10 --checkpoint logs/rsl_rl/go2_walk/<run>/model_999.pt
"""

import argparse

from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Play (and export) a trained walking policy.")
parser.add_argument("--stage", choices=("a", "b"), required=True, help="Which stage's environment settings to use.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (default: until closed).")
add_launch_args(parser, default_mode="gui")
args = parser.parse_args()
if args.video and args.mode == "headless" and args.duration is None:
    parser.error("--video in headless mode needs --duration (how many seconds to record)")
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
from go2_box_climb.envs.locomotion.agents import WalkPPORunnerCfg, WalkPushPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.locomotion.walk_env_cfg import WalkEnvCfg, WalkPushEnvCfg  # noqa: E402
from go2_box_climb.utils.video import VideoRecorder  # noqa: E402


def main():
    env_cfg, agent_cfg = (WalkEnvCfg(), WalkPPORunnerCfg()) if args.stage == "a" else (WalkPushEnvCfg(), WalkPushPPORunnerCfg())
    checkpoint = args.checkpoint
    print(f"[INFO] Loading: {checkpoint}")

    env_cfg.play_mode()  # no random pushes, no sensor noise
    env_cfg = resolve_presets(env_cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = os.path.join(os.path.dirname(checkpoint), "exported")
    runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy to: {export_dir}")

    obs = env.get_observations()
    sim_time = 0.0
    video = VideoRecorder(f"play_stage_{args.stage}", fps=1 / C.POLICY_DT, enabled=args.video)  # 1 frame per step
    print("[INFO] Playing. Close the window (or press Ctrl+C) to stop.")
    while simulation_app.is_running():
        if args.duration is not None and sim_time >= args.duration:
            break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        video.capture()
        sim_time += C.POLICY_DT
    video.close()
    env.close()


if __name__ == "__main__":
    main()
    close_app(simulation_app)
