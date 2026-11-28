"""Evaluate the jump-tracking policy (or the bare reference): success rate over many attempts.

Each round resets all robots (new random start conditions from common/constants.py) and
runs until every robot has finished ONE attempt; only that first attempt counts. Each
attempt is logged as one row of the shared trial log (common/formats.py
TRIAL_LOG_COLUMNS: skill, variant, start conditions, success, duration, failure reason),
so the box-placement study and the sequencer can read it.

    --checkpoint <run>/model_<N>.pt   evaluate the trained policy (variant "to_rl")
    --open_loop                       no policy, the motors follow the reference
                                      (variant "reference_open_loop")

Output (never overwritten, a new time stamp per run):
    checkpoint:  <run>/eval/<model_N>_<jump>_<date_time>_trials.csv and _summary.json
    open loop:   outputs/jump_from_tracking/eval/open_loop_<jump>_<date_time>_trials.csv and _summary.json

Examples (from the repository root):
    python scripts/jump_from_tracking/eval.py --jump ground_to_box \
        --checkpoint logs/rsl_rl/go2_jump_track_ground_to_box/<run>/model_1499.pt
    python scripts/jump_from_tracking/eval.py --jump ground_to_box --open_loop --rounds 1 \
        --reference outputs/jump_from_tracking/placeholder_jump_ground_to_box.npz
"""

import argparse

from go2_box_climb.skills.jump.jump_from_tracking.reference import JUMPS, reference_file
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Evaluate the jump-tracking policy (success rate).")
parser.add_argument("--jump", choices=JUMPS, required=True)
parser.add_argument("--reference", type=str, default=None, help="Reference .npz (default: references/).")
who = parser.add_mutually_exclusive_group(required=True)
who.add_argument("--checkpoint", type=str, help="Checkpoint (.pt) to evaluate.")
who.add_argument("--open_loop", action="store_true", help="No policy: follow the reference's joint angles.")
parser.add_argument("--num_envs", type=int, default=256, help="Robots per round (one attempt each).")
parser.add_argument("--rounds", type=int, default=4, help="Rounds (attempts = num_envs * rounds).")
add_launch_args(parser, default_mode="headless")
args = parser.parse_args()
reference = reference_file(args.jump, args.reference)  # fail fast, before Isaac Sim starts

simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import importlib.metadata  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from go2_box_climb.common.formats import append_trial_log  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.agents import JumpTrackPPORunnerCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.jump_env_cfg import COMMAND, JumpTrackEnvCfg  # noqa: E402
from go2_box_climb.envs.jump_from_tracking.mdp import open_loop_actions  # noqa: E402
from go2_box_climb.skills.jump.jump_from_tracking.jump_policy import SKILL_NAMES  # noqa: E402
from go2_box_climb.skills.jump.jump_from_tracking.reference import PLACEHOLDER_DIR  # noqa: E402

OUTCOMES = ("success", "fall_contact", "fall_tilt", "time_out")  # priority if several happen at once


def main():
    # shared jump rules only: no training aids (early stop, part-way starts), full start/box ranges
    env_cfg = resolve_presets(JumpTrackEnvCfg().configure(args.jump, str(reference)).eval_mode())
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    u = env.unwrapped
    cmd = u.command_manager.get_term(COMMAND)
    tm = u.termination_manager

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.checkpoint:
        agent_cfg = JumpTrackPPORunnerCfg().for_jump(args.jump)
        agent_cfg.device = args.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=u.device)
        variant = "to_rl"
        out_dir = Path(args.checkpoint).parent / "eval"
        prefix = f"{Path(args.checkpoint).stem}_{args.jump}_{stamp}"
    else:

        def policy(_obs):
            return open_loop_actions(u, COMMAND)

        variant = "reference_open_loop"
        out_dir = PLACEHOLDER_DIR / "eval"
        prefix = f"open_loop_{args.jump}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_trials.csv"

    n, dev = args.num_envs, u.device
    trial_id = 0
    totals = dict.fromkeys(OUTCOMES, 0)
    success_durations = []
    for r in range(args.rounds):
        u.reset()
        obs = env.get_observations()
        # start conditions of this round's attempts (the command term draws them at reset)
        start = {
            "box_gap": cmd.box_gap.clone(),
            "box_offset_y": cmd.box_offset_y.clone(),
            "box_yaw": cmd.box_yaw.clone(),
            "robot_x": cmd.start_xy[:, 0].clone(),
            "robot_y": cmd.start_xy[:, 1].clone(),
            "robot_yaw": cmd.start_yaw.clone(),
        }
        finished = torch.zeros(n, dtype=torch.bool, device=dev)
        outcome = torch.full((n,), -1, dtype=torch.long, device=dev)
        duration = torch.zeros(n, device=dev)
        for k in range(u.max_episode_length + 1):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            new = dones.bool() & ~finished
            if new.any():
                undecided = new.clone()
                for i, name in enumerate(OUTCOMES):
                    hit = undecided & tm.get_term(name)
                    outcome[hit] = i
                    undecided &= ~hit
                duration[new] = (k + 1) * u.step_dt
                finished |= new
            if finished.all():
                break
        for e in range(n):
            o = int(outcome[e])
            name = OUTCOMES[o] if o >= 0 else "not_finished"
            ok = name == "success"
            totals[name] = totals.get(name, 0) + 1
            if ok:
                success_durations.append(float(duration[e]))
            append_trial_log(
                str(csv_path),
                {
                    "skill": SKILL_NAMES[args.jump],
                    "variant": variant,
                    "trial_id": trial_id,
                    "success": int(ok),
                    "duration_s": round(float(duration[e]), 3),
                    **{key: round(float(v[e]), 4) for key, v in start.items()},
                    "failure_reason": "" if ok else name,
                },
            )
            trial_id += 1
        print(f"[INFO] Round {r + 1}/{args.rounds}: success {(outcome == 0).float().mean().item():.1%}")

    attempts = trial_id
    summary = {
        "jump": args.jump,
        "variant": variant,
        "checkpoint": args.checkpoint,
        "reference": str(reference),
        "attempts": attempts,
        "success_rate": round(totals["success"] / attempts, 4),
        "outcomes": {k: round(v / attempts, 4) for k, v in totals.items()},
        "mean_success_duration_s": round(sum(success_durations) / len(success_durations), 3) if success_durations else None,
        "trials_csv": str(csv_path),
    }
    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] {attempts} attempts: " + ", ".join(f"{k} {v:.1%}" for k, v in summary["outcomes"].items()))
    print(f"[INFO] Saved: {csv_path}\n[INFO] Saved: {out_dir / f'{prefix}_summary.json'}")
    env.close()


if __name__ == "__main__":
    main()
    close_app(simulation_app)
