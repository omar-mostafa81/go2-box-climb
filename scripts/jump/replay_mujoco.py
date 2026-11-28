"""Draw the planned jump with the real Isaac Sim Go2 (kinematic replay: no physics, no controller).

Every frame puts the robot in the planned pose (base position, pitch and 12 joint angles)
and MuJoCo only computes where the links are and renders the Go2's real visual meshes
from the Isaac Sim USD. The plan itself already obeys the robot's dynamics and limits.

Examples (from the repository root):
    python scripts/jump/replay_mujoco.py                  # 3D check + side and 3/4 videos
    python scripts/jump/replay_mujoco.py --viewer         # also watch it in MuJoCo's 3D window (loops)
    python scripts/jump/replay_mujoco.py --replan         # solve the plan again first (~30 s)

Output (ignored by git): outputs/jump_to/ground_to_box/replay_side.mp4, replay_three_quarter.mp4
"""

import argparse
import pickle
from pathlib import Path

from go2_box_climb.skills.jump.to.mujoco_replay import check_replay, load_replay_scene, replay_video, watch
from go2_box_climb.skills.jump.to.optimizer import ground_to_box_task, solve_jump
from go2_box_climb.skills.jump.to.robot_params import jump_robot_params

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replan", action="store_true", help="solve the trajectory optimization again first")
    parser.add_argument("--no-video", action="store_true", help="skip the MP4 videos")
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo's 3D viewer and loop the jump")
    parser.add_argument("--slowdown", type=float, default=3.0, help="how many times slower than real time")
    args = parser.parse_args()

    out = ROOT / "outputs" / "jump_to" / "ground_to_box"
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / "solution.pkl"
    if args.replan or not plan_file.exists():
        print("Solving the jump plan (about 30 s)...")
        sol = solve_jump(ground_to_box_task(), jump_robot_params())
        with open(plan_file, "wb") as f:
            pickle.dump(sol, f)
    else:
        with open(plan_file, "rb") as f:
            sol = pickle.load(f)
        print(f"Using the saved plan {plan_file} (run plan_jump.py or --replan to make a new one)")

    model, data = load_replay_scene()
    print("3D check with the Isaac Sim Go2 (forward kinematics only):")
    for line in check_replay(model, data, sol).lines():
        print("  " + line)
    if not args.no_video:
        for camera in ("side", "three_quarter"):
            video = replay_video(model, data, sol, out / f"replay_{camera}.mp4", camera=camera, slowdown=args.slowdown)
            print(f"Video: {video}")
    if args.viewer:
        watch(model, data, sol, slowdown=args.slowdown)


if __name__ == "__main__":
    main()
