"""Simulate the planned jump in MuJoCo (a second physics engine; no Isaac Sim, no GPU needed).

The MuJoCo robot is built from the same Go2 USD as Isaac Sim, and it is driven with the
project's motor contract (50 Hz targets, 200 Hz PD 25 / 0.5, DC-motor limit). See
go2_box_climb/skills/jump/to/mujoco_model.py and mujoco_sim.py.

Examples (from the repository root):
    python scripts/jump/simulate_mujoco.py                     # all three controllers + figure + video
    python scripts/jump/simulate_mujoco.py --viewer            # also watch it in MuJoCo's 3D window
    python scripts/jump/simulate_mujoco.py --replan            # solve the plan again first (~30 s)
    python scripts/jump/simulate_mujoco.py --camera three_quarter

Output (ignored by git): outputs/jump_to/ground_to_box/
    mujoco_comparison.png, mujoco_<controller>.mp4
"""

import argparse
import pickle
import shutil
import time
from pathlib import Path

from go2_box_climb.skills.jump.to import plots
from go2_box_climb.skills.jump.to.mujoco_model import load_scene
from go2_box_climb.skills.jump.to.mujoco_sim import render_video, simulate_jump, stand_test
from go2_box_climb.skills.jump.to.optimizer import ground_to_box_task, solve_jump
from go2_box_climb.skills.jump.to.robot_params import jump_robot_params

ROOT = Path(__file__).resolve().parents[2]
MODES = ("feedback", "feedforward", "replay")


def watch(model, result, slowdown: float = 3.0):
    """Play a finished simulation in MuJoCo's interactive viewer (close the window to stop)."""
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            for k in range(0, len(result.t), 2):
                if not viewer.is_running():
                    break
                data.qpos[:] = result.qpos[k]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, start + result.t[k] * slowdown - time.time()))
            time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replan", action="store_true", help="solve the trajectory optimization again first")
    parser.add_argument("--camera", default="side", choices=["side", "three_quarter"], help="video camera")
    parser.add_argument("--no-video", action="store_true", help="skip the MP4 videos")
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo's 3D viewer on the feedback run")
    parser.add_argument("--figures", action="store_true", help="copy the comparison figure to docs/figures/")
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
        print(f"Using the saved plan {plan_file} (run with --replan to solve it again)")

    model, data = load_scene(real_meshes=True)  # the real Isaac Sim meshes are drawn only: same physics
    print(f"Stand test: holding the default pose, the base settles at {stand_test(model, data):.3f} m "
          "(Isaac Sim: about 0.28 m)")  # fmt: skip

    results = {}
    for mode in MODES:
        res = simulate_jump(model, data, sol, mode)
        results[mode] = res
        print(f"  {mode:11s} {'SUCCESS' if res.success else 'FAIL   '}  {res.reason}")
    fig = plots.plot_mujoco_comparison(sol, results, out / "mujoco_comparison.png")
    print(f"Figure: {fig}")
    if args.figures:
        shutil.copy(fig, ROOT / "docs" / "figures" / "jump_ground_to_box_mujoco_comparison.png")
        print("Copied the figure to docs/figures/")
    if not args.no_video:
        for mode, res in results.items():
            video = render_video(model, res, sol, out / f"mujoco_{mode}.mp4", camera=args.camera)
            print(f"Video: {video}")
    if args.viewer:
        watch(model, results["feedback"])


if __name__ == "__main__":
    main()
