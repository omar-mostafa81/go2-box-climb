"""Plan a jump with trajectory optimization, save the reference, and draw the plots.

Isaac Sim is NOT started: this runs on any computer (CasADi + numpy + matplotlib).

Examples (from the repository root):
    python scripts/jump/plan_jump.py                          # ground -> box, draft to outputs/
    python scripts/jump/plan_jump.py --start-distance 0.40    # start 0.40 m behind the box
    python scripts/jump/plan_jump.py --figures                # also copy the figures to docs/figures/
    python scripts/jump/plan_jump.py --release                # also write references/jump_ground_to_box.npz
    python scripts/jump/plan_jump.py --video                  # also write an MP4 video (phones / WhatsApp)

Output (ignored by git): outputs/jump_to/<jump>/
    solution.pkl (the full plan), reference.npz (JumpReference), foot_paths.png,
    foot_timeline.png, feet_3d.png, jump_dynamics.png, jump_animation.gif
"""

import argparse
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from go2_box_climb.common.formats import REFERENCE_FILES
from go2_box_climb.skills.jump.to import plots
from go2_box_climb.skills.jump.to.optimizer import ground_to_box_task, solve_jump
from go2_box_climb.skills.jump.to.physics_check import check_jump
from go2_box_climb.skills.jump.to.reference import to_jump_reference
from go2_box_climb.skills.jump.to.robot_params import jump_robot_params

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jump", default="ground_to_box", choices=["ground_to_box"], help="which jump to plan")
    parser.add_argument("--start-distance", type=float, default=0.45, help="base x behind the box front face [m]")
    parser.add_argument("--no-animation", action="store_true", help="skip the GIF (it takes ~20 s)")
    parser.add_argument("--video", action="store_true",
                        help="also write jump_animation.mp4 (phone/WhatsApp video; needs imageio-ffmpeg)")  # fmt: skip
    parser.add_argument("--figures", action="store_true", help="copy the PNG figures to docs/figures/")
    parser.add_argument("--release", action="store_true", help="also write the final reference to references/")
    args = parser.parse_args()

    robot = jump_robot_params()
    task = ground_to_box_task(args.start_distance)
    print(f"Robot: {robot.mass:.2f} kg, pitch inertia {robot.pitch_inertia:.3f} kg m^2 (from {robot.usd_url})")

    t0 = time.time()
    sol = solve_jump(task, robot)
    print(f"Optimizer: {sol.status} ({sol.info['iterations']} iterations, {time.time() - t0:.1f} s)")
    if sol.status != "solved":
        raise SystemExit("The optimizer did not converge; nothing is saved.")
    names = ("push", "rear push", "flight", "landing")
    print("Phase durations: " + ", ".join(f"{n} {d:.2f} s" for n, d in zip(names, sol.durations)))
    print(f"Jump time {sol.t[-1]:.2f} s, highest center of mass {sol.com[:, 1].max():.2f} m, "
          f"peak torque {np.abs(sol.torques).max():.1f} N m")  # fmt: skip
    report = check_jump(sol, robot)
    print("Physics check (exact shapes, 5 moments per step):")
    for line in report.lines():
        print("  " + line)

    out = ROOT / "outputs" / "jump_to" / task.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "solution.pkl", "wb") as f:  # the full plan (forces, torques...), reused by simulate_mujoco.py
        pickle.dump(sol, f)
    ref = to_jump_reference(sol)
    ref.save(out / "reference.npz")
    print(f"Reference: {len(ref.t)} samples every {ref.t[1] - ref.t[0]:.2f} s -> {out / 'reference.npz'}")

    figs = [
        plots.plot_foot_paths(sol, robot, out / "foot_paths.png"),
        plots.plot_foot_timeline(sol, out / "foot_timeline.png"),
        plots.plot_feet_3d(sol, robot, out / "feet_3d.png"),
        plots.plot_jump_dynamics(sol, robot, out / "jump_dynamics.png"),
    ]
    if not args.no_animation:
        figs.append(plots.animate_jump(sol, robot, out / "jump_animation.gif"))
    if args.video:  # 1600 x 800 pixels, 30 frames per second, 3x slower than real time
        figs.append(plots.animate_jump(sol, robot, out / "jump_animation.mp4", dt=1 / 90, slowdown=3.0, dpi=200))
    for f in figs:
        print(f"Figure: {f}")

    if args.figures:
        dst = ROOT / "docs" / "figures"
        for f in figs:
            if f.suffix == ".png":
                shutil.copy(f, dst / f"jump_{task.name}_{f.name}")
        print(f"Copied the PNG figures to {dst}")
    if args.release:
        ref.save(REFERENCE_FILES[task.name])
        print(f"Released reference: {REFERENCE_FILES[task.name]}")


if __name__ == "__main__":
    main()
