"""Open a box-climbing scene and let it run.

The robot holds its default standing pose (PD control toward DEFAULT_JOINT_POS);
nothing else moves. Use this to check the scene layout.

Scenes (--scene): basic (plain floor, default) or warehouse (the final test world).

Examples (run from the repository root, with the `isaaclab` conda env active):
    python scripts/view_scene.py                        # Isaac Sim window
    python scripts/view_scene.py --scene warehouse      # the warehouse world
    python scripts/view_scene.py --mode stream          # watch from another computer
    python scripts/view_scene.py --mode headless --duration 5
    python scripts/view_scene.py --num_envs 4
"""

import argparse

from go2_box_climb.envs.scene import SCENES
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="View a Go2 box-climbing scene.")
parser.add_argument("--scene", choices=SCENES, default="basic", help="Which world to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of copies of the scene.")
parser.add_argument(
    "--duration", type=float, default=None, help="Stop after this many simulated seconds (default: run until closed)."
)
add_launch_args(parser)
args = parser.parse_args()
if args.scene == "warehouse" and args.num_envs != 1:
    parser.error("the warehouse scene is built around a single environment; use --num_envs 1")
simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402
from go2_box_climb.envs.scene import get_scene_cfg_class  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=C.SIM_DT, device=args.device))
    # Camera behind-left of the robot, looking at the box
    sim.set_camera_view(eye=(-1.5, -3.0, 2.0), target=(1.5, 0.0, 0.2))

    scene_cfg_class = get_scene_cfg_class(args.scene)
    print(f"[INFO] Loading scene '{args.scene}' (the warehouse downloads its 3D models on first use; be patient)")
    scene = InteractiveScene(scene_cfg_class(num_envs=args.num_envs))
    sim.reset()

    robot, box = scene["robot"], scene["box"]
    # Safety check: the simulator's joint order must match the shared contract
    assert list(robot.joint_names) == list(C.JOINT_NAMES), f"Joint order mismatch: {robot.joint_names}"
    print(f"[INFO] Robot joints (in order): {robot.joint_names}")
    print(f"[INFO] Robot base start (env 0): {robot.data.root_pos_w.torch[0].tolist()}")
    print(f"[INFO] Box start (env 0):        {box.data.root_pos_w.torch[0].tolist()}")
    print("[INFO] Scene running. Close the window (or press Ctrl+C) to stop.")

    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    sim_time = 0.0
    while simulation_app.is_running():
        if args.duration is not None and sim_time >= args.duration:
            break
        robot.set_joint_position_target_index(target=default_joint_pos)  # hold the standing pose
        scene.write_data_to_sim()
        sim.step()
        scene.update(C.SIM_DT)
        sim_time += C.SIM_DT

    print(f"[INFO] Stopped after {sim_time:.2f} s of simulation.")
    print(f"[INFO] Robot base end (env 0): {robot.data.root_pos_w.torch[0].tolist()}")
    print(f"[INFO] Box end (env 0):        {box.data.root_pos_w.torch[0].tolist()}")


if __name__ == "__main__":
    main()
    close_app(simulation_app)
