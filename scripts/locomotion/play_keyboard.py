"""Drive the Go2 with the keyboard using a trained walking policy.

The robot is controlled exactly the way the final task will control it: the
Isaac-free WalkingPolicy (go2_box_climb/skills/locomotion) reads the robot state
and the velocity command and returns the joint targets. The scene has the box and
the platform, so you can try pushing the box against the platform.

Keys work in TWO places (use either):

  In the terminal you started the script from (works in every mode, also over SSH):
    each key press CHANGES the command, the robot keeps going until you change it
    Up / Down       forward speed +0.1 / -0.1 m/s
    Left / Right    turn rate +0.2 / -0.2 rad/s (left / right)
    Z / X           sideways speed +0.1 / -0.1 m/s (left / right)
    W / S           body height +1 / -1 cm (stage B policies; normal 32 cm, pushing ~27 cm)
    = / -           speed level up / down (scales the current speeds; see the window keys below)
    Space           stop (all speeds to zero)
    R               reset the robot and the box
    C               camera: follow the robot on / off

  In the Isaac Sim window (gui) or the streaming client (stream); click the view first:
    hold Up / Down, Left / Right, Z / X to move; release to stop (overrides the terminal)
    = / -           speed level up / down: scales forward, sideways AND turning speed together
    W / S, Space, R, C   as above

Examples (from the repository root):
    python scripts/locomotion/play_keyboard.py --policy logs/rsl_rl/go2_walk_push/<run>/exported/policy.pt --scene warehouse
    python scripts/locomotion/play_keyboard.py --policy <...>/policy.pt --mode stream     # basic scene, streamed
    python scripts/locomotion/play_keyboard.py                                            # released policy (policies/walking)

    # No keyboard: replay fixed commands "vx,vy,wz,seconds[,height];..." (for tests)
    python scripts/locomotion/play_keyboard.py --policy <...>/policy.pt --mode headless --auto "0.3,0,0,8,0.27;0,0,0,2"

--policy accepts an exported TorchScript policy.pt, or a training checkpoint
model_<N>.pt (it is exported automatically to <run>/exported/model_<N>_policy.pt).
"""

import argparse

from go2_box_climb.envs.scene import SCENES
from go2_box_climb.skills.locomotion.walking_policy import DEFAULT_POLICY_PATH
from go2_box_climb.utils.launch import add_launch_args, close_app, launch

parser = argparse.ArgumentParser(description="Drive the Go2 with the keyboard (walking policy).")
parser.add_argument(
    "--policy", type=str, default=str(DEFAULT_POLICY_PATH), help="Exported policy.pt or a training checkpoint model_<N>.pt."
)
parser.add_argument("--scene", choices=SCENES, default="basic", help="World to load.")
parser.add_argument(
    "--speed", type=float, default=0.5, help="Starting speed level, 0.1-1.0 (fraction of the max speed on every axis)."
)
parser.add_argument("--auto", type=str, default=None, help='Fixed commands instead of keys: "vx,vy,wz,seconds[,h];..."')
add_launch_args(parser, default_mode="gui")
args = parser.parse_args()
simulation_app = launch(args)

# --- Everything below needs the simulator to be running ---
import atexit  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import sys  # noqa: E402
import termios  # noqa: E402
import time  # noqa: E402
import tty  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402
from go2_box_climb.common.state import TaskState  # noqa: E402
from go2_box_climb.envs.scene import get_scene_cfg_class  # noqa: E402
from go2_box_climb.skills.locomotion import COMMAND_LIMITS, DEFAULT_BODY_HEIGHT, WalkSkill, has_fallen  # noqa: E402

# Window keys: holding a key commands  level x max  on that axis (max = the trained limit).
# '=' / '-' change the level in 0.1 steps, so forward, sideways and turning speed change together.
MAX_SPEED = {"vx": COMMAND_LIMITS["vx"][1], "vy": COMMAND_LIMITS["vy"][1], "wz": COMMAND_LIMITS["wz"][1]}
STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.2, "h": 0.01}  # change per key press (terminal; W/S also in the window)


_saved_terminal: list = []  # (fd, original settings) while the terminal is in key-reading mode


def restore_terminal():
    """Give the terminal back its normal mode (typed characters visible again). Safe to call twice."""
    while _saved_terminal:
        fd, settings = _saved_terminal.pop()
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)


class KeyboardCommand:
    """Velocity command from the keyboard: Isaac Sim window keys (held) and terminal keys (stepped)."""

    def __init__(self, speed: float):
        self.speed = speed
        self.held: set[str] = set()  # window keys currently held down
        self.set_cmd = np.zeros(3)  # command set from the terminal (vx, vy, wz)
        self.height = DEFAULT_BODY_HEIGHT  # body-height command (W / S)
        self.reset_requested = False
        self.follow_camera = True
        self._open_window_keyboard()
        self._open_terminal()

    # ---- Isaac Sim window (gui / stream): key press AND release events ----
    def _open_window_keyboard(self):
        try:
            import carb
            import omni.appwindow  # only available when there is a window (gui / stream)

            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_window_event)
        except Exception:
            self._carb = None

    def _on_window_event(self, event, *args) -> bool:
        name = event.input.name
        kinds = self._carb.input.KeyboardEventType
        if event.type == kinds.KEY_PRESS:
            self.held.add(name)
            if name == "R":
                self.reset_requested = True
            elif name == "C":
                self.follow_camera = not self.follow_camera
            elif name == "SPACE":
                self.set_cmd[:] = 0.0
            elif name in ("W", "S"):
                self._change_height(+1 if name == "W" else -1)
            elif name in ("EQUAL", "NUMPAD_ADD"):
                self._change_level(+1)
            elif name in ("MINUS", "NUMPAD_SUBTRACT"):
                self._change_level(-1)
        elif event.type == kinds.KEY_RELEASE:
            self.held.discard(name)
        return True

    # ---- Terminal: only key presses exist (no release), so each press steps the command ----
    def _open_terminal(self):
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self._fd is not None:
            _saved_terminal.append((self._fd, termios.tcgetattr(self._fd)))
            # Restored explicitly before the simulator closes (see the end of this file): Isaac Sim's
            # close() ends the process without running normal exit handlers, so atexit alone is not enough.
            atexit.register(restore_terminal)
            tty.setcbreak(self._fd)  # keys arrive immediately and are not echoed; Ctrl+C still works

    def _poll_terminal(self):
        if self._fd is None:
            return
        data = b""
        while select.select([self._fd], [], [], 0)[0]:
            data += os.read(self._fd, 64)
        # arrow keys arrive as escape sequences; map them to single private characters
        arrows = {"\x1b[A": "\x01", "\x1b[B": "\x02", "\x1b[C": "\x03", "\x1b[D": "\x04"}
        keys = data.decode(errors="ignore")
        for seq, ch in arrows.items():
            keys = keys.replace(seq, ch)
        for k in keys:
            c = self.set_cmd
            if k == "\x01":  # Up
                c[0] += STEP["vx"]
            elif k == "\x02":  # Down
                c[0] -= STEP["vx"]
            elif k == "\x04":  # Left
                c[2] += STEP["wz"]
            elif k == "\x03":  # Right
                c[2] -= STEP["wz"]
            elif k in ("z", "Z"):
                c[1] += STEP["vy"]
            elif k in ("x", "X"):
                c[1] -= STEP["vy"]
            elif k == " ":
                c[:] = 0.0
            elif k in ("r", "R"):
                self.reset_requested = True
            elif k in ("c", "C"):
                self.follow_camera = not self.follow_camera
            elif k in ("=", "+", "-"):
                self._change_level(+1 if k in ("=", "+") else -1)
            elif k in ("w", "W", "s", "S"):
                self._change_height(+1 if k in ("w", "W") else -1)
            for j, name in enumerate(("vx", "vy", "wz")):  # stay inside the trained range
                c[j] = float(np.clip(round(c[j], 2), *COMMAND_LIMITS[name]))

    def _change_level(self, sign: int):
        """Speed level up/down by 0.1; a command set from the terminal is scaled along with it."""
        old = self.speed
        self.speed = float(np.clip(round(self.speed + 0.1 * sign, 1), 0.1, 1.0))
        self.set_cmd *= self.speed / old
        for j, name in enumerate(("vx", "vy", "wz")):
            self.set_cmd[j] = float(np.clip(self.set_cmd[j], *COMMAND_LIMITS[name]))

    def _change_height(self, sign: int):
        self.height = float(np.clip(round(self.height + sign * STEP["h"], 2), *COMMAND_LIMITS["h"]))

    def command(self) -> tuple[float, float, float, float]:
        self._poll_terminal()
        h = self.held
        if h & {"UP", "DOWN", "LEFT", "RIGHT", "Z", "X"}:  # window keys held: they win
            vx = self.speed * MAX_SPEED["vx"] * (("UP" in h) - ("DOWN" in h))
            vy = self.speed * MAX_SPEED["vy"] * (("Z" in h) - ("X" in h))
            wz = self.speed * MAX_SPEED["wz"] * (("LEFT" in h) - ("RIGHT" in h))
            return vx, vy, wz, self.height
        return (*self.set_cmd, self.height)

    def clear(self):
        self.set_cmd[:] = 0.0


class AutoCommand:
    """Replays a fixed list of commands (no keyboard), for tests."""

    def __init__(self, spec: str):
        self.steps = []
        for part in spec.split(";"):
            if part.strip():
                v = [float(x) for x in part.split(",")]
                self.steps.append((v[0], v[1], v[2], v[3], v[4] if len(v) > 4 else DEFAULT_BODY_HEIGHT))
        self.t, self.speed, self.reset_requested, self.follow_camera = 0.0, 0.0, False, True
        self.height = DEFAULT_BODY_HEIGHT
        self.total = sum(s[3] for s in self.steps)

    def command(self):
        self.t += C.POLICY_DT
        acc = 0.0
        for vx, vy, wz, dur, h in self.steps:
            acc += dur
            if self.t <= acc:
                self.height = h
                return vx, vy, wz, h
        return 0.0, 0.0, 0.0, self.height

    def done(self) -> bool:
        return self.t >= self.total


def _np(x):
    return (x.torch if hasattr(x, "torch") else x)[0].detach().cpu().numpy().astype(float)


def main():
    if args.scene == "warehouse":
        print("[INFO] Loading the warehouse (first start downloads its 3D models; be patient)")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=C.SIM_DT, device=args.device))
    scene = InteractiveScene(get_scene_cfg_class(args.scene)(num_envs=1))
    sim.reset()
    robot, box = scene["robot"], scene["box"]
    assert list(robot.joint_names) == list(C.JOINT_NAMES), f"Joint order mismatch: {robot.joint_names}"
    origin = scene.env_origins[0].cpu().numpy()

    walk = WalkSkill(args.policy)
    print(f"[INFO] Policy: {args.policy}")
    if not walk.policy.has_height_input:
        print("[INFO] This is a stage A policy (no body-height input): W / S have no effect.")
    if walk.policy.policy_file != Path(args.policy):
        print(f"[INFO] Training checkpoint exported to: {walk.policy.policy_file}")
    keys = AutoCommand(args.auto) if args.auto else KeyboardCommand(args.speed)
    if not args.auto:
        print(__doc__[__doc__.index("Keys work"): __doc__.index("Examples")])

    default_q = torch.tensor(C.DEFAULT_JOINT_POS, device=sim.device, dtype=torch.float32)

    def read_state() -> TaskState:
        return TaskState(
            base_pos=_np(robot.data.root_pos_w) - origin,
            base_quat=_np(robot.data.root_quat_w),
            base_lin_vel=_np(robot.data.root_lin_vel_w),
            base_ang_vel=_np(robot.data.root_ang_vel_w),
            joint_pos=_np(robot.data.joint_pos),
            joint_vel=_np(robot.data.joint_vel),
            box_pos=_np(box.data.root_pos_w) - origin,
            box_quat=_np(box.data.root_quat_w),
        )

    def reset_world():
        for asset in (robot, box):
            pose = (asset.data.default_root_pose.torch if hasattr(asset.data.default_root_pose, "torch")
                    else asset.data.default_root_pose).clone()
            pose[:, :3] += scene.env_origins
            asset.write_root_pose_to_sim_index(root_pose=pose)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=sim.device))
        robot.write_joint_state_to_sim_index(
            position=default_q.unsqueeze(0), velocity=torch.zeros(1, C.NUM_JOINTS, device=sim.device)
        )
        scene.reset()
        walk.reset(read_state())

    reset_world()
    print("[INFO] Running. Close the window (or press Ctrl+C) to stop.")
    step, fallen_reported = 0, False
    while simulation_app.is_running():
        t0 = time.time()
        if keys.reset_requested:
            keys.reset_requested, fallen_reported = False, False
            if hasattr(keys, "clear"):
                keys.clear()  # a reset also stops the robot
            reset_world()
            print("\n[INFO] Reset.")

        # --- one policy step (50 Hz): state -> walking policy -> joint targets ---
        state = read_state()
        cmd = walk.set_command(*keys.command())
        action = walk.act(state)
        target = default_q + C.ACTION_SCALE * torch.tensor(action, device=sim.device, dtype=torch.float32)
        robot.set_joint_position_target_index(target=target.unsqueeze(0))
        for i in range(C.DECIMATION):  # 4 physics steps; draw only the last one
            scene.write_data_to_sim()
            sim.step(render=(i == C.DECIMATION - 1))
            scene.update(C.SIM_DT)
        step += 1

        # --- camera: behind the robot, looking where it walks ---
        if keys.follow_camera and args.mode != "headless":
            yaw = 2 * np.arctan2(state.base_quat[2], state.base_quat[3])
            back = np.array([-1.8 * np.cos(yaw) + 0.9 * np.sin(yaw), -1.8 * np.sin(yaw) - 0.9 * np.cos(yaw)])
            p = state.base_pos + origin
            sim.set_camera_view(eye=(p[0] + back[0], p[1] + back[1], 1.1), target=(p[0], p[1], 0.25))

        # --- status line every 0.5 s ---
        if step % 25 == 0:
            yaw_rate = state.base_ang_vel[2]
            v_body = state.base_lin_vel  # world frame; fine for a status line
            box_front = state.box_pos[0] + C.BOX_SIZE[0] / 2
            box_yaw = np.degrees(2 * np.arctan2(state.box_quat[2], state.box_quat[3]))
            print(
                f"\rcmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f} h {cmd[3]:.2f} (level {keys.speed:.1f}) | "
                f"robot v {np.linalg.norm(v_body[:2]):.2f} m/s wz {yaw_rate:+.2f} h {state.base_pos[2]:.2f} | "
                f"box gap to platform {100 * (C.PLATFORM_FRONT_X - box_front):5.1f} cm, "
                f"sideways {100 * state.box_pos[1]:+5.1f} cm, yaw {box_yaw:+5.1f} deg   ",
                end="",
                flush=True,
            )
        if has_fallen(state) and not fallen_reported:
            fallen_reported = True
            print("\n[INFO] The robot fell. Press R to reset.")

        if args.auto and keys.done():
            break
        if args.mode != "headless":  # keep real time so the keys feel right
            time.sleep(max(0.0, C.POLICY_DT - (time.time() - t0)))

    s = read_state()
    print(f"\n[INFO] End: robot at {np.round(s.base_pos, 2).tolist()}, box at {np.round(s.box_pos, 3).tolist()}, "
          f"box gap to platform {100 * (C.PLATFORM_FRONT_X - s.box_pos[0] - C.BOX_SIZE[0] / 2):.1f} cm, "
          f"fallen: {has_fallen(s)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped (Ctrl+C).")
    finally:
        restore_terminal()  # before close(): close() exits without running exit handlers
    close_app(simulation_app)
