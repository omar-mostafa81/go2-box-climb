"""Kinematic replay of a planned jump with the real Isaac Sim Go2. Part: TO.

No physics is simulated here: no time stepping, no motors, no controller. The plan
already obeys the robot's dynamics and limits (optimizer constraints, checked by
physics_check.py). This module only DRAWS it: at every moment it puts the robot in the
planned pose (base position, pitch and the 12 joint angles) and lets MuJoCo compute
where every link is (forward kinematics), then renders the robot with its real visual
meshes from the Isaac Sim Go2 USD.

It also checks the plan in 3D with the real model:
    * stance feet: the foot centers of the 3D robot sit on their surface,
    * clearance: the true 3D distance between the robot's collision shapes (from the USD)
      and the ground, box and platform, at every frame (feet that are planned to touch
      a surface are left out of this check).
"""

from dataclasses import dataclass

import numpy as np

from go2_box_climb.common import constants as C

from .mujoco_model import export_visual_meshes, go2_scene_xml
from .optimizer import PHASES, JumpSolution
from .reference import joints_12

FEET = ("FL", "FR", "RL", "RR")
SIDE_LEG = (0, 0, 1, 1)  # side-view leg (0 front pair, 1 rear pair) of FL, FR, RL, RR
TRAIL_RGBA = {0: (0.16, 0.47, 0.84, 1.0), 1: (0.92, 0.41, 0.20, 1.0)}  # front feet blue, rear feet orange


def load_replay_scene():
    """The jump-1 scene with the real Go2 meshes. Returns (mujoco.MjModel, mujoco.MjData)."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(go2_scene_xml(visual_meshes=export_visual_meshes()))
    return model, mujoco.MjData(model)


class _Pose:
    """Puts the MuJoCo robot in the planned pose at any time t (no physics)."""

    def __init__(self, model, sol: JumpSolution):
        import mujoco

        self.m, self.sol = model, sol
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINT_NAMES]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.root = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
        self.box = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_root")]
        self.feet = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f}_foot_center") for f in FEET])

    def set(self, data, t: float):
        import mujoco

        sol = self.sol
        t = float(np.clip(t, 0.0, sol.t[-1]))
        x, z, pitch = (np.interp(t, sol.t, sol.base[:, j]) for j in range(3))
        side = np.array([[np.interp(t, sol.t, sol.joints[:, j]) for j in range(4)]])
        data.qpos[self.root : self.root + 7] = [x, 0.0, z, np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]  # (w, x, y, z)
        data.qpos[self.qadr] = joints_12(side)[0]
        data.qpos[self.box : self.box + 7] = [*C.BOX_TARGET_POS, 1.0, 0.0, 0.0, 0.0]  # the box stays where it is
        mujoco.mj_kinematics(self.m, data)  # where every link is: forward kinematics only, no dynamics
        mujoco.mj_camlight(self.m, data)  # camera and light positions (for rendering)


@dataclass
class ReplayCheck:
    stance_foot_error: float  # largest |foot height - surface height| of feet planned on a surface [m]
    min_clearance: float  # smallest 3D distance robot shape <-> ground/box/platform (feet in contact excluded) [m]
    min_clearance_where: str

    def lines(self) -> list[str]:
        return [
            f"stance feet on their surface: within {self.stance_foot_error * 1000:.1f} mm (3D Go2)",
            f"closest robot shape to the scene: {self.min_clearance * 100:+.1f} cm ({self.min_clearance_where})",
        ]


def check_replay(model, data, sol: JumpSolution, dt: float = 0.005) -> ReplayCheck:
    """Check the plan in 3D with the real robot model (see the module docstring)."""
    import mujoco

    pose = _Pose(model, sol)
    names = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    scene = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("ground", "box", "platform")]
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] >= base_body and model.geom_group[g] != 1]
    foot_geoms = {f: [g for g in robot if names(g).startswith(f"{f}_foot")] for f in FEET}
    r = float(model.geom_size[foot_geoms["FL"][0], 0])
    fromto = np.zeros(6)
    worst_foot, best, where = 0.0, np.inf, ""
    for t in np.arange(0.0, sol.t[-1] + 1e-9, dt):
        pose.set(data, t)
        k = int(np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1))
        k1 = min(k + 1, len(sol.t) - 1)
        surface = sol.task.start_height if sol.phase[k] < 2 else sol.task.target_height
        skip = set()
        for i, f in enumerate(FEET):
            leg = SIDE_LEG[i]
            if sol.contact[k, leg] and sol.contact[k1, leg]:  # standing for this whole step
                worst_foot = max(worst_foot, abs(data.site_xpos[pose.feet[i], 2] - (surface + r)))
            if sol.contact[k, leg] or sol.contact[k1, leg]:  # touching, lifting off or touching down
                skip.update(foot_geoms[f])
        for g in robot:
            if g in skip:
                continue
            for s in scene:
                d = mujoco.mj_geomDistance(model, data, g, s, 0.2, fromto)
                if d < best:
                    best, where = d, f"{names(g)} to {names(s)} at t={t:.2f} s"
    return ReplayCheck(stance_foot_error=float(worst_foot), min_clearance=float(best), min_clearance_where=where)


def replay_video(model, data, sol: JumpSolution, path, camera: str = "side", slowdown: float = 3.0, fps: int = 30,
                 width: int = 1600, height: int = 800, hold_s: float = 1.0):  # fmt: skip
    """MP4 (H.264, plays on phones) of the planned jump drawn with the real Go2, with foot trails."""
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    pose = _Pose(model, sol)
    renderer = mujoco.Renderer(model, height, width)
    frame_dt = 1.0 / (fps * slowdown)
    times = list(np.arange(0.0, sol.t[-1] + 1e-9, frame_dt))
    n_hold = int(round(hold_s * fps))
    times = [0.0] * n_hold + times + [sol.t[-1]] * n_hold
    try:
        font, small = ImageFont.truetype("arial.ttf", 30), ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-movflags", "+faststart"], macro_block_size=8)  # fmt: skip
    writer.send(None)
    trail = []  # (position, side-view leg) of every foot, every frame so far
    for t in times:
        pose.set(data, t)
        if not trail or t > trail[-1][2]:
            trail += [(data.site_xpos[pose.feet[i]].copy(), SIDE_LEG[i], t) for i in range(4)]
        renderer.update_scene(data, camera=camera)
        scn = renderer.scene
        for pos, leg, _ in trail[::2]:  # a dotted trail behind each foot
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]), pos,
                                np.eye(3).ravel(), np.array(TRAIL_RGBA[leg], dtype=np.float32))  # fmt: skip
            scn.ngeom += 1
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        k = int(np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1))
        draw.text((24, 18), "Go2 ground-to-box jump - planned trajectory (Isaac Sim robot model)", fill=(20, 20, 20), font=font)
        draw.text((24, 58), f"t = {t:.2f} s   phase: {PHASES[sol.phase[k]][0]}   ({slowdown:g}x slower)   "
                  "trails: blue = front feet, orange = rear feet", fill=(60, 60, 60), font=small)  # fmt: skip
        writer.send(np.asarray(img).tobytes())
    writer.close()
    renderer.close()
    return path


def watch(model, data, sol: JumpSolution, slowdown: float = 3.0):
    """Play the planned jump in MuJoCo's interactive 3D viewer, in a loop (close the window to stop)."""
    import time

    import mujoco.viewer

    pose = _Pose(model, sol)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            while viewer.is_running():
                t = (time.time() - start) / slowdown
                if t > sol.t[-1] + 1.0:
                    break
                pose.set(data, t)
                viewer.sync()
                time.sleep(1 / 60)
