"""Run a planned jump in MuJoCo with the project's motor contract. Part: TO.

The robot is driven exactly like in Isaac Lab (common/constants.py):
    * every 0.02 s (POLICY_DT) a controller picks 12 joint targets,
    * every 0.005 s (SIM_DT) the motors apply  tau = KP (target - q) - KD q_dot,
      clipped by Isaac Lab's DC-motor model: tau <= 23.5 (1 - q_dot / 30),
      tau >= 23.5 (-1 - q_dot / 30) and |tau| <= 23.5.

Three controllers turn the jump plan into targets (all of them only output joint
targets, so a policy could output the same thing):
    replay        target = planned joint angle. The simplest thing to try.
    feedforward   target = planned angle + (planned torque + KD x planned joint speed) / KP.
                  With this "lead", the PD motor produces the planned torque whenever the
                  robot is exactly on the plan, and pulls it back when it drifts.
    feedback      like feedforward, but while feet stand on a surface the planned ground
                  forces are corrected from the measured center of mass and pitch (a
                  spring-damper toward the plan), kept inside the friction cone, and turned
                  into torques for all 3 joints of each leg with that foot's full 3D
                  Jacobian at the measured pose (tau = -J^T f). This also gives the hip
                  (sideways) joint the torque it needs to stop the legs from splaying,
                  which the side-view plan cannot see. A leg only counts as standing if
                  its foot really touches the surface.

The result is judged with the shared jump rules (constants.py): failure if the trunk or
head touches anything or the body tilts more than 60 deg; success if, during the last
1.0 s of the 3.0 s attempt, all four feet stand on the box top, the body is upright
(tilt < 20 deg) and nearly still (speed < 0.2 m/s).
"""

from dataclasses import dataclass, field

import numpy as np

from go2_box_climb.common import constants as C

from .optimizer import JumpSettings, JumpSolution
from .reference import joints_12, to_jump_reference
from .robot_params import JumpRobotParams

FEET = ("FL", "FR", "RL", "RR")
SIDE_LEG = (0, 0, 1, 1)  # side-view leg (0 front pair, 1 rear pair) of FL, FR, RL, RR


@dataclass(frozen=True)
class FeedbackGains:
    """Spring-damper that pulls the center of mass and the pitch back to the plan (feedback mode)."""

    position: float = 100.0  # [1/s^2]  (natural frequency 10 rad/s)
    velocity: float = 20.0  # [1/s]    (critically damped)
    pitch: float = 100.0  # [1/s^2]
    pitch_rate: float = 20.0  # [1/s]
    contact_height: float = 0.01  # a foot within this of its surface counts as standing [m]


def motor_torque(target, q, qd):
    """Isaac Lab DCMotor: PD torque clipped by the torque-speed curve and the effort limit."""
    tau = C.KP * (target - q) - C.KD * qd
    hi = np.minimum(C.EFFORT_LIMIT, C.EFFORT_LIMIT * (1.0 - qd / C.VELOCITY_LIMIT))
    lo = np.maximum(-C.EFFORT_LIMIT, C.EFFORT_LIMIT * (-1.0 - qd / C.VELOCITY_LIMIT))
    return np.clip(tau, lo, hi)


class _Indices:
    """Where things live in MuJoCo's arrays."""

    def __init__(self, model):
        import mujoco

        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.JOINT_NAMES]
        self.qpos = np.array([model.jnt_qposadr[j] for j in jid])
        self.qvel = np.array([model.jnt_dofadr[j] for j in jid])
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_q = model.jnt_qposadr[root]
        self.root_v = model.jnt_dofadr[root]
        box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_root")
        self.box_q = model.jnt_qposadr[box]
        self.feet = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{f}_foot_center") for f in FEET])
        self.fail_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in C.JUMP_FAILURE_CONTACT_BODIES}


def set_robot_state(model, data, base_pos, base_quat_xyzw, joint_pos):
    """Put the robot in a pose at rest (base quaternion in the project's (x, y, z, w) order)."""
    import mujoco

    ix = _Indices(model)
    x, y, z, w = base_quat_xyzw
    data.qpos[ix.root_q : ix.root_q + 7] = [*base_pos, w, x, y, z]  # MuJoCo stores (w, x, y, z)
    data.qpos[ix.qpos] = joint_pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _tilt(quat_wxyz) -> float:
    """Angle between the body's up axis and the world's up axis [rad]."""
    w, x, y, z = quat_wxyz
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def stand_test(model, data, seconds: float = 3.0) -> float:
    """Drop the robot at ROBOT_START_POS holding the default pose; return its final base height."""
    ix = _Indices(model)
    set_robot_state(model, data, C.ROBOT_START_POS, (0.0, 0.0, 0.0, 1.0), C.DEFAULT_JOINT_POS)
    import mujoco

    for _ in range(int(round(seconds / C.SIM_DT))):
        data.ctrl[:] = motor_torque(C.DEFAULT_JOINT_POS, data.qpos[ix.qpos], data.qvel[ix.qvel])
        mujoco.mj_step(model, data)
    return float(data.qpos[ix.root_q + 2])


@dataclass
class SimResult:
    mode: str
    t: np.ndarray  # (N,) time of every physics step [s]
    qpos: np.ndarray  # (N, nq) full MuJoCo state (for rendering)
    base_pos: np.ndarray  # (N, 3)
    pitch: np.ndarray  # (N,) [rad], + = nose down
    tilt: np.ndarray  # (N,) [rad]
    base_speed: np.ndarray  # (N,) [m/s]
    joint_pos: np.ndarray  # (N, 12)
    joint_vel: np.ndarray  # (N, 12)
    torque: np.ndarray  # (N, 12) applied motor torque [N m]
    feet: np.ndarray  # (N, 4, 3) foot centers [m]
    box_pos: np.ndarray  # (N, 3)
    fail_contact_time: float | None  # first time the trunk/head touched something [s]
    success: bool = False
    reason: str = ""
    info: dict = field(default_factory=dict)


def _feedback_targets(model, data, ix, sol: JumpSolution, robot: JumpRobotParams, t: float, q_ref, qd_ref,
                      tau_ff, gains: FeedbackGains, settings: JumpSettings):  # fmt: skip
    """Joint targets of the feedback controller at time t (see the module docstring)."""
    import mujoco

    r = robot.foot_radius
    base_id = model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    mujoco.mj_subtreeVel(model, data)
    com = data.subtree_com[base_id][[0, 2]]
    com_vel = data.subtree_linvel[base_id][[0, 2]]
    w, x, y, z = data.qpos[ix.root_q + 3 : ix.root_q + 7]
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    pitch_rate = float(data.qvel[ix.root_v + 4])  # body y angular velocity

    # plan at time t (the step that starts at the last node at or before t)
    k = int(np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1))
    plan = lambda a: np.array([np.interp(t, sol.t, a[:, j]) for j in range(a.shape[1])])  # noqa: E731
    com_ref, vel_ref = plan(sol.com), plan(sol.com_vel)
    f_plan = sol.forces[k].copy()  # front fx, fz, rear fx, fz (pair totals)

    # which pairs stand on a surface: planned contact AND both feet really down
    surface = {0: sol.task.start_height, 1: sol.task.target_height}
    feet = data.site_xpos[ix.feet]
    standing = []
    for leg in (0, 1):
        if not sol.contact[k, leg]:
            continue
        s = surface[0] if sol.phase[k] < 2 else surface[1]
        idx = [i for i in range(4) if SIDE_LEG[i] == leg]
        if all(feet[i, 2] < s + r + gains.contact_height for i in idx):
            standing.append(leg)

    q = data.qpos[ix.qpos]
    tau_cmd = tau_ff.copy()
    if standing:
        # wrench (forward force, vertical force, pitch torque about the COM) the plan asks for, plus feedback
        m = robot.mass
        dF = m * (gains.position * (com_ref[:2] - com) + gains.velocity * (vel_ref[:2] - com_vel))
        dT = robot.pitch_inertia * (gains.pitch * (plan(sol.base)[2] - pitch) + gains.pitch_rate * (vel_ref[2] - pitch_rate))
        cols = []
        for leg in standing:
            idx = [i for i in range(4) if SIDE_LEG[i] == leg]
            px, pz = feet[idx][:, [0, 2]].mean(axis=0)
            cols += [(1.0, 0.0, pz - com[1]), (0.0, 1.0, -(px - com[0]))]  # effect of fx and fz of this pair
        A = np.array(cols).T  # (3, 2 x number of standing pairs)
        f = np.concatenate([f_plan[2 * leg : 2 * leg + 2] for leg in standing])
        f = f + np.linalg.pinv(A) @ np.array([dF[0], dF[1], dT])
        jac = np.zeros((3, model.nv))
        for n, leg in enumerate(standing):  # keep each pair inside its friction cone, pushing only
            s = "start" if sol.phase[k] < 2 else "target"
            mu = (sol.task.friction_start if s == "start" else sol.task.friction_target) * settings.foot_friction
            fz = max(f[2 * n + 1], 0.0)
            fx = float(np.clip(f[2 * n], -settings.friction_margin * mu * fz, settings.friction_margin * mu * fz))
            for i in range(4):
                if SIDE_LEG[i] == leg:  # each real leg holds half of its pair's force
                    mujoco.mj_jacSite(model, data, jac, None, ix.feet[i])
                    cols = ix.qvel[[i, 4 + i, 8 + i]]  # this leg's hip, thigh and calf
                    tau_leg = -jac[:, cols].T @ np.array([fx / 2, 0.0, fz / 2])
                    tau_cmd[[i, 4 + i, 8 + i]] = tau_leg
    return q_ref + (tau_cmd + C.KD * qd_ref) / C.KP


def simulate_jump(model, data, sol: JumpSolution, mode: str = "feedback", total_s: float = C.JUMP_TIME_LIMIT_S,
                  robot: JumpRobotParams | None = None, gains: FeedbackGains = FeedbackGains(),
                  settings: JumpSettings = JumpSettings(), start_shift_x: float = 0.0) -> SimResult:  # fmt: skip
    """Simulate the planned jump for ``total_s`` seconds (the jump time limit) and judge it.

    ``start_shift_x`` starts the robot that far ahead (+) or behind (-) the planned start,
    to test how forgiving the jump is; the controllers still follow the original plan.
    """
    import mujoco

    if mode not in ("replay", "feedforward", "feedback"):
        raise ValueError(mode)
    if robot is None:
        from .robot_params import jump_robot_params

        robot = jump_robot_params()
    ix = _Indices(model)
    ref = to_jump_reference(sol)
    q_ref = ref.joint_pos
    qd_ref = np.gradient(q_ref, ref.t, axis=0)
    qd_ref[-1] = 0.0  # standing still at the end
    tau_side = np.stack([np.interp(ref.t, sol.t, sol.torques[:, j]) for j in range(4)], axis=1)
    tau_ff = joints_12(tau_side)
    tau_ff[:, 0:4] = 0.0  # hips (sideways joints) only hold their default angles
    if mode == "replay":
        targets = q_ref
    else:
        targets = q_ref + (tau_ff + C.KD * qd_ref) / C.KP

    # start exactly where the plan starts, at rest
    mujoco.mj_resetData(model, data)
    set_robot_state(model, data, ref.base_pos[0] + np.array([start_shift_x, 0.0, 0.0]), ref.base_quat[0], q_ref[0])

    n_steps = int(round(total_s / C.SIM_DT))
    log = {k: [] for k in ("t", "qpos", "torque")}
    fail_time = None
    target = targets[0]
    for step in range(n_steps):
        if step % C.DECIMATION == 0:  # new targets at 50 Hz, held (zero-order hold) in between
            i = min(step // C.DECIMATION, len(targets) - 1)
            if mode == "feedback":
                target = _feedback_targets(model, data, ix, sol, robot, min(data.time, sol.t[-1]), q_ref[i],
                                           qd_ref[i], tau_ff[i], gains, settings)  # fmt: skip
            else:
                target = targets[i]
        tau = motor_torque(target, data.qpos[ix.qpos], data.qvel[ix.qvel])
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if fail_time is None:
            for c in data.contact[: data.ncon]:
                bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
                if bodies & ix.fail_bodies:
                    fail_time = data.time
                    break
        log["t"].append(data.time)
        log["qpos"].append(data.qpos.copy())
        log["torque"].append(tau.copy())

    # unpack the log
    qpos = np.array(log["qpos"])
    t = np.array(log["t"])
    quat = qpos[:, ix.root_q + 3 : ix.root_q + 7]
    w, x, y, z = quat.T
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))  # MuJoCo pitch: + = nose down (about +y)
    tilt = np.array([_tilt(q) for q in quat])
    base_pos = qpos[:, ix.root_q : ix.root_q + 3]
    base_speed = np.linalg.norm(np.gradient(base_pos, t, axis=0), axis=1)
    feet = np.zeros((len(t), 4, 3))
    for k in range(len(t)):  # foot positions need forward kinematics
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, data)
        feet[k] = data.site_xpos[ix.feet]
    joint_pos = qpos[:, ix.qpos]
    result = SimResult(
        mode=mode, t=t, qpos=qpos, base_pos=base_pos, pitch=pitch, tilt=tilt, base_speed=base_speed,
        joint_pos=joint_pos, joint_vel=np.gradient(joint_pos, t, axis=0), torque=np.array(log["torque"]),
        feet=feet, box_pos=qpos[:, ix.box_q : ix.box_q + 3], fail_contact_time=fail_time,
    )  # fmt: skip
    _judge(result, sol)
    return result


def _judge(r: SimResult, sol: JumpSolution):
    """Shared jump rules (constants.py)."""
    box_half = C.BOX_SIZE[0] / 2
    if r.fail_contact_time is not None:
        r.success, r.reason = False, f"trunk or head touched something at t = {r.fail_contact_time:.2f} s"
    elif r.tilt.max() > C.JUMP_MAX_TILT:
        k = int(np.argmax(r.tilt > C.JUMP_MAX_TILT))
        r.success, r.reason = False, f"fell over (tilt {np.degrees(r.tilt.max()):.0f} deg) at t = {r.t[k]:.2f} s"
    else:
        window = r.t >= r.t[-1] - C.JUMP_SUCCESS_SETTLE_S
        box_top = r.box_pos[window, 2] + C.BOX_SIZE[2] / 2
        on_top = np.abs(r.feet[window, :, 2] - (box_top[:, None] + _foot_radius())) < C.JUMP_FOOT_ON_SURFACE_TOL
        inside = np.abs(r.feet[window, :, 0] - r.box_pos[window, None, 0]) < box_half
        checks = {
            "all feet on the box top": bool((on_top & inside).all()),
            "upright (tilt < 20 deg)": bool((r.tilt[window] < C.JUMP_SUCCESS_MAX_TILT).all()),
            "still (speed < 0.2 m/s)": bool((r.base_speed[window] < C.JUMP_SUCCESS_MAX_SPEED).all()),
        }
        r.info["checks"] = checks
        r.success = all(checks.values())
        r.reason = "landed and settled on the box" if r.success else "not settled: " + ", ".join(
            k for k, ok in checks.items() if not ok)  # fmt: skip
    r.info.update(
        max_tilt_deg=float(np.degrees(r.tilt.max())),
        final_base=r.base_pos[-1].round(3).tolist(),
        box_moved_cm=float(100 * np.linalg.norm(r.box_pos[-1] - r.box_pos[0])),
        peak_torque=float(np.abs(r.torque).max()),
    )


def render_video(model, result: SimResult, sol: JumpSolution, path, camera: str = "side", slowdown: float = 3.0,
                 fps: int = 30, width: int = 1600, height: int = 800):  # fmt: skip
    """MP4 video (H.264, plays on phones) of a simulation, with time, phase and result written on it.

    The PLANNED foot paths are drawn as faint dots (whole plan, from the start), the ACTUAL
    foot paths of the simulation as solid dots (so far): any gap between them shows where
    the real robot leaves the plan. Blue = front feet, orange = rear feet.
    """
    import imageio_ffmpeg
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    from .optimizer import PHASES
    from .robot_params import jump_robot_params

    hip_y = jump_robot_params().hip_y
    leg_rgb = {0: (0.16, 0.47, 0.84), 1: (0.92, 0.41, 0.20)}
    planned = []  # (3D position, side-view leg) of the planned feet along the whole plan
    for k in range(0, len(sol.t), 2):
        for leg in (0, 1):
            for y in (hip_y, -hip_y):
                planned.append((np.array([sol.feet[k, leg, 0], y, sol.feet[k, leg, 1]]), leg))

    def add_dot(scn, pos, rgba, size):
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), pos,
                                np.eye(3).ravel(), np.array(rgba, dtype=np.float32))  # fmt: skip
            scn.ngeom += 1

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    frame_dt = 1.0 / (fps * slowdown)  # simulated time between frames
    times = np.arange(0.0, result.t[-1] + 1e-9, frame_dt)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        small = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-movflags", "+faststart"], macro_block_size=8)  # fmt: skip
    writer.send(None)
    verdict = ("SUCCESS: " if result.success else "FAIL: ") + result.reason
    for t in times:
        k = int(np.clip(np.searchsorted(result.t, t), 0, len(result.t) - 1))
        data.qpos[:] = result.qpos[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        scn = renderer.scene
        for pos, leg in planned:  # the plan: faint
            add_dot(scn, pos, (*leg_rgb[leg], 0.3), 0.005)
        for j in range(0, k + 1, 8):  # the simulation so far: solid
            for i in range(4):
                add_dot(scn, result.feet[j, i], (*leg_rgb[0 if i < 2 else 1], 1.0), 0.007)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        node = int(np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1))
        phase = PHASES[sol.phase[node]][0] if t <= sol.t[-1] else "standing"
        draw.text((24, 18), f"MuJoCo PHYSICS - Go2 jump with joint control ({result.mode} controller)", fill=(20, 20, 20),
                  font=font)  # fmt: skip
        draw.text((24, 58), f"t = {t:.2f} s   planned phase: {phase}   ({slowdown:g}x slower)   "
                  "feet: faint = planned, solid = actual", fill=(60, 60, 60), font=small)  # fmt: skip
        if t >= result.t[-1] - C.JUMP_SUCCESS_SETTLE_S:
            color = (12, 130, 12) if result.success else (200, 40, 40)
            draw.text((24, height - 48), verdict, fill=color, font=small)
        writer.send(np.asarray(img).tobytes())
    writer.close()
    renderer.close()
    return path


def _foot_radius() -> float:
    from .robot_params import jump_robot_params

    return jump_robot_params().foot_radius
