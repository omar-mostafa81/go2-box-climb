"""Jump trajectory optimization (CasADi + IPOPT) in the side view. Part: TO.

Model (``planar_model.py``): the whole robot's mass sits in one rigid trunk, moving
with its center of mass (x, z) and pitch theta; the front pair and the rear pair of
legs each act as one two-link leg. The legs' own mass is lumped into the trunk (most of
it is the hip and thigh motors, within a few cm of the hips), but the torque needed to
SWING a leg is still counted against its motors.

The jump has four phases with a fixed contact order; the optimizer picks each length:

    1. push       all four feet on the start surface; the robot crouches and pushes
    2. rear push  front feet lift off, the rear feet keep pushing (the body pitches up)
    3. flight     no foot touches anything; the body flies on a ballistic arc
    4. landing    all four feet on the target surface; the robot catches itself and stops

Decision variables (per time node k): center of mass (x, z), pitch theta, their
velocities, and the four side-view joint angles. Per time step: the ground force on the
front pair and on the rear pair (x and z components). Plus the four phase lengths.

Constraints (all numbers from the robot USD and common/constants.py):
    * trunk dynamics (Newton-Euler about the center of mass)
    * feet fixed on the surface while in contact; forces only push, inside the friction
      cone (foot friction x surface friction, multiplied like Isaac Lab does)
    * motor torque = (holding the ground force) + (swinging the leg), inside the
      DC-motor torque-speed limit 23.5 N m x (1 - speed / 30 rad/s), as in the simulator
    * joint angle limits from the USD
    * every collision shape of the robot (trunk box, head, hips, thighs, knees, calves,
      swinging feet) stays above the ground, the box and the platform
    * start: standing still in the default pose; end: standing still in a balanced pose
      with all feet on the target surface

Cost: low effort (forces), smooth joints (speed and acceleration), little pitching, and a
final pose close to the default standing pose.
"""

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from go2_box_climb.common import constants as C

from . import planar_model as pm
from .robot_params import JumpRobotParams

GRAVITY = 9.81

# Phase names and which surface ("start"/"target"/None) each side-view leg touches
PHASES = (
    # name,       front contact, rear contact
    ("push", "start", "start"),
    ("rear push", None, "start"),
    ("flight", None, None),
    ("landing", "target", "target"),
)


@dataclass(frozen=True)
class JumpTask:
    """One jump over flat steps. The scene is a height profile: ``start_height`` before
    the first step, then each step ``(x, height)`` raises the surface from x onward."""

    name: str
    start_height: float  # surface the robot starts on [m]
    steps: tuple[tuple[float, float], ...]  # (x where it starts, height) for each step up [m]
    target_x: tuple[float, float]  # usable x range of the landing surface [m]
    target_height: float  # height of the landing surface [m]
    start_base_x: float  # base x at the start [m]
    friction_start: float  # friction of the start surface (combined with the foot by multiplying)
    friction_target: float  # friction of the landing surface


def ground_to_box_task(start_distance: float = 0.45) -> JumpTask:
    """Jump 1 for the nominal box placement (flush against the platform, centered, square).

    The scene has two steps: the box top (0.25 m) from the box front face, and the
    platform top (0.5 m) from the platform face, right behind the box.

    Args:
        start_distance: base x behind the box's front face [m]; within JUMP1_START_DISTANCE.
    """
    lo, hi = C.JUMP1_START_DISTANCE
    if not lo <= start_distance <= hi:
        raise ValueError(f"start_distance {start_distance} outside JUMP1_START_DISTANCE {C.JUMP1_START_DISTANCE}")
    box_front = C.BOX_TARGET_POS[0] - C.BOX_SIZE[0] / 2
    box_back = C.BOX_TARGET_POS[0] + C.BOX_SIZE[0] / 2
    return JumpTask(
        name="ground_to_box",
        start_height=0.0,
        steps=((box_front, C.BOX_SIZE[2]), (C.PLATFORM_FRONT_X, C.PLATFORM_SIZE[2])),
        target_x=(box_front, box_back),
        target_height=C.BOX_SIZE[2],
        start_base_x=box_front - start_distance,
        friction_start=C.GROUND_FRICTION,
        friction_target=C.BOX_FRICTION,
    )


@dataclass(frozen=True)
class JumpSettings:
    """Numbers that shape the optimization (not robot or scene facts)."""

    nodes: tuple[int, int, int, int] = (20, 12, 20, 20)  # time steps per phase
    duration_bounds: tuple[tuple[float, float], ...] = ((0.2, 1.0), (0.05, 0.4), (0.1, 0.6), (0.3, 1.0))
    duration_guess: tuple[float, float, float, float] = (0.4, 0.15, 0.3, 0.5)
    foot_friction: float = 1.0  # robot foot material; the jump env may randomize it
    friction_margin: float = 0.8  # use 80% of the combined friction coefficient
    torque_margin: float = 0.9  # use 90% of the motor torque curve
    joint_margin: float = 0.05  # stay this far inside the joint limits [rad]
    max_pitch: float = 0.6  # |pitch| limit [rad] (JUMP_MAX_TILT is 60 deg = 1.05 rad)
    body_clearance: float = 0.01  # gap between robot collision shapes and the scene [m]
    head_clearance: float = 0.025  # gap for the head [m]: the soft motors let the body sag ~1 cm (MuJoCo check)
    foot_clearance: float = 0.03  # extra gap under a swinging foot [m]
    land_margin: float = 0.05  # landed feet at least this far inside the landing surface [m]
    w_force: float = 1.0  # cost weight on (force / weight)^2
    w_force_rate: float = 1e-3  # cost weight on (force change / weight)^2 per second, while contacts stay the same
    w_joint_speed: float = 1e-3  # cost weight on joint speed^2
    w_joint_acc: float = 1e-6  # cost weight on joint acceleration^2
    w_pitch_rate: float = 1e-2  # cost weight on pitch rate^2
    w_end_pose: float = 1.0  # cost weight on (final joints - default pose)^2
    max_iter: int = 3000


@dataclass
class JumpSolution:
    """Optimized jump on the optimizer's time nodes (K nodes, K-1 steps)."""

    task: JumpTask
    t: np.ndarray  # (K,) time [s]
    phase: np.ndarray  # (K,) phase index into PHASES
    base: np.ndarray  # (K, 3) base origin x, z and pitch [m, m, rad]
    com: np.ndarray  # (K, 2) center of mass x, z [m]
    com_vel: np.ndarray  # (K, 3) center of mass x, z speed and pitch rate
    joints: np.ndarray  # (K, 4) front thigh, front calf, rear thigh, rear calf [rad]
    joint_vel: np.ndarray  # (K, 4) joint speeds during the step that starts at node k [rad/s]
    forces: np.ndarray  # (K, 4) front fx, fz, rear fx, fz for the step that starts at node k [N] (pair total)
    feet: np.ndarray  # (K, 2, 2) world (x, z) of the front and rear foot centers [m]
    knees: np.ndarray  # (K, 2, 2) world (x, z) of the front and rear knees [m]
    torques: np.ndarray  # (K, 4) per-leg motor torque (ground force + leg swing) [N m]
    contact: np.ndarray  # (K, 2) bool, front / rear pair touching a surface
    durations: np.ndarray  # (4,) phase durations [s]
    status: str
    info: dict = field(default_factory=dict)


def terrain_height(x, task: JumpTask, xp=ca, width=0.008, lead=0.025):
    """Smooth height profile of the scene: steps up at each ``task.steps`` x.

    Each step is a steep S-curve (tanh) centered ``lead`` before the edge, so with the
    defaults it is already 95.8% of the way up at the edge. Together with the 1 cm gap
    every robot shape keeps (``body_clearance``) nothing clips a corner; ``physics_check``
    confirms this with the exact, sharp edges.
    """
    h = task.start_height
    prev = task.start_height
    for x_step, height in task.steps:
        h = h + (height - prev) * 0.5 * (1.0 + xp.tanh((x - (x_step - lead)) / (2.0 * width)))
        prev = height
    return h


def body_circles(robot: JumpRobotParams) -> list[tuple[float, float, float]]:
    """Collision shapes fixed on the trunk, as side-view circles (body x, body z, radius).

    The trunk box (0.376 m x 0.114 m) is covered by a chain of 7 circles as tall as the
    box, with extra small circles on its bottom corners; the head is two circles.
    """
    hx, hz = robot.trunk_half_size
    circles = [(x, 0.0, hz) for x in np.linspace(-(hx - hz), hx - hz, 7)]
    circles += [(-hx + 0.01, -hz + 0.01, 0.01), (hx - 0.01, -hz + 0.01, 0.01)]  # bottom corners
    circles += [robot.head_upper, robot.head_lower]
    return circles


def solve_jump(task: JumpTask, robot: JumpRobotParams, settings: JumpSettings = JumpSettings()) -> JumpSolution:
    """Build and solve the jump optimization. Returns the solution on the time nodes."""
    S = settings
    L1, L2, r = robot.thigh_length, robot.calf_length, robot.foot_radius
    m, inertia = robot.mass, robot.pitch_inertia
    cox, coz = robot.com_offset
    hips = (robot.front_hip_x, robot.rear_hip_x)
    surf = {"start": task.start_height, "target": task.target_height}
    mu = {"start": task.friction_start * S.foot_friction * S.friction_margin,
          "target": task.friction_target * S.foot_friction * S.friction_margin}  # fmt: skip
    q_default = np.array([C.DEFAULT_JOINT_POS[4], C.DEFAULT_JOINT_POS[8], C.DEFAULT_JOINT_POS[6],
                          C.DEFAULT_JOINT_POS[10]])  # fmt: skip
    swing_inertia = (robot.thigh_swing_inertia, robot.calf_swing_inertia) * 2
    tau_sat, speed_max = C.EFFORT_LIMIT, C.VELOCITY_LIMIT
    weight = m * GRAVITY

    # phase of every node (the last node belongs to the last phase)
    phase_of = np.concatenate([np.full(n, i) for i, n in enumerate(S.nodes)] + [[len(PHASES) - 1]])
    K = len(phase_of)

    opti = ca.Opti()
    X = opti.variable(6, K)  # com x, com z, theta, their speeds
    Q = opti.variable(4, K)  # front thigh, front calf, rear thigh, rear calf
    QD = opti.variable(4, K)  # joint speeds
    QDD = opti.variable(4, K - 1)  # joint accelerations (constant over each step)
    F = opti.variable(4, K - 1)  # front fx, fz, rear fx, fz (pair totals)
    T = opti.variable(len(PHASES))  # phase durations
    dt = [T[phase_of[k]] / S.nodes[phase_of[k]] for k in range(K - 1)]

    def base(k):
        """Base origin from the center of mass (the COM sits at com_offset in the body frame)."""
        ox, oz = pm.body_to_world(cox, coz, X[2, k], ca)
        return X[0, k] - ox, X[1, k] - oz

    thigh_z, leg_tilt = robot.thigh_joint_z, robot.leg_tilt_cos  # default sideways hip angle (see robot_params)

    def foot(k, leg):
        bx, bz = base(k)
        return pm.foot_world(bx, bz, X[2, k], hips[leg], Q[2 * leg, k], Q[2 * leg + 1, k], L1, L2, ca, thigh_z,
                             leg_tilt)  # fmt: skip

    def knee(k, leg):
        bx, bz = base(k)
        return pm.knee_world(bx, bz, X[2, k], hips[leg], Q[2 * leg, k], L1, ca, thigh_z, leg_tilt)

    def contact_surface(k, leg):
        return PHASES[phase_of[k]][1 + leg]

    # The scene's step edges are softened in a first solve and made sharp in a second one,
    # which starts from the first answer (much faster than solving the sharp version directly).
    edge_width = opti.parameter()
    edge_lead = opti.parameter()

    def above_scene(px, pz, radius, extra=0.0):
        """A circle at (px, pz) stays clear of the scene below and in front of it."""
        ground = terrain_height(px + radius, task, width=edge_width, lead=edge_lead)
        opti.subject_to(pz >= ground + radius + S.body_clearance + extra)

    cost = 0
    for k in range(K - 1):
        # --- joints: acceleration -> speed -> angle (semi-implicit Euler) ---
        opti.subject_to(QD[:, k + 1] == QD[:, k] + dt[k] * QDD[:, k])
        opti.subject_to(Q[:, k + 1] == Q[:, k] + dt[k] * QD[:, k + 1])

        fx = (F[0, k], F[2, k])
        fz = (F[1, k], F[3, k])

        # --- dynamics of the center of mass and the pitch (semi-implicit Euler) ---
        torque_y = 0
        for leg in (0, 1):
            px, pz = foot(k, leg)
            torque_y += (pz - X[1, k]) * fx[leg] - (px - X[0, k]) * fz[leg]
        # The legs carry angular momentum too: swinging them one way turns the trunk the other
        # way (both pairs = 4 legs). Total momentum about the center of mass:
        # I theta_dot + sum over legs (I_thigh q_dot_thigh + I_calf q_dot_calf).
        leg_momentum_rate = 2 * sum(robot.thigh_swing_inertia_nominal * QDD[2 * leg, k]
                                    + robot.calf_swing_inertia * QDD[2 * leg + 1, k] for leg in (0, 1))  # fmt: skip
        acc = ca.vertcat((fx[0] + fx[1]) / m, (fz[0] + fz[1]) / m - GRAVITY, (torque_y - leg_momentum_rate) / inertia)
        v_next = X[3:6, k] + dt[k] * acc
        opti.subject_to(X[3:6, k + 1] == v_next)
        opti.subject_to(X[0:3, k + 1] == X[0:3, k] + dt[k] * v_next)

        # --- ground forces: only in contact, pushing, inside the friction cone ---
        for leg in (0, 1):
            s = contact_surface(k, leg)
            if s is None:
                opti.subject_to(fx[leg] == 0)
                opti.subject_to(fz[leg] == 0)
            else:
                opti.subject_to(fz[leg] >= 0)
                opti.subject_to(opti.bounded(-mu[s] * fz[leg], fx[leg], mu[s] * fz[leg]))

        # --- cost ---
        cost += dt[k] * S.w_force * ca.sumsqr(F[:, k] / weight)
        if k > 0 and phase_of[k] == phase_of[k - 1]:  # forces change smoothly unless a foot lands or lifts
            cost += S.w_force_rate * ca.sumsqr((F[:, k] - F[:, k - 1]) / weight) / dt[k]
        cost += dt[k] * S.w_joint_speed * ca.sumsqr(QD[:, k + 1])
        cost += dt[k] * S.w_joint_acc * ca.sumsqr(QDD[:, k])
        cost += dt[k] * S.w_pitch_rate * X[5, k] ** 2

    lim = (robot.thigh_limits_front, robot.calf_limits, robot.thigh_limits_rear, robot.calf_limits)
    for k in range(K):
        kf = min(k, K - 2)  # the last node keeps the last step's force and acceleration (standing still)
        speed = QD[:, k]

        # --- motors, inside the DC-motor curve. A foot on a surface: the motor holds the ground
        # force (the leg moves with the trunk, whose mass already includes it). A foot in the air:
        # the motor accelerates the leg. (Adding both would let them cancel and hide a too-big torque.)
        for leg in (0, 1):
            if contact_surface(kf, leg) is not None:
                taus = pm.leg_torques(Q[2 * leg, k], Q[2 * leg + 1, k], X[2, k], F[2 * leg, kf] / 2,
                                      F[2 * leg + 1, kf] / 2, L1, L2, ca, leg_tilt)  # fmt: skip
            else:
                taus = (swing_inertia[2 * leg] * QDD[2 * leg, kf], swing_inertia[2 * leg + 1] * QDD[2 * leg + 1, kf])
            for j, tau in zip((2 * leg, 2 * leg + 1), taus):
                # Isaac Lab DCMotor: tau in [S(-1 - w/w_max), S(1 - w/w_max)], and |tau| <= effort limit
                opti.subject_to(tau <= tau_sat * (S.torque_margin - speed[j] / speed_max))
                opti.subject_to(tau >= tau_sat * (-S.torque_margin - speed[j] / speed_max))
                opti.subject_to(opti.bounded(-S.torque_margin * tau_sat, tau, S.torque_margin * tau_sat))

        # --- joint limits, pitch limit ---
        for j in range(4):
            opti.subject_to(opti.bounded(lim[j][0] + S.joint_margin, Q[j, k], lim[j][1] - S.joint_margin))
        opti.subject_to(opti.bounded(-S.max_pitch, X[2, k], S.max_pitch))

        # --- feet: on the surface in contact, clear of the scene in the air ---
        for leg in (0, 1):
            px, pz = foot(k, leg)
            s = contact_surface(k, leg)
            if s is not None:
                opti.subject_to(pz == surf[s] + r)
                if k > 0 and contact_surface(k - 1, leg) == s:
                    px_prev, _ = foot(k - 1, leg)
                    opti.subject_to(px == px_prev)  # no sliding
            elif k > 0 and contact_surface(k - 1, leg) is not None:
                # first node after lift-off: only above the surface it just left
                opti.subject_to(pz >= surf[contact_surface(k - 1, leg)] + r)
            else:
                above_scene(px, pz, r, extra=S.foot_clearance - S.body_clearance)

        # --- the robot's body never touches the scene ---
        bx, bz = base(k)
        heads = {robot.head_upper, robot.head_lower}
        for cx, cz, rad in body_circles(robot):
            wx, wz = pm.point_world(bx, bz, X[2, k], cx, cz, ca)
            above_scene(wx, wz, rad, extra=S.head_clearance - S.body_clearance if (cx, cz, rad) in heads else 0.0)
        for leg in (0, 1):
            hx, hz = pm.point_world(bx, bz, X[2, k], hips[leg], 0.0, ca)
            kx, kz = knee(k, leg)
            fx_, fz_ = foot(k, leg)
            above_scene(hx, hz, robot.hip_radius)
            above_scene(0.5 * (hx + kx), 0.5 * (hz + kz), robot.thigh_half_thickness)  # middle of the thigh
            above_scene(kx, kz, robot.thigh_half_thickness)  # knee
            above_scene(0.5 * (kx + fx_), 0.5 * (kz + fz_), robot.calf_radius)  # middle of the calf

        if k == K - 1:
            continue
        for leg in (0, 1):
            s_now, s_next = contact_surface(k, leg), contact_surface(k + 1, leg)
            if s_now is not None and s_next is not None:
                continue
            # a foot or knee in the air moves (about) in a straight line between two nodes:
            # the middle of that line must also clear the scene, not only its ends
            (ax, az), (bx2, bz2) = foot(k, leg), foot(k + 1, leg)
            above_scene(0.5 * (ax + bx2), 0.5 * (az + bz2), r)
            (ax, az), (bx2, bz2) = knee(k, leg), knee(k + 1, leg)
            above_scene(0.5 * (ax + bx2), 0.5 * (az + bz2), robot.thigh_half_thickness)
            if s_now is None and s_next == "target":
                # land from above: just before touch-down the foot is over the target, higher than its top
                ax, az = foot(k, leg)
                opti.subject_to(az >= task.target_height + r + S.foot_clearance)
                opti.subject_to(ax >= task.target_x[0] + r)

    # --- start: standing still in the default pose ---
    opti.subject_to(base(0)[0] == task.start_base_x)
    opti.subject_to(X[3:6, 0] == 0)
    opti.subject_to(Q[:, 0] == ca.DM(q_default))
    opti.subject_to(QD[:, 0] == 0)
    # --- end: standing still (no speed, no acceleration) with all feet inside the landing surface ---
    opti.subject_to(X[3:6, K - 1] == 0)
    opti.subject_to(X[3:6, K - 2] == 0)
    opti.subject_to(QD[:, K - 1] == 0)
    opti.subject_to(QD[:, K - 2] == 0)
    for leg in (0, 1):
        px, _ = foot(K - 1, leg)
        opti.subject_to(opti.bounded(task.target_x[0] + S.land_margin + r, px, task.target_x[1] - S.land_margin - r))
    cost += S.w_end_pose * ca.sumsqr(Q[:, K - 1] - ca.DM(q_default))
    for p, (lo, hi) in enumerate(S.duration_bounds):
        opti.subject_to(opti.bounded(lo, T[p], hi))

    opti.minimize(cost)

    # --- initial guess: stand, rise along an arc, stand on the target ---
    s_lin = np.linspace(0.0, 1.0, K)
    x_end = 0.5 * (task.target_x[0] + task.target_x[1]) - 0.08
    z0 = task.start_height + 0.31
    opti.set_initial(X[0, :], task.start_base_x + (x_end - task.start_base_x) * s_lin)
    opti.set_initial(X[1, :], z0 + (task.target_height - task.start_height) * s_lin + 0.12 * np.sin(np.pi * s_lin))
    opti.set_initial(Q, np.tile(q_default[:, None], (1, K)))
    opti.set_initial(QD, 0)
    opti.set_initial(QDD, 0)
    f_guess = np.zeros((4, K - 1))
    for k in range(K - 1):
        n_contact = sum(contact_surface(k, leg) is not None for leg in (0, 1))
        for leg in (0, 1):
            if contact_surface(k, leg) is not None:
                f_guess[2 * leg + 1, k] = weight / n_contact
    opti.set_initial(F, f_guess)
    opti.set_initial(T, np.array(S.duration_guess))

    opti.solver("ipopt", {"print_time": False}, {"max_iter": S.max_iter, "print_level": 0, "tol": 1e-6})
    iterations = 0
    status = "solved"
    stats = {}
    stages = ((0.03, 0.06), (0.008, 0.025))  # stage 1: soft step edges (3 cm wide, rising 6 cm early); 2: sharp
    for i, (width, lead) in enumerate(stages):
        opti.set_value(edge_width, width)
        opti.set_value(edge_lead, lead)
        try:
            sol = opti.solve()
            stats = dict(opti.stats())
            iterations += stats["iter_count"]
            values = {v: np.array(sol.value(v)) for v in (X, Q, QD, QDD, F, T)}
            if i < len(stages) - 1:
                opti.set_initial(sol.value_variables())  # start the next stage from this answer
        except RuntimeError:
            stats = dict(opti.stats())
            iterations += stats["iter_count"]
            status = "failed: " + stats["return_status"]
            values = {v: np.array(opti.debug.value(v)) for v in (X, Q, QD, QDD, F, T)}
            break
    stats["iter_count"] = iterations
    arrays = [values[v] for v in (X, Q, QD, QDD, F, T)]
    return _collect(task, robot, S, arrays, phase_of, contact_surface, status, stats)


def _collect(task, robot, S, arrays, phase_of, contact_surface, status, stats) -> JumpSolution:
    """Turn the optimizer's numbers into a JumpSolution (base, feet, knees, torques, times)."""
    L1, L2 = robot.thigh_length, robot.calf_length
    hips = (robot.front_hip_x, robot.rear_hip_x)
    cox, coz = robot.com_offset
    Xv, Qv, QDv, QDDv, Fv, Tv = arrays
    Tv = Tv.ravel()
    K = Xv.shape[1]
    dts = np.array([Tv[phase_of[k]] / S.nodes[phase_of[k]] for k in range(K - 1)])
    t = np.concatenate([[0.0], np.cumsum(dts)])
    forces = np.concatenate([Fv, Fv[:, -1:]], axis=1).T  # the last node keeps the last step's force
    joint_vel = QDv.T
    qdd = np.concatenate([QDDv, QDDv[:, -1:]], axis=1).T
    swing_inertia = np.array([robot.thigh_swing_inertia, robot.calf_swing_inertia] * 2)

    base = np.zeros((K, 3))
    feet = np.zeros((K, 2, 2))
    knees = np.zeros((K, 2, 2))
    torques = np.zeros((K, 4))
    contact = np.zeros((K, 2), dtype=bool)
    for k in range(K):
        th = Xv[2, k]
        ox, oz = pm.body_to_world(cox, coz, th)
        x, z = Xv[0, k] - ox, Xv[1, k] - oz
        base[k] = (x, z, th)
        kf = min(k, K - 2)
        for leg in (0, 1):
            tq, cq = Qv[2 * leg, k], Qv[2 * leg + 1, k]
            thigh_z, leg_tilt = robot.thigh_joint_z, robot.leg_tilt_cos
            feet[k, leg] = pm.foot_world(x, z, th, hips[leg], tq, cq, L1, L2, np, thigh_z, leg_tilt)
            knees[k, leg] = pm.knee_world(x, z, th, hips[leg], tq, L1, np, thigh_z, leg_tilt)
            contact[k, leg] = contact_surface(k, leg) is not None
            if contact_surface(kf, leg) is not None:  # holding the ground force
                fx, fz = forces[k, 2 * leg], forces[k, 2 * leg + 1]
                torques[k, 2 * leg : 2 * leg + 2] = pm.leg_torques(tq, cq, th, fx / 2, fz / 2, L1, L2, np, leg_tilt)
            else:  # swinging the leg
                torques[k, 2 * leg : 2 * leg + 2] = qdd[k, 2 * leg : 2 * leg + 2] * swing_inertia[2 * leg : 2 * leg + 2]
    return JumpSolution(
        task=task,
        t=t,
        phase=phase_of.copy(),
        base=base,
        com=Xv[0:2].T,
        com_vel=Xv[3:6].T,
        joints=Qv.T,
        joint_vel=joint_vel,
        forces=forces,
        feet=feet,
        knees=knees,
        torques=torques,
        contact=contact,
        durations=Tv,
        status=status,
        info={"iterations": stats.get("iter_count"), "return_status": stats.get("return_status")},
    )
