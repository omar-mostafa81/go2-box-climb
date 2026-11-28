"""Independent physics check of a planned jump. Part: TO.

The optimizer uses some shortcuts (smoothed box edges, checks only at time nodes and
halfway between them). This check does NOT reuse them:
    * the ground, box and platform are exact rectangles (no smoothing),
    * the robot is checked at 5 moments inside every time step,
    * every collision shape is used: the trunk box (as its outline), both head shapes,
      the hip motors, the whole thigh and calf (as thick line segments) and the feet,
    * the motor torques are compared with Isaac Lab's DC-motor torque-speed curve,
    * the ground forces are compared with the friction cone.
Numbers are reported so a teammate can see at a glance whether a plan is physical.
"""

from dataclasses import dataclass

import numpy as np

from go2_box_climb.common import constants as C

from . import planar_model as pm
from .optimizer import JumpSettings, JumpSolution
from .robot_params import JumpRobotParams


@dataclass
class PhysicsReport:
    worst_overlap: float  # deepest any robot shape goes into the scene [m] (<= 0 means no contact)
    worst_overlap_part: str  # which robot part and which object, and when
    torque_curve_use: float  # largest |torque| / |torque available at that speed| (<= 1 is possible)
    torque_curve_where: str
    max_joint_speed: float  # [rad/s]
    friction_use: float  # largest |forward force| / (friction coefficient x vertical force)

    def lines(self) -> list[str]:
        return [
            f"deepest overlap with the scene: {self.worst_overlap * 100:+.1f} cm ({self.worst_overlap_part})",
            f"motor torque-speed curve used: {self.torque_curve_use * 100:.0f}% ({self.torque_curve_where})",
            f"fastest joint: {self.max_joint_speed:.1f} rad/s (limit {C.VELOCITY_LIMIT:.0f})",
            f"friction used: {self.friction_use * 100:.0f}% of the cone",
        ]


def _rectangles(sol: JumpSolution):
    """The scene as rectangles (name, x0, x1, top): the ground, then one per step.

    Each step reaches from its front face to +inf (the next, higher step covers the rest),
    which is the same solid as box + platform side by side.
    """
    task = sol.task
    names = ("box", "platform") if task.name == "ground_to_box" else tuple(f"step {i}" for i in range(len(task.steps)))
    rects = [("ground", -np.inf, np.inf, task.start_height)]
    rects += [(names[i], x0, np.inf, top) for i, (x0, top) in enumerate(task.steps)]
    return rects


def _circle_overlap(cx, cz, radius, x0, x1, top) -> float:
    """How deep a circle reaches into the rectangle [x0, x1] x (-inf, top]. Negative = gap."""
    dx = max(x0 - cx, 0.0, cx - x1)
    dz = max(cz - top, 0.0)
    if dx == 0.0 and dz == 0.0:  # center inside the rectangle
        return radius + min(cx - x0, x1 - cx, top - cz)
    return radius - float(np.hypot(dx, dz))


def check_jump(sol: JumpSolution, robot: JumpRobotParams, settings: JumpSettings = JumpSettings(),
               substeps: int = 5) -> PhysicsReport:  # fmt: skip
    L1, L2, r = robot.thigh_length, robot.calf_length, robot.foot_radius
    hips = (robot.front_hip_x, robot.rear_hip_x)
    rects = _rectangles(sol)
    hx, hz = robot.trunk_half_size
    outline = [(x, z) for x in np.linspace(-hx, hx, 9) for z in (-hz, hz)] + [(-hx, 0.0), (hx, 0.0)]
    worst, where = -np.inf, ""

    def test(cx, cz, radius, part, t):
        nonlocal worst, where
        for name, x0, x1, top in rects:
            d = _circle_overlap(cx, cz, radius, x0, x1, top)
            if d > worst:
                worst, where = d, f"{part} into {name} at t={t:.2f} s"

    K = len(sol.t)
    for k in range(K - 1):
        for s in np.linspace(0.0, 1.0, substeps, endpoint=False):
            t = sol.t[k] + s * (sol.t[k + 1] - sol.t[k])
            x, z, th = (1 - s) * sol.base[k] + s * sol.base[k + 1]
            q = (1 - s) * sol.joints[k] + s * sol.joints[k + 1]
            for bx, bz in outline:
                test(*pm.point_world(x, z, th, bx, bz), 0.0, "trunk", t)
            for bx, bz, rad in (robot.head_upper, robot.head_lower):
                test(*pm.point_world(x, z, th, bx, bz), rad, "head", t)
            for leg in (0, 1):
                name = ("front", "rear")[leg]
                hip = np.array(pm.point_world(x, z, th, hips[leg], 0.0))
                hz, tilt = robot.thigh_joint_z, robot.leg_tilt_cos
                knee = np.array(pm.knee_world(x, z, th, hips[leg], q[2 * leg], L1, np, hz, tilt))
                foot = np.array(pm.foot_world(x, z, th, hips[leg], q[2 * leg], q[2 * leg + 1], L1, L2, np, hz, tilt))
                test(*hip, robot.hip_radius, f"{name} hip", t)
                for a in np.linspace(0.0, 1.0, 6):
                    test(*(hip + a * (knee - hip)), robot.thigh_half_thickness, f"{name} thigh", t)
                for a in np.linspace(0.0, 0.85, 6):  # the calf up to just above the foot
                    test(*(knee + a * (foot - knee)), robot.calf_radius, f"{name} calf", t)
                in_contact = sol.contact[k, leg] and sol.contact[min(k + 1, K - 1), leg]
                if not in_contact:  # a foot on a surface touches it on purpose
                    test(*foot, r - 0.002, f"{name} foot", t)  # 2 mm allowance for touching down

    # DC motor: available torque at speed w is 23.5 (1 - w/30) pushing with w, 23.5 (1 + w/30) against
    speed = sol.joint_vel
    up = np.minimum(C.EFFORT_LIMIT, C.EFFORT_LIMIT * (1 - speed / C.VELOCITY_LIMIT))
    down = np.maximum(-C.EFFORT_LIMIT, C.EFFORT_LIMIT * (-1 - speed / C.VELOCITY_LIMIT))
    use = np.where(sol.torques >= 0, sol.torques / np.maximum(up, 1e-6), sol.torques / np.minimum(down, -1e-6))
    k, j = np.unravel_index(np.argmax(use), use.shape)
    joint = ("front thigh", "front calf", "rear thigh", "rear calf")[j]

    mu = {"start": sol.task.friction_start, "target": sol.task.friction_target}
    friction = 0.0
    for k2 in range(K - 1):
        for leg in (0, 1):
            fz = sol.forces[k2, 2 * leg + 1]
            if sol.contact[k2, leg] and fz > 1.0:
                surface = "start" if sol.phase[k2] < 2 else "target"
                friction = max(friction, abs(sol.forces[k2, 2 * leg]) / (mu[surface] * settings.foot_friction * fz))
    return PhysicsReport(
        worst_overlap=float(worst),
        worst_overlap_part=where,
        torque_curve_use=float(use[k, j]),
        torque_curve_where=f"{joint}, {sol.torques[k, j]:.1f} N m at {speed[k, j]:.1f} rad/s, t={sol.t[k]:.2f} s",
        max_joint_speed=float(np.abs(speed).max()),
        friction_use=float(friction),
    )
