"""Side-view (sagittal plane) kinematics of the Go2 for jump planning. Plain functions.

The jump goes straight forward, so we plan it in the x-z plane (x forward, z up):
    * the trunk is one rigid body: position (x, z) of the base origin and pitch angle theta,
    * the front legs (FL + FR) move together and act as one leg, same for the rear legs,
    * each leg is two links: thigh (length L1) and calf (length L2, to the foot center).

Every function takes plain numbers. Pass ``xp=numpy`` (default) for numbers or
``xp=casadi`` for symbols inside the optimizer, so the same math is used in both places
and can be unit-tested.

Angle conventions (same as the Go2 / Isaac Lab, see envs/locomotion/standing_pose.py):
    * pitch theta: rotation about the body y axis (y points left). Positive = nose DOWN.
    * thigh angle t: measured from straight down, positive = the leg swings backward.
    * calf angle c: relative to the thigh, negative = knee bent (Go2 knees bend backward).
"""

import numpy as np


def leg_fk(t, c, L1: float, L2: float, xp=np):
    """Foot center relative to the thigh joint, in the BODY frame. Returns (x, z)."""
    x = -L1 * xp.sin(t) - L2 * xp.sin(t + c)
    z = -L1 * xp.cos(t) - L2 * xp.cos(t + c)
    return x, z


def leg_ik(x, z, L1: float, L2: float):
    """Inverse of ``leg_fk`` (numbers only), choosing the Go2's backward-bent knee.

    Args:
        x, z: foot center relative to the thigh joint in the body frame [m] (arrays ok).
    Returns:
        (t, c) thigh and calf angles [rad].
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    r2 = x**2 + z**2
    cos_c = np.clip((r2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    c = -np.arccos(cos_c)  # knee bent backward (negative), like the default pose
    beta = np.arctan2(-x, -z)  # direction thigh joint -> foot, measured from straight down
    t = beta - np.arctan2(L2 * np.sin(c), L1 + L2 * np.cos(c))
    return t, c


def leg_jacobian(t, c, L1: float, L2: float, xp=np):
    """d(foot x, foot z)/d(t, c) in the body frame, as ((dx/dt, dx/dc), (dz/dt, dz/dc))."""
    dx_dt = -L1 * xp.cos(t) - L2 * xp.cos(t + c)
    dx_dc = -L2 * xp.cos(t + c)
    dz_dt = L1 * xp.sin(t) + L2 * xp.sin(t + c)
    dz_dc = L2 * xp.sin(t + c)
    return (dx_dt, dx_dc), (dz_dt, dz_dc)


def body_to_world(bx, bz, theta, xp=np):
    """Rotate a body-frame vector (bx, bz) into the world frame by the pitch theta."""
    ct, st = xp.cos(theta), xp.sin(theta)
    return ct * bx + st * bz, -st * bx + ct * bz


def world_to_body(wx, wz, theta, xp=np):
    """Rotate a world-frame vector (wx, wz) into the body frame (inverse of body_to_world)."""
    ct, st = xp.cos(theta), xp.sin(theta)
    return ct * wx - st * wz, st * wx + ct * wz


def point_world(base_x, base_z, theta, px, pz, xp=np):
    """World position of a point fixed on the trunk at body coordinates (px, pz)."""
    wx, wz = body_to_world(px, pz, theta, xp)
    return base_x + wx, base_z + wz


def foot_world(base_x, base_z, theta, hip_x: float, t, c, L1: float, L2: float, xp=np, hip_z: float = 0.0,
               tilt: float = 1.0):  # fmt: skip
    """World position of a foot center.

    The thigh joint sits at body (hip_x, hip_z). ``tilt`` (cosine of the hip's sideways
    angle) shortens the leg's vertical reach: with the Go2's default hip angle the leg's
    plane leans outward a little, and the thigh joint, 9.55 cm to the side of the hip
    axis, is lifted by hip_z (see robot_params).
    """
    lx, lz = leg_fk(t, c, L1, L2, xp)
    return point_world(base_x, base_z, theta, hip_x + lx, hip_z + tilt * lz, xp)


def knee_world(base_x, base_z, theta, hip_x: float, t, L1: float, xp=np, hip_z: float = 0.0, tilt: float = 1.0):
    """World position of a knee (same hip_z / tilt as foot_world)."""
    return point_world(base_x, base_z, theta, hip_x - L1 * xp.sin(t), hip_z - tilt * L1 * xp.cos(t), xp)


def leg_torques(t, c, theta, fx, fz, L1: float, L2: float, xp=np, tilt: float = 1.0):
    """Motor torques (thigh, calf) that hold a ground force on one foot.

    Args:
        fx, fz: force the ground pushes on THIS foot, world frame [N].
        tilt: cosine of the hip's sideways angle (as in foot_world).
    Returns:
        (tau_thigh, tau_calf) [N m]. For a light leg in balance: tau = -J^T f (body frame).
    """
    fbx, fbz = world_to_body(fx, fz, theta, xp)
    (dx_dt, dx_dc), (dz_dt, dz_dc) = leg_jacobian(t, c, L1, L2, xp)
    tau_t = -(dx_dt * fbx + tilt * dz_dt * fbz)
    tau_c = -(dx_dc * fbx + tilt * dz_dc * fbz)
    return tau_t, tau_c


def pitch_to_quat(theta):
    """Base orientation for a pure pitch theta, as (x, y, z, w) quaternions. Shape (..., 4)."""
    theta = np.asarray(theta, dtype=float)
    half = 0.5 * theta
    zeros = np.zeros_like(theta)
    return np.stack([zeros, np.sin(half), zeros, np.cos(half)], axis=-1)
