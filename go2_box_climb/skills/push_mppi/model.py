"""Planar robot + box model used inside MPPI (plain numpy, vectorized over K rollouts).

State per rollout (all in the env frame, meters / radians):
    x = [rx, ry, ryaw, bx, by, byaw]     robot base pose, box center pose
Control per rollout (robot body frame, the walking policy's command):
    u = [vx, vy, wz]                     [m/s, m/s, rad/s]

What the model does in one step of length dt:
    1. Robot: holonomic kinematics. The body-frame velocity is rotated into the world
       frame and integrated (the walking policy tracks commands well, see policies/walking).
    2. Contact: the "head point" sits head_x meters ahead of the base along the robot's
       heading. The box is square, so any of its 4 sides may be pushed. Sides are named in
       the box's own frame: 0 front (+x), 1 left (+y), 2 back (-x), 3 right (-y). If the head
       point is inside the box outline, the robot pushes the side it is nearest to.
    3. Push: the box slides along that side's inward normal (the push direction) by the
       penetration depth p. An off-center contact (lateral offset d from the side center,
       measured to the left of the push direction) also turns it:
           dyaw = -rot_gain * d * p / (c^2 + d^2)
       where c is the box's radius of gyration. This is the usual quasi-static
       "pushing a sliding object" approximation: a push through the center only
       translates, a push far from the center mostly rotates.
    4. Wall: the box cannot pass the platform face. Its front-most point is capped at
       wall_x. When pushed against the wall, a crooked box straightens toward the nearest
       multiple of 90 deg (any side may end flush).
    5. The robot cannot pass through the box: if the head is still inside after the box
       moved, it is moved back out along the pushed side's normal.
Only the head point touches the box: the robot's body and legs are not modeled.
"""

from dataclasses import dataclass

import numpy as np

from go2_box_climb.common import constants as C


@dataclass(frozen=True)
class PlanarParams:
    dt: float  # model time step [s]
    head_x: float  # head contact point ahead of the base origin [m]
    box_length: float = C.BOX_SIZE[0]  # along the push direction (box x) [m]
    box_width: float = C.BOX_SIZE[1]  # the face being pushed (box y) [m]
    wall_x: float = C.PLATFORM_FRONT_X  # platform face [m]
    rot_gain: float = 1.0  # scales how much off-center pushes rotate the box [-]

    @property
    def gyration_sq(self) -> float:
        """Radius of gyration squared of the box footprint [m^2]."""
        return (self.box_length**2 + self.box_width**2) / 12.0


def wrap_angle(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def head_point(x: np.ndarray, head_x: float) -> np.ndarray:
    """World (x, y) of the head contact point, shape (..., 2)."""
    return np.stack([x[..., 0] + head_x * np.cos(x[..., 2]), x[..., 1] + head_x * np.sin(x[..., 2])], axis=-1)


def head_in_box_frame(x: np.ndarray, head_x: float) -> np.ndarray:
    """Head point expressed in the box frame (origin at box center, x toward the platform)."""
    d = head_point(x, head_x) - x[..., 3:5]
    c, s = np.cos(x[..., 5]), np.sin(x[..., 5])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)


def box_front_extent(byaw, p: PlanarParams):
    """How far the box's front-most point is ahead of its center along world x [m]."""
    return p.box_length / 2 * np.abs(np.cos(byaw)) + p.box_width / 2 * np.abs(np.sin(byaw))


SIDE_NAMES = ("front", "left", "back", "right")  # box frame: +x, +y, -x, -y
_NORMALS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])  # outward normal of each side (box frame)


def wrap_yaw90(a):
    """Box yaw error for a square box: yaw mod 90 deg, in [-45, 45) deg [rad]."""
    return (np.asarray(a) + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _half_depths(p: PlanarParams) -> np.ndarray:
    """Distance from the box center to each side [m], shape (4,)."""
    return np.array([p.box_length, p.box_width, p.box_length, p.box_width]) / 2


def nearest_side(x: np.ndarray, p: PlanarParams) -> dict:
    """Where the head is relative to the box outline, arrays of shape (K,).

    side     index of the nearest side (0 front, 1 left, 2 back, 3 right; box frame)
    gap      signed distance from that side's line (> 0 outside, <= 0 inside the box) [m]
    lateral  head offset from that side's center, to the left of the push direction [m]
    dist     distance from the head to the nearest point on the box outline (0 inside) [m]
    """
    h = head_in_box_frame(x, p.head_x)  # (K, 2)
    half = _half_depths(p)
    g_all = h @ _NORMALS.T - half  # (K, 4) signed distance to each side's line
    side = np.argmax(g_all, axis=-1)
    gap = np.take_along_axis(g_all, side[..., None], axis=-1)[..., 0]
    n = _NORMALS[side]  # outward normal; push direction e = -n; "left of e" = (-e_y, e_x) = (n_y, -n_x)
    lateral = n[..., 1] * h[..., 0] - n[..., 0] * h[..., 1]
    outside = np.maximum(np.abs(h) - np.array([p.box_length, p.box_width]) / 2, 0.0)
    return {"side": side, "gap": gap, "lateral": lateral, "dist": np.linalg.norm(outside, axis=-1)}


def side_direction(byaw, side):
    """World angle of the push direction into `side` (the inward normal) [rad]."""
    return wrap_angle(byaw + side * np.pi / 2 + np.pi)


def face_gap(x: np.ndarray, p: PlanarParams) -> tuple[np.ndarray, np.ndarray]:
    """(gap, lateral) of the head relative to the nearest side.

    gap > 0: head is outside that side by that many meters; gap <= 0: touching / pushing.
    lateral: head offset from the side center, to the left of the push direction [m].
    """
    s = nearest_side(x, p)
    return s["gap"], s["lateral"]


def step(x: np.ndarray, u: np.ndarray, p: PlanarParams) -> tuple[np.ndarray, dict]:
    """Advance K states (K, 6) by one step with controls (K, 3).

    Returns the next states and an info dict of (K,) arrays:
        contact  bool, the robot pushed the box this step
        gap, lateral, side, dist  where the head is after the step (see nearest_side)
    """
    x = np.array(x, dtype=float, copy=True)
    rx, ry, ryaw = x[:, 0], x[:, 1], x[:, 2]

    # 1. robot kinematics (body-frame command -> world)
    c, s = np.cos(ryaw), np.sin(ryaw)
    x[:, 0] = rx + (c * u[:, 0] - s * u[:, 1]) * p.dt
    x[:, 1] = ry + (s * u[:, 0] + c * u[:, 1]) * p.dt
    x[:, 2] = wrap_angle(ryaw + u[:, 2] * p.dt)

    # 2. contact: head inside the box outline; the nearest side is the one being pushed
    ns = nearest_side(x, p)
    side, lat = ns["side"], ns["lateral"]
    contact = ns["gap"] < 0
    push = np.where(contact, -ns["gap"], 0.0)

    # 3. push: slide along the side's inward normal, rotate by the off-center offset
    e = side_direction(x[:, 5], side)  # world push direction (before the box turns)
    x[:, 3] += push * np.cos(e)
    x[:, 4] += push * np.sin(e)
    x[:, 5] = wrap_angle(x[:, 5] - p.rot_gain * lat * push / (p.gyration_sq + lat**2))

    # 4. wall: cap the front-most point at the platform face; straighten toward the nearest 90 deg
    max_x = p.wall_x - box_front_extent(x[:, 5], p)
    at_wall = x[:, 3] > max_x
    over = np.where(at_wall, x[:, 3] - max_x, 0.0)
    x[:, 3] -= over
    r = wrap_yaw90(x[:, 5])
    x[:, 5] = np.where(at_wall, wrap_angle(x[:, 5] - r + r * np.exp(-over / (p.box_length / 2))), x[:, 5])
    x[:, 3] = np.minimum(x[:, 3], p.wall_x - box_front_extent(x[:, 5], p))

    # 5. robot cannot stay inside the box: move it back out along the pushed side's outward normal
    h = head_in_box_frame(x, p.head_x)
    n = _NORMALS[side]
    g = np.sum(h * n, axis=-1) - _half_depths(p)[side]  # head's signed distance to the same side
    back = np.where(contact & (g < 0), -g, 0.0)
    out = x[:, 5] + side * np.pi / 2  # world angle of the outward normal
    x[:, 0] += back * np.cos(out)
    x[:, 1] += back * np.sin(out)

    ns = nearest_side(x, p)
    return x, {"contact": contact, "gap": ns["gap"], "lateral": ns["lateral"], "side": ns["side"], "dist": ns["dist"]}


def rollout(x0: np.ndarray, U: np.ndarray, p: PlanarParams) -> tuple[np.ndarray, dict]:
    """Roll out K control sequences U (K, T, 3) from one start state x0 (6,).

    Returns states (K, T+1, 6) (index 0 is x0) and info arrays of shape (K, T)
    (one entry per step, taken after that step).
    """
    K, T, _ = U.shape
    X = np.empty((K, T + 1, 6))
    X[:, 0] = x0
    types = {"contact": bool, "gap": float, "lateral": float, "side": int, "dist": float}
    info = {k: np.empty((K, T), dtype=d) for k, d in types.items()}
    for t in range(T):
        X[:, t + 1], inf = step(X[:, t], U[:, t], p)
        for k in info:
            info[k][:, t] = inf[k]
    return X, info
