"""MPPI cost for the push: running cost (per step, with a per-term breakdown) and terminal cost.

All terms are in SI units (m, rad, m/s, rad/s) and are >= 0. Each weight lives in
``CostWeights`` so the cost can be tuned without touching code. Shapes: K rollouts.

Push point ("shopping cart" rule): to move the box along d (box center -> goal), stand on
the opposite side of the box and push along d. The push point is where the line from the
box center along -d leaves the box outline. It is recomputed for every rollout state;
approach and heading pull the robot to it. The box is square, so box yaw errors are taken
mod 90 deg: any side may end flush against the platform.
"""

from dataclasses import dataclass

import numpy as np

from go2_box_climb.common import constants as C

from .model import PlanarParams, head_point, wrap_angle, wrap_yaw90

GOAL = np.array(C.BOX_TARGET_POS[:2])  # box center flush against the platform, centered [m]


@dataclass
class CostWeights:
    box_pos: float = 50.0  # (box xy - goal)^2                          [1/m^2]
    box_x_l1: float = 20.0  # |box x - goal x|: stays strong when only a few cm short [1/m]
    box_y_l1: float = 20.0  # |box y - goal y|: same for the sideways error  [1/m]
    box_yaw: float = 20.0  # (box yaw mod 90 deg)^2, only within aim_zone of the goal  [1/rad^2]
    aim: float = 20.0  # (box yaw - direction from box to goal, mod 90 deg)^2, farther than aim_zone [1/rad^2]
    aim_zone: float = 0.15  # box x within this of the goal: square up instead of aiming [m]
    approach: float = 5.0  # distance from the head to the push point    [1/m]
    heading: float = 5.0  # (robot yaw - direction the box must move)^2 [1/rad^2]
    box_speed: float = 5.0  # box speed^2, only near the goal           [s^2/m^2]
    box_speed_near: float = 0.03  # "near the goal": box center within this of the goal [m]
    effort_vx: float = 0.5  # vx^2 (same price as sideways)                [s^2/m^2]
    effort_vy: float = 0.5  # vy^2                                       [s^2/m^2]
    effort_wz: float = 0.3  # wz^2                                       [s^2/rad^2]
    smooth: float = 2.0  # ||u_t - u_{t-1}||^2                           [mixed]
    corner: float = 500.0  # head within corner_margin of a corner of the side it touches, ^2  [1/m^2]
    corner_margin: float = 0.10  # [m]
    platform: float = 1000.0  # head at or past the platform face (indicator) [-]
    terminal_scale: float = 50.0  # terminal cost = this x (box_pos + box_yaw + aim) at the last state


def box_pos_cost(x: np.ndarray, w: CostWeights) -> np.ndarray:
    """Box position cost (K,): squared distance to the goal plus a linear term on the x error."""
    err = x[:, 3:5] - GOAL
    return w.box_pos * np.sum(err**2, axis=1) + w.box_x_l1 * np.abs(err[:, 0]) + w.box_y_l1 * np.abs(err[:, 1])


AT_GOAL = 0.02  # box center within this of the goal: no push point, approach and heading are 0 [m]


def push_point(x: np.ndarray, p: PlanarParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(push point (K, 2), d (K, 2), valid (K,)) for K states (K, 6).

    d: unit vector from the box center to the goal (the way the box must move).
    push point = box center - r d, with r the center-to-outline distance along d
    (0.30 m through a side center, up to 0.42 m through a corner of the 0.6 m box).
    valid: False when the box is within AT_GOAL of the goal (then d and the point are meaningless).
    """
    err = GOAL - x[:, 3:5]
    dist = np.linalg.norm(err, axis=1)
    valid = dist > AT_GOAL
    d = np.where(valid[:, None], err / np.maximum(dist, 1e-9)[:, None], np.array([1.0, 0.0]))  # any d when invalid
    c, s = np.cos(x[:, 5]), np.sin(x[:, 5])
    dbx, dby = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]  # d in the box frame
    with np.errstate(divide="ignore"):
        r = np.minimum(p.box_length / 2 / np.abs(dbx), p.box_width / 2 / np.abs(dby))
    return x[:, 3:5] - r[:, None] * d, d, valid


def box_yaw_costs(x: np.ndarray, w: CostWeights) -> tuple[np.ndarray, np.ndarray]:
    """(box_yaw, aim) costs (K,).

    Far from the goal the box should point AT the goal (aim), so pushing it straight
    ahead also fixes its sideways offset. Within aim_zone it should be square. The box is
    square, so the yaw error is taken mod 90 deg (any side may face the platform).
    """
    far = (GOAL[0] - x[:, 3]) > w.aim_zone
    to_goal = np.arctan2(GOAL[1] - x[:, 4], GOAL[0] - x[:, 3])
    return w.box_yaw * wrap_yaw90(x[:, 5]) ** 2 * ~far, w.aim * wrap_yaw90(x[:, 5] - to_goal) ** 2 * far


def running_cost(
    x: np.ndarray, x_prev: np.ndarray, u: np.ndarray, u_prev: np.ndarray, info: dict, p: PlanarParams, w: CostWeights
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cost of one step for K rollouts. info: from ``model.step`` (gap and lateral of the nearest side)."""
    pp, d, valid = push_point(x, p)
    yaw_cost, aim_cost = box_yaw_costs(x, w)
    box_speed = np.linalg.norm(x[:, 3:5] - x_prev[:, 3:5], axis=1) / p.dt
    near_goal = np.linalg.norm(x[:, 3:5] - GOAL, axis=1) < w.box_speed_near  # full xy distance
    corner_depth = np.maximum(np.abs(info["lateral"]) - (p.box_width / 2 - w.corner_margin), 0.0)
    close = info["gap"] < 0.05

    terms = {
        "box_pos": box_pos_cost(x, w),
        "box_yaw": yaw_cost,
        "aim": aim_cost,
        "approach": np.where(valid, w.approach * np.linalg.norm(head_point(x, p.head_x) - pp, axis=1), 0.0),
        "heading": np.where(valid, w.heading * wrap_angle(x[:, 2] - np.arctan2(d[:, 1], d[:, 0])) ** 2, 0.0),
        "box_speed": w.box_speed * box_speed**2 * near_goal,
        "effort": w.effort_vx * u[:, 0] ** 2 + w.effort_vy * u[:, 1] ** 2 + w.effort_wz * u[:, 2] ** 2,
        "smooth": w.smooth * np.sum((u - u_prev) ** 2, axis=1),
        "corner": w.corner * corner_depth**2 * close,
        "platform": w.platform * (head_point(x, p.head_x)[:, 0] >= p.wall_x),
    }
    return sum(terms.values()), terms


def terminal_cost(x: np.ndarray, w: CostWeights) -> np.ndarray:
    """Cost of the final state (K,): terminal_scale x (box position + box yaw + aim) cost."""
    yaw_cost, aim_cost = box_yaw_costs(x, w)
    return w.terminal_scale * (box_pos_cost(x, w) + yaw_cost + aim_cost)
