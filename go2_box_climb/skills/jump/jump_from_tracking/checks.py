"""Jump success and fall checks from the shared jump rules (Isaac-free, torch).

The rules themselves (limits and tolerances) are the shared ones in
common/constants.py, so this policy is judged exactly like the from-scratch one:

    fall     body tilt > JUMP_MAX_TILT (the trunk/head contact check needs the
             simulator's contact sensor and is done in the environment)
    success  at the same time, for JUMP_SUCCESS_SETTLE_S without a break:
               * tilt < JUMP_SUCCESS_MAX_TILT
               * base speed < JUMP_SUCCESS_MAX_SPEED
               * all 4 feet on the target surface: the foot (its lowest point, one foot
                 radius below its center) within JUMP_FOOT_ON_SURFACE_TOL of the surface
                 height, and inside the surface's outline (box top / platform top)

"All 4 feet" is this part's reading of the shared rule (constants.py does not say how
many feet); it is a parameter (``min_feet``) so it can follow a team decision.
"""

import torch

from go2_box_climb.common import constants as C

from .jump_math import quat_rotate_inverse, tilt_of
from .reference import check_jump

FEET_ON_TARGET_DEFAULT = 4  # how many feet must be on the target surface


def feet_on_target(
    jump: str, feet_pos: torch.Tensor, box_pos: torch.Tensor, box_quat: torch.Tensor, foot_radius: float
) -> torch.Tensor:
    """Which feet stand on the jump's target surface.

    Args:
        jump: "ground_to_box" (target: box top) or "box_to_platform" (target: platform top).
        feet_pos: (N, 4, 3) foot CENTERS in env frame [m].
        box_pos: (N, 3) box center in env frame [m].
        box_quat: (N, 4) box orientation (x, y, z, w).
        foot_radius: foot collision radius [m] (go2_geometry().foot_radius).
    Returns:
        (N, 4) bool.
    """
    check_jump(jump)
    bottom = feet_pos[..., 2] - foot_radius
    if jump == "ground_to_box":
        surface = box_pos[:, 2:3] + C.BOX_SIZE[2] / 2  # box top (box center + half its height)
        rel = quat_rotate_inverse(box_quat.unsqueeze(1).expand(-1, 4, -1), feet_pos - box_pos.unsqueeze(1))
        inside = (rel[..., 0].abs() <= C.BOX_SIZE[0] / 2) & (rel[..., 1].abs() <= C.BOX_SIZE[1] / 2)
    else:
        surface = torch.full_like(bottom[:, :1], C.PLATFORM_SIZE[2])  # the platform never moves
        x, y = feet_pos[..., 0], feet_pos[..., 1]
        inside = (
            (x >= C.PLATFORM_FRONT_X)
            & (x <= C.PLATFORM_FRONT_X + C.PLATFORM_SIZE[0])
            & ((y - C.PLATFORM_POS[1]).abs() <= C.PLATFORM_SIZE[1] / 2)
        )
    return inside & ((bottom - surface).abs() <= C.JUMP_FOOT_ON_SURFACE_TOL)


def success_now(
    base_quat: torch.Tensor, base_lin_vel: torch.Tensor, on_target: torch.Tensor, min_feet: int = FEET_ON_TARGET_DEFAULT
) -> torch.Tensor:
    """(N,) bool: all success conditions hold at this instant (the 1 s hold is counted by the caller)."""
    upright = tilt_of(base_quat) < C.JUMP_SUCCESS_MAX_TILT
    settled = torch.linalg.norm(base_lin_vel, dim=-1) < C.JUMP_SUCCESS_MAX_SPEED
    return upright & settled & (on_target.sum(dim=-1) >= min_feet)


def tilted_over(base_quat: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the body is tilted beyond the fall limit."""
    return tilt_of(base_quat) > C.JUMP_MAX_TILT


def settle_steps() -> int:
    """Policy steps the success conditions must hold without a break."""
    return int(round(C.JUMP_SUCCESS_SETTLE_S / C.POLICY_DT))
