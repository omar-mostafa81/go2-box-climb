"""Standing joint pose for any body height (Isaac-free, torch).

The "stand still" penalty compares the joints with a reference standing pose. The
default pose (DEFAULT_JOINT_POS) is the stance at the default body height (0.32 m).
For another commanded height h, the legs must bend more or less, so the reference is
computed with simple 2-link leg geometry (inverse kinematics) in each leg's side
view (leg lengths read from the Go2 USD via go2_box_climb.common.robot_geometry):

    * every foot keeps the same fore/aft offset from its hip as in the default pose,
    * the hip-to-foot vertical distance changes by (h - DEFAULT_BODY_HEIGHT),
    * the hip (ab/adduction) joints keep their default angles.

At h = DEFAULT_BODY_HEIGHT this returns exactly DEFAULT_JOINT_POS.

Joint angle convention (Go2): thigh angle t from straight down, positive = knee moves
back; calf angle c relative to the thigh (negative = knee bent). With x forward and
z up, the foot relative to the thigh joint is
    x = -L1 sin(t) - L2 sin(t + c),   z = -L1 cos(t) - L2 cos(t + c).
"""

import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.skills.locomotion.walking_policy import DEFAULT_BODY_HEIGHT


def _lengths() -> tuple[float, float]:
    """Thigh and calf lengths [m], read from the Go2 USD (rule 13)."""
    g = go2_geometry()
    return g.thigh_length, g.calf_length


def _foot(t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    L1, L2 = _lengths()
    x = -L1 * torch.sin(t) - L2 * torch.sin(t + c)
    z = -L1 * torch.cos(t) - L2 * torch.cos(t + c)
    return x, z


def standing_joint_pos(height: torch.Tensor) -> torch.Tensor:
    """Reference standing joint angles for commanded body heights.

    Args:
        height: (N,) commanded body heights [m].
    Returns:
        (N, 12) joint angles in JOINT_NAMES order (hips, thighs, calves).
    """
    q0 = torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=height.dtype, device=height.device)
    t0, c0 = q0[4:8], q0[8:12]  # default thigh / calf angles (FL, FR, RL, RR)
    x0, z0 = _foot(t0, c0)  # default foot position relative to each thigh joint
    depth = (-z0).unsqueeze(0) + (height - DEFAULT_BODY_HEIGHT).unsqueeze(1)  # (N, 4) hip-to-foot drop
    x = x0.unsqueeze(0).expand_as(depth)
    L1, L2 = _lengths()
    r2 = x**2 + depth**2
    cos_c = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    c = -torch.acos(torch.clamp(cos_c, -1.0, 1.0))  # knee bent backwards (negative), like the default pose
    beta = torch.atan2(-x, depth)  # direction hip -> foot, measured from straight down
    t = beta - torch.atan2(L2 * torch.sin(c), L1 + L2 * torch.cos(c))
    hips = q0[0:4].unsqueeze(0).expand_as(t)
    return torch.cat([hips, t, c], dim=1)
