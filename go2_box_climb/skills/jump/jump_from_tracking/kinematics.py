"""Foot positions from joint angles (forward kinematics; Isaac-free, torch).

Used where the simulator's own foot positions are not available: building the
placeholder reference (feet exactly on the ground / box / platform) and the
skill's success check from a TaskState (which has no foot positions).
In the training environment the simulator's foot positions are used instead.

Leg geometry comes from the Go2 USD through ``go2_geometry()`` (rule 13). Same
joint conventions as envs/locomotion/standing_pose.py:
    * hip (ab/adduction) joint: rotation about the body x axis,
    * thigh and calf joints: rotation about the y axis; thigh angle t is measured from
      straight down (positive = knee moves back), calf angle c relative to the thigh,
      so the foot relative to the thigh joint (in the hip frame) is
          x = -L1 sin(t) - L2 sin(t + c),   z = -L1 cos(t) - L2 cos(t + c).
The joint frames in the Go2 USD are assumed to have no extra rotation; the replay
script (scripts/jump_from_tracking/play.py --open_loop) prints the difference
to the simulator's feet so this can be checked.
"""

import torch

from go2_box_climb.common.robot_geometry import LEGS, go2_geometry

from .jump_math import quat_rotate


def foot_positions_base(joint_pos: torch.Tensor) -> torch.Tensor:
    """Foot CENTER positions in the base frame.

    Args:
        joint_pos: (N, 12) joint angles in JOINT_NAMES order (4 hips, 4 thighs, 4 calves).
    Returns:
        (N, 4, 3) foot centers, legs in FL, FR, RL, RR order [m].
    """
    g = go2_geometry()
    L1, L2 = g.thigh_length, g.calf_length
    feet = []
    for i, leg in enumerate(LEGS):
        hip, thigh, calf = joint_pos[:, i], joint_pos[:, 4 + i], joint_pos[:, 8 + i]
        off = torch.tensor(g.thigh_joint_offset[leg], dtype=joint_pos.dtype, device=joint_pos.device)
        # foot relative to the hip joint, in the hip frame (before the hip rotation)
        x = off[0] - L1 * torch.sin(thigh) - L2 * torch.sin(thigh + calf)
        y = off[1].expand_as(x)
        z = off[2] - L1 * torch.cos(thigh) - L2 * torch.cos(thigh + calf)
        # hip rotation about x
        ch, sh = torch.cos(hip), torch.sin(hip)
        p = torch.stack((x, ch * y - sh * z, sh * y + ch * z), dim=-1)
        feet.append(p + torch.tensor(g.hip_joint_pos[leg], dtype=joint_pos.dtype, device=joint_pos.device))
    return torch.stack(feet, dim=1)


def foot_positions(base_pos: torch.Tensor, base_quat: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
    """Foot centers in the same frame as ``base_pos`` (e.g. env frame). Shapes (N,3), (N,4), (N,12) -> (N,4,3)."""
    p_b = foot_positions_base(joint_pos)
    q = base_quat.unsqueeze(1).expand(-1, p_b.shape[1], -1)
    return base_pos.unsqueeze(1) + quat_rotate(q, p_b)


def stance_height(joint_pos: torch.Tensor) -> torch.Tensor:
    """Base height above a flat surface when the body is level and the lowest foot touches it [m].

    Args:
        joint_pos: (N, 12) joint angles.
    Returns:
        (N,) height of the base origin above the surface (foot center is one foot radius above it).
    """
    lowest = foot_positions_base(joint_pos)[..., 2].min(dim=1).values
    return -lowest + go2_geometry().foot_radius
