"""Random start conditions of a jump attempt (Isaac-free, torch).

Drawn from the shared ranges in common/constants.py (the same ranges as the
box-placement study), in env frame:

    box     pushed against the platform with placement errors: gap BOX_GAP_RANGE,
            sideways offset BOX_LATERAL_RANGE, rotation BOX_YAW_RANGE. The gap is measured
            from the platform face to the box's NEAREST corner, so a rotated box never
            overlaps the platform.
    robot   in the reference's first pose (joint angles, base height, body tilt), with a
            heading offset drawn from JUMP_START_YAW, placed
                ground_to_box:   JUMP1_START_DISTANCE behind the middle of the box's front
                                 face (along -x), centered on it sideways; heading
                                 relative to +x (the jump direction)
                box_to_platform: where the reference starts relative to the nominal box
                                 (BOX_TARGET_POS), carried along with the actual box; heading
                                 relative to the BOX's heading (relative to +x, a robot on a
                                 box rotated by 15 deg could stand with a rear foot over the
                                 box edge)

``scale`` (0..1) narrows every range around its nominal value (box flush, centered,
square; robot at the reference's own start distance, heading straight): 0 = always
nominal, 1 = the full shared ranges. Training widens it gradually (a curriculum);
evaluation always uses 1.

``reference_state`` gives the robot's state at any time of the (anchored) reference,
with velocities, for starting an attempt part-way through the jump ("reference state
initialization", used in training only).

The training environment applies them at every reset (envs/jump_from_tracking/mdp.py).
"""

from dataclasses import dataclass

import torch

from go2_box_climb.common import constants as C

from .jump_math import quat_conj, quat_from_yaw, quat_mul, quat_rotate
from .reference import Anchor, ReferenceTrack, check_jump, target_anchor


@dataclass
class StartConditions:
    box_pos: torch.Tensor  # (N, 3) box center [m]
    box_quat: torch.Tensor  # (N, 4)
    box_gap: torch.Tensor  # (N,) platform face -> nearest box corner [m]
    box_offset_y: torch.Tensor  # (N,) sideways offset [m]
    box_yaw: torch.Tensor  # (N,) [rad]
    robot_pos: torch.Tensor  # (N, 3) base position [m]
    robot_quat: torch.Tensor  # (N, 4) base orientation
    robot_yaw: torch.Tensor  # (N,) heading [rad] (relative to +x)
    joint_pos: torch.Tensor  # (N, 12) the reference's first joint angles
    start_distance: torch.Tensor  # (N,) ground_to_box: base behind the box face [m] (0 for box_to_platform)
    anchor: Anchor  # where the reference is placed in this attempt (reference.py, "Anchoring")


def _u(n: int, lo_hi, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(float(lo_hi[0]), float(lo_hi[1]))


def _scaled(lo_hi, nominal: float, scale: float) -> tuple[float, float]:
    """The range ``lo_hi`` shrunk around ``nominal`` by ``scale`` (0 = just nominal, 1 = full range)."""
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return nominal + scale * (lo - nominal), nominal + scale * (hi - nominal)


def box_half_extent_x(box_yaw: torch.Tensor) -> torch.Tensor:
    """How far a box rotated by ``box_yaw`` reaches along x from its center [m]."""
    return C.BOX_SIZE[0] / 2 * torch.cos(box_yaw).abs() + C.BOX_SIZE[1] / 2 * torch.sin(box_yaw).abs()


def reference_start_distance(track: ReferenceTrack) -> float:
    """Jump 1: how far the reference starts behind the nominal box face, clipped to JUMP1_START_DISTANCE [m]."""
    d = (C.BOX_TARGET_POS[0] - C.BOX_SIZE[0] / 2) - float(track.start_pos[0])
    return min(max(d, float(C.JUMP1_START_DISTANCE[0])), float(C.JUMP1_START_DISTANCE[1]))


def sample_start(
    jump: str, track: ReferenceTrack, n: int, device: str | torch.device = "cpu", scale: float = 1.0
) -> StartConditions:
    """Draw ``n`` random start conditions for ``jump`` (env frame); ``scale`` narrows the ranges (see above)."""
    check_jump(jump)
    gap = _u(n, _scaled(C.BOX_GAP_RANGE, 0.0, scale), device)
    offset_y = _u(n, _scaled(C.BOX_LATERAL_RANGE, 0.0, scale), device)
    box_yaw = _u(n, _scaled(C.BOX_YAW_RANGE, 0.0, scale), device)
    box_pos = torch.zeros(n, 3, device=device)
    box_pos[:, 0] = C.PLATFORM_FRONT_X - gap - box_half_extent_x(box_yaw)
    box_pos[:, 1] = C.BOX_TARGET_POS[1] + offset_y
    box_pos[:, 2] = C.BOX_TARGET_POS[2]
    box_quat = quat_from_yaw(box_yaw)

    ref0 = track.at(torch.zeros(n, device=device))
    yaw = _u(n, _scaled(C.JUMP_START_YAW, 0.0, scale), device)
    robot_pos = torch.zeros(n, 3, device=device)
    distance = torch.zeros(n, device=device)
    if jump == "ground_to_box":
        back = torch.tensor([-C.BOX_SIZE[0] / 2, 0.0, 0.0], device=device).expand(n, 3)
        face = box_pos + quat_rotate(box_quat, back)  # middle of the box's front face
        distance = _u(n, _scaled(C.JUMP1_START_DISTANCE, reference_start_distance(track), scale), device)
        robot_pos[:, 0] = face[:, 0] - distance
        robot_pos[:, 1] = face[:, 1]
    else:
        rel = ref0.base_pos - torch.tensor(C.BOX_TARGET_POS, device=device)
        rel[:, 2] = 0.0
        robot_pos[:, :2] = (box_pos + quat_rotate(box_quat, rel))[:, :2]
        yaw = box_yaw + yaw  # heading relative to the box (see the module docstring)
    robot_pos[:, 2] = ref0.base_pos[:, 2]
    robot_quat = quat_mul(quat_from_yaw(yaw - track.start_yaw), ref0.base_quat)
    target_xy, target_yaw = target_anchor(jump, box_pos, box_quat)
    return StartConditions(
        box_pos=box_pos,
        box_quat=box_quat,
        box_gap=gap,
        box_offset_y=offset_y,
        box_yaw=box_yaw,
        robot_pos=robot_pos,
        robot_quat=robot_quat,
        robot_yaw=yaw,
        joint_pos=ref0.joint_pos,
        start_distance=distance,
        anchor=Anchor(start_xy=robot_pos[:, :2].clone(), start_yaw=yaw, target_xy=target_xy, target_yaw=target_yaw),
    )


@dataclass
class RefState:
    """The robot's state on the anchored reference at given times (batch of N, env frame)."""

    base_pos: torch.Tensor  # (N, 3) [m]
    base_quat: torch.Tensor  # (N, 4)
    lin_vel: torch.Tensor  # (N, 3) base linear velocity, world frame [m/s]
    ang_vel: torch.Tensor  # (N, 3) base angular velocity, world frame [rad/s]
    joint_pos: torch.Tensor  # (N, 12) [rad]
    joint_vel: torch.Tensor  # (N, 12) [rad/s]


def reference_state(track: ReferenceTrack, anchor: Anchor, t0: torch.Tensor, h: float = 0.01) -> RefState:
    """State on the anchored reference at times ``t0`` (N,), velocities by central differences (+-h).

    Where t0 = 0 (a normal start) the velocities are exactly zero: the robot starts standing still.
    """
    now = track.at_anchored(t0, anchor)
    a, b = track.at_anchored(t0 - h, anchor), track.at_anchored(t0 + h, anchor)
    span = ((t0 + h).clamp(max=track.duration) - (t0 - h).clamp(min=0.0)).clamp(min=1e-6).unsqueeze(-1)
    moving = (t0 > 0).float().unsqueeze(-1)
    dq = quat_mul(b.base_quat, quat_conj(a.base_quat))  # rotation from a to b, world frame
    dq = torch.where(dq[:, 3:4] < 0, -dq, dq)
    return RefState(
        base_pos=now.base_pos,
        base_quat=now.base_quat,
        lin_vel=moving * (b.base_pos - a.base_pos) / span,
        ang_vel=moving * 2.0 * dq[:, :3] / span,  # small-angle rotation vector / time
        joint_pos=now.joint_pos,
        joint_vel=moving * (b.joint_pos - a.joint_pos) / span,
    )
