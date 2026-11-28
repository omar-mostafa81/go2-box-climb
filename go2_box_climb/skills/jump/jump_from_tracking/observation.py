"""The jump policy's observation (Isaac-free, torch).

ONE function builds it, used both by the training environment
(envs/jump_from_tracking) and by the skill that runs the trained policy
(jump_policy.py), so the two can never disagree.

Observation (OBS_DIM = 77 numbers, this order, no scaling; "body frame" = the robot's
current base frame, positions in the env frame of common/constants.py):

    [0:3]    base linear velocity, body frame [m/s]
    [3:6]    base angular velocity, body frame [rad/s]
    [6:9]    gravity direction in the body frame (unit vector; (0, 0, -1) when level)
    [9:21]   joint positions minus DEFAULT_JOINT_POS [rad], JOINT_NAMES order
    [21:33]  joint velocities [rad/s]
    [33:45]  previous action (raw network output)
    [45]     phase: time since the jump started / reference duration, clipped to [0, 1]
    [46:58]  reference joint positions at the NEXT step, minus DEFAULT_JOINT_POS [rad]
    [58:61]  reference base position at the next step minus the current one, body frame [m]
    [61:65]  reference orientation at the next step relative to the current one,
             quaternion (x, y, z, w) with w >= 0
    [65:69]  reference foot contacts at the next step (1 = on a surface), FEET order
    [69:72]  box center relative to the base, body frame [m]
    [72:74]  box heading relative to the robot's heading: cos, sin
    [74:77]  platform front edge (top of the face, centered) relative to the base, body frame [m]

"Next step" = time + POLICY_DT: the action chosen now is applied until then, and the
tracking rewards compare the robot with the reference at that time.
"""

import torch

from go2_box_climb.common import constants as C

from .jump_math import quat_conj, quat_mul, quat_rotate_inverse, wrap_angle, yaw_of
from .reference import Anchor, ReferenceTrack

OBS_DIM = 77

#: Slices of the observation (for tests and debugging).
OBS_SLICES = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "joint_pos": slice(9, 21),
    "joint_vel": slice(21, 33),
    "last_action": slice(33, 45),
    "phase": slice(45, 46),
    "ref_joint_pos": slice(46, 58),
    "ref_base_pos": slice(58, 61),
    "ref_orientation": slice(61, 65),
    "ref_foot_contact": slice(65, 69),
    "box_pos": slice(69, 72),
    "box_heading": slice(72, 74),
    "platform_edge": slice(74, 77),
}

#: Top edge of the platform's front face, centered (env frame) [m].
PLATFORM_EDGE = (C.PLATFORM_FRONT_X, C.PLATFORM_POS[1], C.PLATFORM_SIZE[2])


def jump_observation(
    *,
    track: ReferenceTrack,
    time: torch.Tensor,
    anchor: Anchor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    last_action: torch.Tensor,
    box_pos: torch.Tensor,
    box_quat: torch.Tensor,
) -> torch.Tensor:
    """Build the (N, OBS_DIM) observation.

    Args (batch of N; positions in env frame, velocities in world frame):
        track: the reference (on the same device as the tensors).
        time: (N,) time since the jump started [s].
        anchor: where the reference is placed in this attempt (robot start, landing target).
        base_pos, base_quat: (N, 3), (N, 4) base position and orientation (x, y, z, w).
        base_lin_vel, base_ang_vel: (N, 3) base velocities, world frame.
        joint_pos, joint_vel: (N, 12) in JOINT_NAMES order.
        last_action: (N, 12) previous network output.
        box_pos, box_quat: (N, 3), (N, 4) box center and orientation.
    """
    dev, dt = base_pos.device, base_pos.dtype
    q0 = torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=dt, device=dev)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=dt, device=dev).expand_as(base_pos)
    ref = track.at_anchored(time + C.POLICY_DT, anchor)

    ref_quat_rel = quat_mul(quat_conj(base_quat), ref.base_quat)
    ref_quat_rel = torch.where(ref_quat_rel[:, 3:4] < 0, -ref_quat_rel, ref_quat_rel)
    box_heading = wrap_angle(yaw_of(box_quat) - yaw_of(base_quat))
    edge = torch.tensor(PLATFORM_EDGE, dtype=dt, device=dev).expand_as(base_pos)
    phase = (time / track.duration).clamp(0.0, 1.0).unsqueeze(-1)

    parts = [
        quat_rotate_inverse(base_quat, base_lin_vel),
        quat_rotate_inverse(base_quat, base_ang_vel),
        quat_rotate_inverse(base_quat, down),
        joint_pos - q0,
        joint_vel,
        last_action,
        phase.to(dt),
        ref.joint_pos.to(dt) - q0,
        quat_rotate_inverse(base_quat, ref.base_pos.to(dt) - base_pos),
        ref_quat_rel.to(dt),
        ref.foot_contact.to(dt),
        quat_rotate_inverse(base_quat, box_pos - base_pos),
        torch.stack((torch.cos(box_heading), torch.sin(box_heading)), dim=-1),
        quat_rotate_inverse(base_quat, edge - base_pos),
    ]
    return torch.cat(parts, dim=-1)
