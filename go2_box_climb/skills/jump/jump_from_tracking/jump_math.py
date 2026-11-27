"""Small rotation helpers for the jump-from-tracking part (Isaac-free, torch).

Quaternions are (x, y, z, w), the project convention (common/constants.py).
Every function works on batches: quaternions (..., 4), vectors (..., 3).
These are kept here (not taken from isaaclab.utils.math) so that the training
environment and the Isaac-free skill wrapper use exactly the same math.
"""

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Product a * b (apply b first, then a)."""
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Inverse rotation of a unit quaternion."""
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by q (body frame -> world frame when q is the body orientation)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of q (world frame -> body frame)."""
    return quat_rotate(quat_conj(q), v)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    """Heading angle (rotation about z) of q [rad]."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Pure rotation about z by ``yaw`` [rad]."""
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1)


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle of the rotation between two orientations [rad], in [0, pi]."""
    d = torch.abs(torch.sum(a * b, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(d)


def tilt_of(q: torch.Tensor) -> torch.Tensor:
    """Angle between the body's up axis and the world's up axis [rad] (roll and pitch together).

    Same quantity as Isaac Lab's ``bad_orientation`` termination: acos of the z component of the
    body z axis in world frame.
    """
    x, y, _, _ = q.unbind(-1)
    return torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))
