"""The task state: what every skill and the sequencer know about the world.

This is a plain-numpy snapshot (no Isaac imports), so planners such as MPPI, the
trajectory optimizer and the sequencer can be written and tested without the
simulator. The simulator side fills it in from Isaac Lab data.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import NUM_JOINTS


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n)


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])  # (x, y, z, w)


@dataclass
class TaskState:
    """State of one environment, expressed in that environment's frame.

    Positions [m], quaternions (x, y, z, w), linear velocities [m/s], angular
    velocities [rad/s], joint values in ``JOINT_NAMES`` order.
    """

    # Robot base
    base_pos: np.ndarray = field(default_factory=lambda: _zeros(3))
    base_quat: np.ndarray = field(default_factory=_identity_quat)
    base_lin_vel: np.ndarray = field(default_factory=lambda: _zeros(3))  # world frame
    base_ang_vel: np.ndarray = field(default_factory=lambda: _zeros(3))  # world frame
    # Robot joints
    joint_pos: np.ndarray = field(default_factory=lambda: _zeros(NUM_JOINTS))
    joint_vel: np.ndarray = field(default_factory=lambda: _zeros(NUM_JOINTS))
    # Box
    box_pos: np.ndarray = field(default_factory=lambda: _zeros(3))
    box_quat: np.ndarray = field(default_factory=_identity_quat)
    box_lin_vel: np.ndarray = field(default_factory=lambda: _zeros(3))
    box_ang_vel: np.ndarray = field(default_factory=lambda: _zeros(3))
    # Time since the episode started [s]
    time: float = 0.0
