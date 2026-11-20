"""The walking skill: follow a velocity command until told otherwise (Isaac-free).

This is the ``Skill`` the sequencer and the push skill (MPPI) use. It has no goal
of its own: whoever owns it changes the command (``set_command``) every step. It
reports FAILURE if the robot has fallen.
"""

import numpy as np

from go2_box_climb.common.skill import Skill, SkillStatus
from go2_box_climb.common.state import TaskState

from .walking_policy import DEFAULT_BODY_HEIGHT, DEFAULT_POLICY_PATH, WalkingPolicy, quat_rotate_inverse

# Fall check: the trunk is too low, or tilted more than this from upright.
FALL_MIN_HEIGHT = 0.15  # [m]  (standing height is about 0.32 m)
FALL_MAX_TILT = np.deg2rad(60.0)


def has_fallen(state: TaskState) -> bool:
    gravity_b = quat_rotate_inverse(state.base_quat, np.array([0.0, 0.0, -1.0]))
    tilt = np.arccos(np.clip(-gravity_b[2], -1.0, 1.0))
    return state.base_pos[2] < FALL_MIN_HEIGHT or tilt > FALL_MAX_TILT


class WalkSkill(Skill):
    name = "walk"

    def __init__(self, policy_path=DEFAULT_POLICY_PATH, device: str = "cpu"):
        self.policy = WalkingPolicy(policy_path, device)

    def set_command(self, vx: float, vy: float, wz: float, h: float = DEFAULT_BODY_HEIGHT) -> np.ndarray:
        """Body-frame velocity command and body height; returns them after clipping to the trained range."""
        return self.policy.set_command(vx, vy, wz, h)

    def reset(self, state: TaskState) -> None:
        self.policy.reset()
        self.policy.set_command(0.0, 0.0, 0.0)  # start by standing still

    def act(self, state: TaskState) -> np.ndarray:
        return self.policy.act(state)

    def status(self, state: TaskState) -> SkillStatus:
        return SkillStatus.FAILURE if has_fallen(state) else SkillStatus.RUNNING
