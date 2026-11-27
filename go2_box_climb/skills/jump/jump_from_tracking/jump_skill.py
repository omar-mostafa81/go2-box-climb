"""The TO-guided jump as a Skill (Isaac-free): what the sequencer calls.

    reset(state)   the jump starts from the current pose
    act(state)     12 joint actions (once per POLICY_DT)
    status(state)  RUNNING / SUCCESS / FAILURE with the shared jump rules (checks.py)

Difference to the training environment: a TaskState has no contact forces and no
foot positions. So the skill cannot see "trunk or head touches something"; it detects
falls by the body tilt only, and it computes the feet from the joint angles
(kinematics.py) instead of reading them from the simulator. The evaluation script
(scripts/jump_from_tracking/eval.py) uses the simulator's own checks.
"""

from pathlib import Path

import numpy as np

from go2_box_climb.common import constants as C
from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.common.skill import Skill, SkillStatus
from go2_box_climb.common.state import TaskState

from .checks import FEET_ON_TARGET_DEFAULT, feet_on_target, settle_steps, success_now, tilted_over
from .jump_policy import SKILL_NAMES, JumpTrackingPolicy, state_tensors
from .kinematics import foot_positions


class JumpTrackingSkill(Skill):
    def __init__(
        self,
        jump: str,
        policy_path: str | Path | None = None,
        reference_path: str | Path | None = None,
        device: str = "cpu",
        min_feet: int = FEET_ON_TARGET_DEFAULT,
    ):
        self.policy = JumpTrackingPolicy(jump, policy_path, reference_path, device)
        self.jump = jump
        self.name = SKILL_NAMES[jump]
        self.min_feet = min_feet
        self._foot_radius = go2_geometry().foot_radius
        self._ok_steps = 0

    def reset(self, state: TaskState) -> None:
        self.policy.reset(state)
        self._ok_steps = 0

    def act(self, state: TaskState) -> np.ndarray:
        return self.policy.act(state)

    def status(self, state: TaskState) -> SkillStatus:
        """Call once per policy step, after the step (it counts how long the landing has held)."""
        s = state_tensors(state)
        if bool(tilted_over(s["base_quat"])[0]):
            return SkillStatus.FAILURE
        feet = foot_positions(s["base_pos"], s["base_quat"], s["joint_pos"])
        on = feet_on_target(self.jump, feet, s["box_pos"], s["box_quat"], self._foot_radius)
        ok = bool(success_now(s["base_quat"], s["base_lin_vel"], on, self.min_feet)[0])
        self._ok_steps = self._ok_steps + 1 if ok else 0
        if self._ok_steps >= settle_steps():
            return SkillStatus.SUCCESS
        if self.policy.elapsed(state) >= C.JUMP_TIME_LIMIT_S - 1e-9:
            return SkillStatus.FAILURE
        return SkillStatus.RUNNING
