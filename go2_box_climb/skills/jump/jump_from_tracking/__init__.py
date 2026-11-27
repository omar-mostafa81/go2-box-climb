"""Runs the trained TO-guided jump policy (Skill interface). Part: jump from tracking.

Loads the released policy from policies/, builds its observation from TaskState
exactly as during training (envs/jump_from_tracking), and reports SUCCESS / FAILURE.
No trajectory optimization at runtime.

Isaac-free (numpy + torch); the training environment imports the same modules:
    jump_math.py     quaternion helpers (x, y, z, w)
    reference.py     ReferenceTrack: load a JumpReference and look it up in time
                     (anchored to the robot's start); placeholder_reference for testing
    kinematics.py    foot positions from joint angles (Go2 geometry from the USD)
    start.py         random start conditions of a jump attempt (box placement, robot start)
    checks.py        success / fall checks with the shared jump rules
    observation.py   the policy's observation (one function for training and deployment)
    jump_policy.py   JumpTrackingPolicy: observation + exported network -> actions
    jump_skill.py    JumpTrackingSkill: the Skill the sequencer uses
"""

from .jump_policy import DEFAULT_POLICY_PATHS, SKILL_NAMES, JumpTrackingPolicy
from .jump_skill import JumpTrackingSkill
from .observation import OBS_DIM
from .reference import JUMPS, ReferenceTrack, load_reference, placeholder_path, placeholder_reference

__all__ = [
    "DEFAULT_POLICY_PATHS",
    "JUMPS",
    "OBS_DIM",
    "SKILL_NAMES",
    "JumpTrackingPolicy",
    "JumpTrackingSkill",
    "ReferenceTrack",
    "load_reference",
    "placeholder_path",
    "placeholder_reference",
]
