"""The interface every skill (walk/push, jump 1, jump 2) implements.

The sequencer only talks to skills through this interface, so a skill can be
swapped (e.g. TO-guided jump vs. RL-from-scratch jump) without changing the
sequencer.
"""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from .state import TaskState


class SkillStatus(Enum):
    RUNNING = "running"  # still executing
    SUCCESS = "success"  # reached its goal
    FAILURE = "failure"  # failed (fell, timed out, box misplaced, ...)


class Skill(ABC):
    """A skill is called once per policy step (every ``POLICY_DT`` seconds)."""

    #: Short unique name, used in logs (e.g. "push", "jump_ground_box").
    name: str = "skill"

    @abstractmethod
    def reset(self, state: TaskState) -> None:
        """Called once when the skill starts, with the current state."""

    @abstractmethod
    def act(self, state: TaskState) -> np.ndarray:
        """Return the 12 joint actions (see ``ACTION_SCALE`` in constants.py)."""

    @abstractmethod
    def status(self, state: TaskState) -> SkillStatus:
        """Return whether the skill is still running, succeeded or failed."""
