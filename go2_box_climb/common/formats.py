"""File formats shared between parts of the project (Isaac-free).

1. ``JumpReference``: what trajectory optimization (TO) + IK produce and what the
   jump tracking policy follows. Saved as ``.npz`` in ``references/`` (REFERENCE_FILES).
2. Trial log: one CSV row per skill attempt. The studies and the sequencer's
   value function are built from these logs.
"""

import csv
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .constants import NUM_JOINTS

FEET = ("FL", "FR", "RL", "RR")  # order of feet in foot_contact

# Where the reference trajectories live (committed to git, beside policies/), one per jump.
REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_FILES = {
    "ground_to_box": REFERENCES_DIR / "jump_ground_to_box.npz",
    "box_to_platform": REFERENCES_DIR / "jump_box_to_platform.npz",
}


@dataclass
class JumpReference:
    """A jump trajectory sampled on a time grid (T samples), in env frame.

    Attributes:
        t: time stamps [s], shape (T,), starting at 0.
        base_pos: base position [m], shape (T, 3).
        base_quat: base orientation (x, y, z, w), shape (T, 4).
        joint_pos: joint angles [rad] in ``JOINT_NAMES`` order, shape (T, 12).
        foot_contact: True where the foot touches a surface, shape (T, 4), ``FEET`` order.
    """

    t: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    joint_pos: np.ndarray
    foot_contact: np.ndarray

    def __post_init__(self):
        T = len(self.t)
        expected = {"base_pos": (T, 3), "base_quat": (T, 4), "joint_pos": (T, NUM_JOINTS), "foot_contact": (T, 4)}
        for name, shape in expected.items():
            actual = np.shape(getattr(self, name))
            if actual != shape:
                raise ValueError(f"JumpReference.{name} has shape {actual}, expected {shape}")

    def save(self, path: str) -> None:
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: str) -> "JumpReference":
        data = np.load(path)
        return cls(**{f.name: data[f.name] for f in fields(cls)})


#: Columns of the trial log CSV. Positions are in env frame; yaw in radians.
TRIAL_LOG_COLUMNS = (
    "skill",  # skill name, e.g. "push", "jump_ground_box"
    "variant",  # e.g. "to_rl", "scratch", "fixed_order"
    "trial_id",
    "success",  # 1 or 0
    "duration_s",
    "box_gap",  # distance between box and platform face at start [m]
    "box_offset_y",  # sideways box offset at start [m]
    "box_yaw",  # box rotation at start [rad]
    "robot_x",
    "robot_y",
    "robot_yaw",
    "failure_reason",  # empty if success
)


def append_trial_log(path: str, row: dict) -> None:
    """Append one row to a trial log CSV (creates the file with a header if needed)."""
    missing = set(TRIAL_LOG_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Trial log row is missing columns: {sorted(missing)}")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row[k] for k in TRIAL_LOG_COLUMNS})
