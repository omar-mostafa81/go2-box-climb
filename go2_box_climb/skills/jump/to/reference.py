"""Turn an optimized side-view jump into the shared JumpReference format. Part: TO.

The optimizer works with two side-view legs and uneven time steps. The tracking policy
needs all 12 joints on the policy's 50 Hz grid (POLICY_DT), so this step:
    * copies the front side-view leg to FL and FR, the rear one to RL and RR,
    * keeps the hip (ab/adduction) joints at their default angles (the jump is straight),
    * resamples everything every POLICY_DT seconds (linear interpolation),
    * turns the pitch into a (x, y, z, w) quaternion,
    * marks each foot's contact from the phase it is in.

The joint angles already come out of the optimizer, which keeps them consistent with
the foot positions (forward kinematics is a constraint), so no separate IK solve is
needed here. ``planar_model.leg_ik`` is the inverse used to check this in the tests.
"""

import numpy as np

from go2_box_climb.common import constants as C
from go2_box_climb.common.formats import FEET, JumpReference

from .optimizer import JumpSolution
from .planar_model import pitch_to_quat

# Side-view leg (0 = front pair, 1 = rear pair) of each foot in FEET order (FL, FR, RL, RR)
SIDE_LEG_OF_FOOT = {"FL": 0, "FR": 0, "RL": 1, "RR": 1}


def joints_12(side_joints: np.ndarray) -> np.ndarray:
    """(N, 4) side-view joints (front thigh, front calf, rear thigh, rear calf) -> (N, 12).

    Output is in JOINT_NAMES order: 4 hips, 4 thighs, 4 calves (FL, FR, RL, RR each).
    """
    side_joints = np.atleast_2d(side_joints)
    n = side_joints.shape[0]
    q = np.tile(C.DEFAULT_JOINT_POS, (n, 1))
    for i, foot in enumerate(FEET):
        leg = SIDE_LEG_OF_FOOT[foot]
        q[:, 4 + i] = side_joints[:, 2 * leg]  # thigh
        q[:, 8 + i] = side_joints[:, 2 * leg + 1]  # calf
    return q


def to_jump_reference(sol: JumpSolution, dt: float = C.POLICY_DT) -> JumpReference:
    """Resample the solution every ``dt`` seconds and pack it as a JumpReference."""
    t = np.arange(0.0, sol.t[-1] + 1e-9, dt)
    if t[-1] < sol.t[-1] - 1e-9:  # always include the final, standing state
        t = np.append(t, sol.t[-1])

    def interp(values: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t, sol.t, values[:, j]) for j in range(values.shape[1])], axis=1)

    base = interp(sol.base)  # x, z, pitch
    base_pos = np.stack([base[:, 0], np.zeros_like(t), base[:, 1]], axis=1)
    joint_pos = joints_12(interp(sol.joints))
    # contact: take the phase of the last optimizer node at or before each sample time
    node = np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1)
    side_contact = sol.contact[node]  # (T, 2)
    foot_contact = np.stack([side_contact[:, SIDE_LEG_OF_FOOT[f]] for f in FEET], axis=1)
    return JumpReference(
        t=t,
        base_pos=base_pos,
        base_quat=pitch_to_quat(base[:, 2]),
        joint_pos=joint_pos,
        foot_contact=foot_contact,
    )
