"""Fast checks of the shared contracts (no Isaac Sim needed): run `pytest`."""

import numpy as np
import pytest

from go2_box_climb.common import constants as C
from go2_box_climb.common.formats import TRIAL_LOG_COLUMNS, JumpReference, append_trial_log
from go2_box_climb.common.state import TaskState


def test_robot_contract():
    assert C.NUM_JOINTS == 12
    assert C.DEFAULT_JOINT_POS.shape == (12,)
    assert C.POLICY_DT == pytest.approx(0.02)
    # joints are ordered by type: 4 hips, 4 thighs, 4 calves
    assert all(n.endswith("hip_joint") for n in C.JOINT_NAMES[:4])
    assert all(n.endswith("thigh_joint") for n in C.JOINT_NAMES[4:8])
    assert all(n.endswith("calf_joint") for n in C.JOINT_NAMES[8:])


def test_scene_geometry():
    # box sits on the ground, platform is twice as high as the box
    assert C.BOX_START_POS[2] == pytest.approx(C.BOX_SIZE[2] / 2)
    assert C.PLATFORM_SIZE[2] == pytest.approx(2 * C.BOX_SIZE[2])
    # the push target puts the box face exactly on the platform face
    assert C.BOX_TARGET_POS[0] + C.BOX_SIZE[0] / 2 == pytest.approx(C.PLATFORM_FRONT_X)
    # box starts between the robot and the platform
    assert C.ROBOT_START_POS[0] < C.BOX_START_POS[0] < C.BOX_TARGET_POS[0]
    # the obstacle-free landing zone fits on the platform
    assert 0 < C.PLATFORM_CLEAR_DEPTH < C.PLATFORM_SIZE[0]


def test_task_state_defaults():
    s = TaskState()
    assert s.joint_pos.shape == (12,)
    assert np.allclose(s.base_quat, [0, 0, 0, 1])


def test_jump_reference_roundtrip(tmp_path):
    T = 5
    ref = JumpReference(
        t=np.linspace(0, 1, T),
        base_pos=np.zeros((T, 3)),
        base_quat=np.tile([0, 0, 0, 1.0], (T, 1)),
        joint_pos=np.tile(C.DEFAULT_JOINT_POS, (T, 1)),
        foot_contact=np.ones((T, 4), dtype=bool),
    )
    path = str(tmp_path / "ref.npz")
    ref.save(path)
    loaded = JumpReference.load(path)
    assert np.allclose(loaded.joint_pos, ref.joint_pos)


def test_jump_reference_rejects_bad_shape():
    with pytest.raises(ValueError):
        JumpReference(
            t=np.zeros(3), base_pos=np.zeros((3, 3)), base_quat=np.zeros((3, 4)),
            joint_pos=np.zeros((3, 11)), foot_contact=np.zeros((3, 4)),
        )  # fmt: skip


def test_trial_log(tmp_path):
    path = str(tmp_path / "log.csv")
    row = {k: 0 for k in TRIAL_LOG_COLUMNS}
    append_trial_log(path, row)
    append_trial_log(path, row)
    assert len(open(path).read().strip().splitlines()) == 3  # header + 2 rows


def test_jump_contract():
    # ranges are (low, high); the nominal push target (no error) lies inside the placement ranges
    for lo, hi in (C.JUMP1_START_DISTANCE, C.JUMP_START_YAW, C.BOX_GAP_RANGE, C.BOX_LATERAL_RANGE, C.BOX_YAW_RANGE):
        assert lo < hi
    assert C.BOX_GAP_RANGE[0] <= 0.0 and C.BOX_LATERAL_RANGE[0] < 0.0 < C.BOX_LATERAL_RANGE[1]
    # success is stricter than failure; the time limit leaves room to settle
    assert C.JUMP_SUCCESS_MAX_TILT < C.JUMP_MAX_TILT
    assert C.JUMP_SUCCESS_SETTLE_S < C.JUMP_TIME_LIMIT_S
    # the largest gap stays well below the box depth (otherwise the box is no longer a step)
    assert C.BOX_GAP_RANGE[1] < C.BOX_SIZE[0]
