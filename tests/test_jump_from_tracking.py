"""Fast checks of the jump-from-tracking part (no Isaac Sim needed): run `pytest`.

Covers everything that does not need the simulator: rotation helpers, reference
lookup and anchoring, the placeholder references, foot kinematics, start conditions,
success checks, the observation layout and the skill wrapper.
"""

import math

import numpy as np
import pytest
import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.formats import JumpReference
from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.common.skill import SkillStatus
from go2_box_climb.common.state import TaskState
from go2_box_climb.envs.locomotion.standing_pose import standing_joint_pos
from go2_box_climb.skills.jump.jump_from_tracking import JumpTrackingSkill
from go2_box_climb.skills.jump.jump_from_tracking import jump_math as M
from go2_box_climb.skills.jump.jump_from_tracking.checks import feet_on_target, settle_steps, success_now, tilted_over
from go2_box_climb.skills.jump.jump_from_tracking.kinematics import foot_positions, foot_positions_base, stance_height
from go2_box_climb.skills.jump.jump_from_tracking.observation import OBS_DIM, OBS_SLICES, jump_observation
from go2_box_climb.skills.jump.jump_from_tracking.reference import (
    JUMPS,
    Anchor,
    ReferenceTrack,
    jump_surfaces,
    nominal_target,
    placeholder_reference,
    target_anchor,
)
from go2_box_climb.skills.jump.jump_from_tracking.start import (
    box_half_extent_x,
    reference_start_distance,
    reference_state,
    sample_start,
)

IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


def _yaw_quat(yaw: float) -> torch.Tensor:
    return M.quat_from_yaw(torch.tensor([yaw]))


def _pitch_quat(pitch: float) -> torch.Tensor:
    return torch.tensor([[0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]])


def _simple_reference() -> JumpReference:
    """3 samples: base moves +0.2 m in x and up/down, joints 0 -> 1 -> 2, airborne in the middle."""
    return JumpReference(
        t=np.array([0.0, 0.1, 0.2]),
        base_pos=np.array([[0.0, 0.0, 0.3], [0.1, 0.0, 0.4], [0.2, 0.0, 0.3]]),
        base_quat=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        joint_pos=np.stack([np.full(12, v) for v in (0.0, 1.0, 2.0)]),
        foot_contact=np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
    )


def _feet_bottom(base_pos, base_quat, joint_pos) -> torch.Tensor:
    return foot_positions(base_pos, base_quat, joint_pos)[..., 2] - go2_geometry().foot_radius


# --- rotation helpers ---------------------------------------------------------------


def test_quaternion_helpers():
    q = M.quat_from_yaw(torch.tensor([0.7, -2.0]))
    v = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    assert torch.allclose(M.quat_rotate_inverse(q, M.quat_rotate(q, v)), v, atol=1e-6)
    assert torch.allclose(M.yaw_of(q), torch.tensor([0.7, -2.0]), atol=1e-6)
    # 90 deg to the left turns +x into +y
    assert torch.allclose(M.quat_rotate(_yaw_quat(math.pi / 2), torch.tensor([[1.0, 0.0, 0.0]])),
                          torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)  # fmt: skip
    assert torch.allclose(M.yaw_of(M.quat_mul(_yaw_quat(0.3), _yaw_quat(0.4))), torch.tensor([0.7]), atol=1e-6)
    assert M.quat_angle(_yaw_quat(0.3), _yaw_quat(-0.2)).item() == pytest.approx(0.5, abs=1e-4)
    assert M.tilt_of(_yaw_quat(1.0)).item() == pytest.approx(0.0, abs=1e-6)
    assert M.tilt_of(_pitch_quat(math.pi / 2)).item() == pytest.approx(math.pi / 2, abs=1e-5)


# --- reference lookup ----------------------------------------------------------------


def test_reference_lookup():
    track = ReferenceTrack(_simple_reference())
    s = track.at(torch.tensor([-1.0, 0.0, 0.04, 0.1, 0.16, 5.0]))
    # before the start: first sample; after the end: last sample; in between: linear
    assert torch.allclose(s.joint_pos[:, 0], torch.tensor([0.0, 0.0, 0.4, 1.0, 1.6, 2.0]), atol=1e-5)
    assert torch.allclose(s.base_pos[2], torch.tensor([0.04, 0.0, 0.34]), atol=1e-5)
    assert s.foot_contact[:, 0].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]  # nearest sample
    assert torch.allclose(torch.linalg.norm(s.base_quat, dim=-1), torch.ones(6))


def test_reference_must_be_valid():
    ref = _simple_reference()
    ref.t = np.array([0.05, 0.1, 0.2])
    with pytest.raises(ValueError):
        ReferenceTrack(ref)
    ref.t = np.array([0.0, 0.2, 0.1])
    with pytest.raises(ValueError):
        ReferenceTrack(ref)


def test_reference_anchoring():
    track = ReferenceTrack(_simple_reference())  # no jump given: nominal target point (0, 0)
    assert track.landing_time == pytest.approx(0.2)  # airborne at 0.1 s, all feet down again at 0.2 s
    # robot starts at (1, 2) facing +y; the landing target is at (5, 5), not rotated
    anchor = Anchor(
        start_xy=torch.tensor([[1.0, 2.0]]), start_yaw=torch.tensor([math.pi / 2]),
        target_xy=torch.tensor([[5.0, 5.0]]), target_yaw=torch.tensor([0.0]),
    )  # fmt: skip
    s0 = track.at_anchored(torch.tensor([0.0]), anchor)  # start: where the robot stands, its heading
    assert torch.allclose(s0.base_pos, torch.tensor([[1.0, 2.0, 0.3]]), atol=1e-6)
    assert M.yaw_of(s0.base_quat).item() == pytest.approx(math.pi / 2, abs=1e-6)
    # landing: placed relative to the target (planned 0.2 m from the nominal target point)
    s2 = track.at_anchored(torch.tensor([0.2]), anchor)
    assert torch.allclose(s2.base_pos, torch.tensor([[5.2, 5.0, 0.3]]), atol=1e-6)
    # half way to the landing: half from the start-anchored path (1, 2.1), half from the target one (5.1, 5)
    s1 = track.at_anchored(torch.tensor([0.1]), anchor)
    assert torch.allclose(s1.base_pos, torch.tensor([[3.05, 3.55, 0.4]]), atol=1e-5)
    # heading stays the robot's own all the way
    assert M.yaw_of(s2.base_quat).item() == pytest.approx(math.pi / 2, abs=1e-6)


def test_target_anchor():
    box_pos, box_quat = torch.tensor([[1.6, 0.1, 0.125]]), _yaw_quat(0.2)
    xy, yaw = target_anchor("ground_to_box", box_pos, box_quat)  # jump 1 lands relative to the actual box
    assert torch.allclose(xy, box_pos[:, :2]) and yaw.item() == pytest.approx(0.2, abs=1e-6)
    xy, yaw = target_anchor("box_to_platform", box_pos, box_quat)  # jump 2: the platform never moves
    assert torch.allclose(xy, torch.tensor([nominal_target("box_to_platform")])) and yaw.item() == 0.0
    # with the box exactly at its nominal place, target anchoring changes nothing at the landing
    track = ReferenceTrack(placeholder_reference("ground_to_box"), jump="ground_to_box")
    ref0 = track.at(torch.zeros(1))
    nominal = Anchor(ref0.base_pos[:, :2], M.yaw_of(ref0.base_quat), torch.tensor([nominal_target("ground_to_box")]),
                     torch.zeros(1))  # fmt: skip
    end = torch.tensor([track.duration])
    assert torch.allclose(track.at_anchored(end, nominal).base_pos, track.at(end).base_pos, atol=1e-6)


# --- kinematics ---------------------------------------------------------------------


def test_kinematics_matches_the_walking_standing_pose():
    q0 = torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=torch.float64).unsqueeze(0)
    feet = foot_positions_base(q0)[0]
    assert feet[0, 0] > feet[2, 0] and feet[1, 0] > feet[3, 0]  # front feet ahead of rear feet
    assert feet[0, 1].item() == pytest.approx(-feet[1, 1].item(), abs=1e-4)  # left/right mirrored
    assert 0.25 < stance_height(q0).item() < 0.40
    # standing_pose.py (walking part) lowers every foot by the height change: same in our kinematics
    lower = standing_joint_pos(torch.tensor([0.30], dtype=torch.float64))
    assert (stance_height(lower) - stance_height(q0)).item() == pytest.approx(-0.02, abs=1e-3)


# --- placeholder references ------------------------------------------------------------


@pytest.mark.parametrize("jump", JUMPS)
def test_placeholder_reference(jump):
    ref = placeholder_reference(jump)
    track = ReferenceTrack(ref)
    assert ref.t[0] == 0.0 and np.allclose(np.diff(ref.t), C.POLICY_DT)
    assert ref.foot_contact[0].all() and ref.foot_contact[-1].all() and (~ref.foot_contact.any(axis=1)).any()
    s0, s1 = jump_surfaces(jump)
    ends = track.at(torch.tensor([0.0, track.duration]))
    bottom = _feet_bottom(ends.base_pos, ends.base_quat, ends.joint_pos)
    assert bottom[0].min().item() == pytest.approx(s0, abs=1e-4)  # lowest foot exactly on the start surface
    assert bottom[1].min().item() == pytest.approx(s1, abs=1e-4)  # ... and on the target surface at the end
    # the end pose counts as landed on the target (nominal box), the start pose as standing on the start surface
    box_pos, box_quat = torch.tensor([C.BOX_TARGET_POS] * 2), IDENTITY.repeat(2, 1)
    feet = foot_positions(ends.base_pos, ends.base_quat, ends.joint_pos)
    r = go2_geometry().foot_radius
    assert feet_on_target(jump, feet, box_pos, box_quat, r)[1].all()
    if jump == "box_to_platform":
        assert feet_on_target("ground_to_box", feet, box_pos, box_quat, r)[0].all()


# --- start conditions ----------------------------------------------------------------


@pytest.mark.parametrize("jump", JUMPS)
def test_start_conditions(jump):
    torch.manual_seed(0)
    track = ReferenceTrack(placeholder_reference(jump))
    s = sample_start(jump, track, 512)
    # the box never overlaps the platform: its nearest corner is exactly `gap` in front of the face
    front = s.box_pos[:, 0] + box_half_extent_x(s.box_yaw)
    assert torch.allclose(C.PLATFORM_FRONT_X - front, s.box_gap, atol=1e-5)
    assert (s.box_gap >= C.BOX_GAP_RANGE[0]).all() and (s.box_gap <= C.BOX_GAP_RANGE[1]).all()
    assert (s.box_yaw.abs() <= C.BOX_YAW_RANGE[1] + 1e-6).all()
    heading = s.robot_yaw if jump == "ground_to_box" else s.robot_yaw - s.box_yaw  # jump 2: relative to the box
    assert (heading.abs() <= C.JUMP_START_YAW[1] + 1e-6).all()
    assert torch.allclose(M.yaw_of(s.robot_quat), s.robot_yaw, atol=1e-5)
    assert torch.allclose(s.robot_pos[:, 2], track.base_pos[0, 2].expand(512))
    # the reference is placed from the robot's start to the actual landing target
    assert torch.allclose(s.anchor.start_xy, s.robot_pos[:, :2]) and torch.allclose(s.anchor.start_yaw, s.robot_yaw)
    target_xy, target_yaw = target_anchor(jump, s.box_pos, s.box_quat)
    assert torch.allclose(s.anchor.target_xy, target_xy) and torch.allclose(s.anchor.target_yaw, target_yaw)
    if jump == "ground_to_box":
        face_x = s.box_pos[:, 0] - C.BOX_SIZE[0] / 2 * torch.cos(s.box_yaw)
        assert torch.allclose(face_x - s.robot_pos[:, 0], s.start_distance, atol=1e-5)
        assert (s.start_distance >= C.JUMP1_START_DISTANCE[0]).all()
    else:  # the robot starts standing on the (displaced) box
        feet = foot_positions(s.robot_pos, s.robot_quat, s.joint_pos)
        on_box = feet_on_target("ground_to_box", feet, s.box_pos, s.box_quat, go2_geometry().foot_radius)
        assert on_box.all()


def test_start_conditions_curriculum():
    track = ReferenceTrack(placeholder_reference("ground_to_box"), jump="ground_to_box")
    s = sample_start("ground_to_box", track, 64, scale=0.0)  # nominal: box flush, centered, square
    assert torch.allclose(s.box_gap, torch.zeros(64)) and torch.allclose(s.box_offset_y, torch.zeros(64))
    assert torch.allclose(s.box_yaw, torch.zeros(64)) and torch.allclose(s.robot_yaw, torch.zeros(64))
    assert torch.allclose(s.start_distance, torch.full((64,), reference_start_distance(track)))
    assert torch.allclose(s.robot_pos[:, :2], track.base_pos[0, :2].expand(64, 2), atol=1e-5)  # the planned start
    half = sample_start("ground_to_box", track, 512, scale=0.5)
    assert (half.box_gap <= C.BOX_GAP_RANGE[1] / 2 + 1e-6).all()


def test_reference_state():
    track = ReferenceTrack(_simple_reference())
    zeros2, zero = torch.zeros(1, 2), torch.zeros(1)
    identity = Anchor(zeros2, zero, zeros2.clone(), zero.clone())  # places the reference exactly as planned
    s = reference_state(track, identity, torch.tensor([0.05]))
    # the reference moves 0.1 m forward and up in 0.1 s, joints 0 -> 1 rad in 0.1 s
    assert torch.allclose(s.lin_vel, torch.tensor([[1.0, 0.0, 1.0]]), atol=1e-4)
    assert torch.allclose(s.joint_vel, torch.full((1, 12), 10.0), atol=1e-3)
    assert torch.allclose(s.ang_vel, torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(s.base_pos, track.at(torch.tensor([0.05])).base_pos, atol=1e-6)
    still = reference_state(track, identity, torch.tensor([0.0]))  # a normal start: standing still
    assert still.lin_vel.abs().max() == 0 and still.joint_vel.abs().max() == 0


# --- success / fall checks -----------------------------------------------------------------


def test_success_checks():
    r = 0.02
    box_pos, top = torch.tensor([C.BOX_TARGET_POS]), C.BOX_SIZE[2]
    feet = torch.tensor([[[1.6, 0.1, top + r], [1.6, -0.1, top + r], [1.8, 0.1, top + r], [1.3, 0.0, top + r]]])
    assert feet_on_target("ground_to_box", feet, box_pos, IDENTITY, r).tolist() == [[True, True, True, False]]
    high = feet.clone()
    high[..., 2] += 0.05  # 5 cm above the box top: not standing on it
    assert not feet_on_target("ground_to_box", high, box_pos, IDENTITY, r).any()
    plat = torch.tensor([[[2.3, 0.0, 0.5 + r], [2.3, 0.2, 0.5 + r], [2.6, 0.0, 0.5 + r], [1.9, 0.0, 0.5 + r]]])
    assert feet_on_target("box_to_platform", plat, box_pos, IDENTITY, r).tolist() == [[True, True, True, False]]

    all_on, three_on = torch.ones(1, 4, dtype=torch.bool), torch.tensor([[True, True, True, False]])
    still, moving = torch.zeros(1, 3), torch.tensor([[0.5, 0.0, 0.0]])
    assert success_now(IDENTITY, still, all_on).item()
    assert not success_now(IDENTITY, moving, all_on).item()
    assert not success_now(IDENTITY, still, three_on).item()
    assert not success_now(_pitch_quat(math.radians(30)), still, all_on).item()
    assert tilted_over(_pitch_quat(math.radians(70))).item() and not tilted_over(_pitch_quat(math.radians(50))).item()
    assert settle_steps() == round(C.JUMP_SUCCESS_SETTLE_S / C.POLICY_DT)


# --- observation ---------------------------------------------------------------------------


def test_observation_layout():
    covered = sorted((s.start, s.stop) for s in OBS_SLICES.values())
    assert covered[0][0] == 0 and covered[-1][1] == OBS_DIM
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))  # no gaps, no overlaps

    track = ReferenceTrack(placeholder_reference("ground_to_box"), jump="ground_to_box")
    ref0 = track.at(torch.zeros(1))
    zeros3, zeros12 = torch.zeros(1, 3), torch.zeros(1, 12)
    box = torch.tensor([C.BOX_TARGET_POS])

    def obs(yaw: float) -> torch.Tensor:
        q = M.quat_mul(_yaw_quat(yaw), ref0.base_quat)
        target_xy, target_yaw = target_anchor("ground_to_box", box, IDENTITY)
        anchor = Anchor(ref0.base_pos[:, :2], M.yaw_of(q), target_xy, target_yaw)
        return jump_observation(
            track=track, time=torch.zeros(1), anchor=anchor,
            base_pos=ref0.base_pos, base_quat=q, base_lin_vel=zeros3, base_ang_vel=zeros3,
            joint_pos=ref0.joint_pos, joint_vel=zeros12, last_action=zeros12, box_pos=box, box_quat=IDENTITY,
        )[0]  # fmt: skip

    o = obs(0.0)
    assert o.shape == (OBS_DIM,)
    g = OBS_SLICES
    assert torch.allclose(o[g["gravity"]], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert o[g["phase"]].item() == 0.0
    assert torch.allclose(o[g["box_pos"]], box[0] - ref0.base_pos[0], atol=1e-6)
    assert torch.allclose(o[g["ref_orientation"]], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)
    q0 = torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=torch.float32)
    assert torch.allclose(o[g["ref_joint_pos"]], track.at(torch.tensor([C.POLICY_DT])).joint_pos[0] - q0, atol=1e-6)
    assert torch.allclose(o[g["box_heading"]], torch.tensor([1.0, 0.0]), atol=1e-6)
    # turned 90 deg to the left: the box straight ahead in x now appears on the robot's right (-y)
    o90 = obs(math.pi / 2)
    assert o90[g["box_pos"]][1].item() == pytest.approx(-(box[0, 0] - ref0.base_pos[0, 0]).item(), abs=1e-5)
    assert torch.allclose(o90[g["box_heading"]], torch.tensor([0.0, -1.0]), atol=1e-5)


# --- skill wrapper -------------------------------------------------------------------------


def test_skill(tmp_path):
    ref_path = tmp_path / "reference.npz"
    placeholder_reference("ground_to_box").save(str(ref_path))
    net = torch.nn.Linear(OBS_DIM, C.NUM_JOINTS)
    torch.nn.init.zeros_(net.weight)
    torch.nn.init.zeros_(net.bias)
    policy_path = tmp_path / "policy.pt"
    torch.jit.script(net).save(str(policy_path))

    skill = JumpTrackingSkill("ground_to_box", policy_path=policy_path, reference_path=ref_path)
    assert skill.name == "jump_ground_box"
    track = skill.policy.track
    start, end = track.at(torch.zeros(1)), track.at(torch.tensor([track.duration]))

    def state(sample, time: float, quat=None) -> TaskState:
        return TaskState(
            base_pos=sample.base_pos[0].numpy().astype(float),
            base_quat=(sample.base_quat if quat is None else quat)[0].numpy().astype(float),
            joint_pos=sample.joint_pos[0].numpy().astype(float),
            box_pos=np.array(C.BOX_TARGET_POS, dtype=float),
            time=time,
        )

    skill.reset(state(start, 0.0))
    action = skill.act(state(start, 0.0))
    assert action.shape == (C.NUM_JOINTS,) and np.allclose(action, 0.0)
    assert skill.status(state(start, 0.0)) == SkillStatus.RUNNING
    assert skill.status(state(start, 0.1, quat=_pitch_quat(math.radians(70)))) == SkillStatus.FAILURE

    # landed on the box and still: SUCCESS exactly when the conditions have held long enough
    skill.reset(state(start, 0.0))
    results = [skill.status(state(end, track.duration + k * C.POLICY_DT)) for k in range(settle_steps())]
    assert results[-1] == SkillStatus.SUCCESS and all(r == SkillStatus.RUNNING for r in results[:-1])

    # not landed when the time limit is reached: FAILURE
    skill.reset(state(start, 0.0))
    assert skill.status(state(start, C.JUMP_TIME_LIMIT_S)) == SkillStatus.FAILURE
