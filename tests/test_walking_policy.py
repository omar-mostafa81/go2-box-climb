"""Fast tests of the walking-policy wrapper (no simulator, dummy network)."""

import numpy as np
import pytest
import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.state import TaskState
from go2_box_climb.envs.locomotion.checkpoint import widen_checkpoint_inputs
from go2_box_climb.skills.locomotion import DEFAULT_BODY_HEIGHT, OBS_DIM, WalkingPolicy, WalkSkill, has_fallen
from go2_box_climb.skills.locomotion.walking_policy import quat_rotate_inverse


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def test_quat_rotate_inverse():
    # robot turned +90 deg (facing world +y): world +x is on its right, i.e. body -y
    assert np.allclose(quat_rotate_inverse(_yaw_quat(np.pi / 2), np.array([1.0, 0, 0])), [0, -1, 0], atol=1e-9)
    # level robot: gravity points straight down in the body frame
    assert np.allclose(quat_rotate_inverse(_yaw_quat(0.3), np.array([0, 0, -1.0])), [0, 0, -1])


@pytest.fixture
def dummy_policy_path(tmp_path):
    torch.manual_seed(0)
    net = torch.jit.script(torch.nn.Linear(OBS_DIM, C.NUM_JOINTS))
    path = tmp_path / "policy.pt"
    net.save(str(path))
    return path


def test_observation_layout(dummy_policy_path):
    p = WalkingPolicy(dummy_policy_path)
    s = TaskState(joint_pos=C.DEFAULT_JOINT_POS.copy())
    p.set_command(0.5, 0.0, 0.2, 0.27)
    obs = p.observation(s)
    assert obs.shape == (OBS_DIM,)
    assert np.allclose(obs[6:9], [0, 0, -1])  # gravity
    assert np.allclose(obs[9:12], [0.5, 0.0, 0.2])  # command
    assert np.allclose(obs[12:24], 0)  # joints at the default pose
    assert obs[48] == pytest.approx((0.27 - 0.32) * 20)  # body-height command (scaled) is the last input


def test_stage_a_policy_has_no_height_input(tmp_path):
    net = torch.jit.script(torch.nn.Linear(48, C.NUM_JOINTS))
    net.save(str(tmp_path / "a.pt"))
    p = WalkingPolicy(tmp_path / "a.pt")
    assert not p.has_height_input
    p.set_command(0.3, 0.0, 0.0, 0.27)
    assert p.observation(TaskState()).shape == (48,)
    assert p.act(TaskState()).shape == (12,)


def test_last_action_feeds_back(dummy_policy_path):
    p = WalkingPolicy(dummy_policy_path)
    s = TaskState(joint_pos=C.DEFAULT_JOINT_POS.copy())
    a = p.act(s)
    assert a.shape == (12,)
    assert np.allclose(p.observation(s)[36:48], a, atol=1e-6)
    p.reset()
    assert np.allclose(p.observation(s)[36:48], 0)


def test_command_is_clipped(dummy_policy_path):
    p = WalkingPolicy(dummy_policy_path)
    assert np.allclose(p.set_command(3.0, -2.0, 0.5), [1.0, -0.6, 0.5, DEFAULT_BODY_HEIGHT])
    assert p.set_command(0, 0, 0, 0.1)[3] == pytest.approx(0.25)  # height clipped to the trained range


def test_fall_check(dummy_policy_path):
    skill = WalkSkill(dummy_policy_path)
    standing = TaskState(base_pos=np.array([0, 0, 0.32]))
    assert not has_fallen(standing)
    assert has_fallen(TaskState(base_pos=np.array([0, 0, 0.10])))
    on_side = TaskState(base_pos=np.array([0, 0, 0.3]), base_quat=np.array([np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)]))
    assert has_fallen(on_side)
    skill.reset(standing)
    assert skill.status(standing).value == "running"


def test_training_checkpoint_is_exported(tmp_path):
    # a fake rsl_rl checkpoint: actor MLP 48 -> 16 -> 12 with ELU (+ an unused std parameter)
    torch.manual_seed(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 16), torch.nn.ELU(), torch.nn.Linear(16, C.NUM_JOINTS))
    actor = {f"mlp.{k}": v for k, v in mlp.state_dict().items()}
    actor["distribution.std_param"] = torch.ones(C.NUM_JOINTS)
    ckpt = tmp_path / "model_7.pt"
    torch.save({"actor_state_dict": actor, "critic_state_dict": {}, "iter": 7}, ckpt)

    p = WalkingPolicy(ckpt)
    assert p.policy_file == tmp_path / "exported" / "model_7_policy.pt"
    x = torch.randn(5, OBS_DIM)
    assert torch.allclose(p.net(x), mlp(x), atol=1e-6)


def test_widen_checkpoint_keeps_outputs(tmp_path):
    # stage A checkpoint with 48 inputs -> widened to 49: same actions when the new input is anything
    torch.manual_seed(2)
    mlp = torch.nn.Sequential(torch.nn.Linear(48, 16), torch.nn.ELU(), torch.nn.Linear(16, C.NUM_JOINTS))
    actor = {f"mlp.{k}": v for k, v in mlp.state_dict().items()}
    critic = {"mlp.0.weight": torch.randn(16, 48), "mlp.0.bias": torch.zeros(16)}
    src, dst = tmp_path / "model_999.pt", tmp_path / "widened.pt"
    torch.save({"actor_state_dict": actor, "critic_state_dict": critic, "optimizer_state_dict": {}, "iter": 999}, src)
    assert widen_checkpoint_inputs(src, 49, dst)
    w = torch.load(dst, weights_only=False)
    assert w["actor_state_dict"]["mlp.0.weight"].shape == (16, 49)
    assert w["critic_state_dict"]["mlp.0.weight"].shape == (16, 49)
    assert "optimizer_state_dict" not in w
    p = WalkingPolicy(dst)  # exported from the widened checkpoint
    x = torch.randn(4, 48)
    assert torch.allclose(p.net(torch.cat([x, torch.full((4, 1), 0.27)], 1)), mlp(x), atol=1e-6)
    assert not widen_checkpoint_inputs(dst, 49, tmp_path / "again.pt")  # already 49: nothing to do


def test_standing_pose_for_height():
    from go2_box_climb.envs.locomotion.standing_pose import _foot, standing_joint_pos

    h = torch.tensor([0.32, 0.27, 0.25, 0.34], dtype=torch.float64)
    q = standing_joint_pos(h)
    # default height -> exactly the default pose
    assert torch.allclose(q[0], torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=torch.float64), atol=1e-9)
    # every leg: hip-to-foot drop changes by exactly (h - 0.32), fore/aft foot offset unchanged
    x0, z0 = _foot(q[0, 4:8], q[0, 8:12])
    for i in range(1, 4):
        x, z = _foot(q[i, 4:8], q[i, 8:12])
        assert torch.allclose(x, x0, atol=1e-9)
        assert torch.allclose(-z, -z0 + (h[i] - 0.32), atol=1e-9)
    assert torch.all(q[1, 8:12] < q[0, 8:12])  # lower body -> knees bent more (calf angle more negative)
    assert torch.allclose(q[:, 0:4], q[0, 0:4].expand(4, 4))  # hip joints unchanged
