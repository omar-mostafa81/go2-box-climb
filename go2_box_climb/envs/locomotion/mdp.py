"""Custom pieces of the walking task, stage B.

Stage B makes the walking policy ready to push the box, without boxes in training:

1. Body-height command h. The policy walks tall (normal walking) or low. When it
   pushes, the push skill commands a LOW height, so the head meets the box face
   instead of going over the box top. The policy does not see the box: the height
   is chosen by whoever commands it.
2. A steady push on the head (the box pushing back), ramped up during training.
3. Standing still at any height: legs held in the standing pose for the commanded
   height, feet planted (needed before the jumps).
"""

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.skills.locomotion.walking_policy import height_observation

from .standing_pose import standing_joint_pos

# Point where the steady push acts: the lower head (the part that touches the box),
# in the body frame [m], read from the Go2 USD (rule 13).
HEAD_OFFSET = go2_geometry().head_lower_pos


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# =============================================================================
# Events
# =============================================================================


def steady_push(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    max_force: tuple[float, float],
    zero_prob: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Draw a new steady force (body frame, applied at the head) for the given envs.

    fx ~ U(-max_force[0], max_force[0]), fy ~ U(-max_force[1], max_force[1]); a
    fraction ``zero_prob`` gets no force. The force stays on until it is re-drawn.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    n = len(env_ids)
    forces = torch.zeros(n, 1, 3, device=asset.device)
    forces[:, 0, 0] = (2 * torch.rand(n, device=asset.device) - 1) * max_force[0]
    forces[:, 0, 1] = (2 * torch.rand(n, device=asset.device) - 1) * max_force[1]
    forces[torch.rand(n, device=asset.device) < zero_prob] = 0.0
    positions = torch.tensor(HEAD_OFFSET, device=asset.device).repeat(n, 1, 1)
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torch.zeros_like(forces),
        positions=positions,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
    )


def push_force_ramp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    event_name: str,
    final_max_force: tuple[float, float],
    ramp_iterations: int,
    steps_per_iteration: int,
) -> float:
    """Curriculum: grow the steady-push limits linearly from 0 to ``final_max_force``.

    Counted from the start of THIS training run. Returns the current forward/back
    limit [N] (TensorBoard: ``Curriculum/push_force``).
    """
    progress = min(1.0, env.common_step_counter / (ramp_iterations * steps_per_iteration))
    term = env.event_manager.get_term_cfg(event_name)
    term.params["max_force"] = (progress * final_max_force[0], progress * final_max_force[1])
    env.event_manager.set_term_cfg(event_name, term)
    return term.params["max_force"][0]


# =============================================================================
# Commands
# =============================================================================


class BodyHeightCommand(CommandTerm):
    """Commanded base height above the ground [m] (one number per env)."""

    cfg: "BodyHeightCommandCfg"

    def __init__(self, cfg: "BodyHeightCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.height = torch.full((self.num_envs, 1), cfg.default, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        h = torch.empty(n, device=self.device).uniform_(*self.cfg.range)
        h = torch.where(torch.rand(n, device=self.device) < self.cfg.default_prob, torch.full_like(h, self.cfg.default), h)
        self.height[ids, 0] = h

    def _update_command(self):
        pass

    def _update_metrics(self):
        z = _t(self.robot.data.root_pos_w)[:, 2]
        self.metrics["error_height"] += torch.abs(z - self.height[:, 0]) * self._env.step_dt / self.cfg.resampling_time_range[1]


@configclass
class BodyHeightCommandCfg(CommandTermCfg):
    class_type: type = BodyHeightCommand
    asset_name: str = "robot"
    range: tuple[float, float] = MISSING  # commanded heights are drawn uniformly from this range [m]
    default: float = MISSING  # normal walking height [m]
    default_prob: float = 0.3  # fraction of draws that use exactly the default height


def height_command_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Observation #49: the body-height command, scaled like the deployment wrapper does."""
    return height_observation(env.command_manager.get_command(command_name))


# =============================================================================
# Rewards
# =============================================================================


def height_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-(z - h)^2 / std^2): base height close to the commanded height."""
    z = _t(env.scene["robot"].data.root_pos_w)[:, 2]
    h = env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-torch.square(z - h) / std**2)


def stand_still_pose(env: ManagerBasedRLEnv, command_name: str, height_command_name: str) -> torch.Tensor:
    """Sum of |joint angle - standing pose for the commanded height| [rad], while told to stand still.

    The reference pose depends on the commanded height (standing_pose.py), so the
    penalty pulls the legs into a still stance at EVERY height without fighting the
    height command. (At the default height it equals the default joint pose.)
    """
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(cmd[:, :2], dim=1) < 0.06) & (torch.abs(cmd[:, 2]) < 0.06)
    q_ref = standing_joint_pos(env.command_manager.get_command(height_command_name)[:, 0])
    return torch.sum(torch.abs(_t(env.scene["robot"].data.joint_pos) - q_ref), dim=1) * standing


def feet_off_ground_when_standing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0, cmd_threshold: float = 0.06
) -> torch.Tensor:
    """Number of feet NOT touching the ground while told to stand still (all four should stay planted)."""
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(cmd[:, :2], dim=1) < cmd_threshold) & (torch.abs(cmd[:, 2]) < cmd_threshold)
    f = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids, :]
    off = (torch.linalg.norm(f, dim=-1) < threshold).float().sum(dim=1)
    return off * standing
