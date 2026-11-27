"""Custom pieces of the jump-tracking environment (Isaac-dependent).

    JumpReferenceCommand   holds the reference, places robot and box at every reset
                           (random start conditions from common/constants.py), and
                           remembers each robot's start pose (the reference's anchor)
    jump_obs               the policy observation (built by the Isaac-free observation.py)
    track_* rewards        closeness to the reference at the current time
    JumpSuccess            success termination (conditions held for JUMP_SUCCESS_SETTLE_S)
    open_loop_actions      actions that just replay the reference's joint angles (checks)

Time. The reference time of an env is ``episode_length_buf * step_dt``: 0 right after
a reset, and the time of the current state when rewards and terminations are computed
(Isaac Lab counts the step before computing them). Training therefore starts every
episode at reference time 0 (``init_at_random_ep_len=False`` in train.py).
"""

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from go2_box_climb.common import constants as C
from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.skills.jump.jump_from_tracking.checks import (
    FEET_ON_TARGET_DEFAULT,
    feet_on_target,
    settle_steps,
    success_now,
)
from go2_box_climb.skills.jump.jump_from_tracking.jump_math import quat_angle
from go2_box_climb.skills.jump.jump_from_tracking.observation import jump_observation
from go2_box_climb.skills.jump.jump_from_tracking.reference import ReferenceSample, load_reference
from go2_box_climb.skills.jump.jump_from_tracking.start import sample_start

FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")  # same order as common/formats.py FEET


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# =============================================================================
# Command: the reference, the start conditions and the anchor
# =============================================================================


class JumpReferenceCommand(CommandTerm):
    """The jump reference of every env, anchored to that env's start pose.

    ``command`` is the phase (time / reference duration), one number per env.
    Each reset (= each new jump attempt) draws new start conditions for the box and the
    robot (skills/jump/jump_from_tracking/start.py, ranges from common/constants.py),
    puts them into the simulation, and anchors the reference to the robot's start.
    """

    cfg: "JumpReferenceCommandCfg"

    def __init__(self, cfg: "JumpReferenceCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.box = env.scene[cfg.box_name]
        self.track = load_reference(cfg.jump, cfg.reference_path, device=self.device)
        n = self.num_envs
        self.start_xy = torch.zeros(n, 2, device=self.device)
        self.start_yaw = torch.zeros(n, device=self.device)
        # start conditions of the current attempt (env frame), kept for the evaluation logs
        self.box_gap = torch.zeros(n, device=self.device)
        self.box_offset_y = torch.zeros(n, device=self.device)
        self.box_yaw = torch.zeros(n, device=self.device)
        self.start_distance = torch.zeros(n, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(n, device=self.device)
        self.metrics["error_base_pos"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            f"JumpReferenceCommand: jump {self.cfg.jump}, reference {self.cfg.reference_path or 'references/'}, "
            f"{self.track.num_samples} samples, {self.track.duration:.2f} s"
        )

    # --- reference time and targets ---------------------------------------------

    def time(self) -> torch.Tensor:
        """(N,) reference time of each env [s] (see the module docstring)."""
        return self._env.episode_length_buf.float() * self._env.step_dt

    def targets(self, lookahead: float = 0.0) -> ReferenceSample:
        """Anchored reference values at time() + lookahead (env frame)."""
        return self.track.at_anchored(self.time() + lookahead, self.start_xy, self.start_yaw)

    @property
    def command(self) -> torch.Tensor:
        return (self.time() / self.track.duration).clamp(0.0, 1.0).unsqueeze(-1)

    # --- reset: start conditions --------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        """Called at every reset: draw new start conditions (start.py) and put robot and box there."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        if n == 0:
            return
        s = sample_start(self.cfg.jump, self.track, n, self.device)
        origins = self._env.scene.env_origins[ids]
        zeros6 = torch.zeros(n, 6, device=self.device)
        self.box.write_root_pose_to_sim_index(root_pose=torch.cat((s.box_pos + origins, s.box_quat), dim=-1), env_ids=ids)
        self.box.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)
        self.robot.write_root_pose_to_sim_index(
            root_pose=torch.cat((s.robot_pos + origins, s.robot_quat), dim=-1), env_ids=ids
        )
        self.robot.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=s.joint_pos, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(s.joint_pos), env_ids=ids)
        # hold the start pose until the first action arrives
        self.robot.set_joint_position_target_index(target=s.joint_pos, env_ids=ids)

        # the reference is anchored to this start
        self.start_xy[ids] = s.robot_pos[:, :2]
        self.start_yaw[ids] = s.robot_yaw
        self.box_gap[ids], self.box_offset_y[ids], self.box_yaw[ids] = s.box_gap, s.box_offset_y, s.box_yaw
        self.start_distance[ids] = s.start_distance

    def _update_command(self):
        pass

    def _update_metrics(self):
        """Add up the tracking errors of this step (turned into per-episode means in reset)."""
        ref = self.targets()
        q = _t(self.robot.data.joint_pos)
        p = _t(self.robot.data.root_pos_w) - self._env.scene.env_origins
        self.metrics["error_joint_pos"] += torch.linalg.norm(q - ref.joint_pos, dim=-1)
        self.metrics["error_base_pos"] += torch.linalg.norm(p - ref.base_pos, dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # The metrics hold sums over the episode's steps: divide by the episode length so the
        # logged values (Metrics/jump_reference/...) are mean errors per step [rad, m], whether the
        # attempt lasted 0.5 s or 3 s. (episode_length_buf is set to 0 only after this reset.)
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        steps = self._env.episode_length_buf[ids].float().clamp(min=1.0)
        for value in self.metrics.values():
            value[ids] = value[ids] / steps
        return super().reset(env_ids)


@configclass
class JumpReferenceCommandCfg(CommandTermCfg):
    class_type: type = JumpReferenceCommand
    asset_name: str = "robot"
    box_name: str = "box"
    jump: str = MISSING  # "ground_to_box" or "box_to_platform"
    reference_path: str | None = None  # None = the final file in references/
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # only at resets (each reset = new attempt)


def reference_term(env: ManagerBasedRLEnv, command_name: str) -> JumpReferenceCommand:
    return env.command_manager.get_term(command_name)


# =============================================================================
# Observation
# =============================================================================


def jump_obs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The full policy observation (layout in skills/jump/jump_from_tracking/observation.py)."""
    cmd = reference_term(env, command_name)
    robot, box = cmd.robot, cmd.box
    origins = env.scene.env_origins
    return jump_observation(
        track=cmd.track,
        time=cmd.time(),
        start_xy=cmd.start_xy,
        start_yaw=cmd.start_yaw,
        base_pos=_t(robot.data.root_pos_w) - origins,
        base_quat=_t(robot.data.root_quat_w),
        base_lin_vel=_t(robot.data.root_lin_vel_w),
        base_ang_vel=_t(robot.data.root_ang_vel_w),
        joint_pos=_t(robot.data.joint_pos),
        joint_vel=_t(robot.data.joint_vel),
        last_action=env.action_manager.action,
        box_pos=_t(box.data.root_pos_w) - origins,
        box_quat=_t(box.data.root_quat_w),
    )


# =============================================================================
# Rewards (tracking: exp(-error^2 / std^2), 1 = perfect)
# =============================================================================


def track_joint_pos_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Joint angles close to the reference (sum of squared errors over the 12 joints)."""
    ref = reference_term(env, command_name).targets()
    err = torch.sum(torch.square(_t(env.scene["robot"].data.joint_pos) - ref.joint_pos), dim=-1)
    return torch.exp(-err / std**2)


def track_base_height_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Base height close to the reference."""
    ref = reference_term(env, command_name).targets()
    z = _t(env.scene["robot"].data.root_pos_w)[:, 2]
    return torch.exp(-torch.square(z - ref.base_pos[:, 2]) / std**2)


def track_base_xy_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Horizontal base position close to the (anchored) reference."""
    ref = reference_term(env, command_name).targets()
    xy = _t(env.scene["robot"].data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
    return torch.exp(-torch.sum(torch.square(xy - ref.base_pos[:, :2]), dim=-1) / std**2)


def track_base_orientation_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Base orientation close to the (anchored) reference (rotation angle between them)."""
    ref = reference_term(env, command_name).targets()
    angle = quat_angle(_t(env.scene["robot"].data.root_quat_w), ref.base_quat)
    return torch.exp(-torch.square(angle) / std**2)


def track_foot_contact(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Fraction of feet whose contact (force > threshold) matches the reference's plan, in [0, 1]."""
    ref = reference_term(env, command_name).targets()
    f = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids, :]
    contact = (torch.linalg.norm(f, dim=-1) > threshold).float()
    return 1.0 - torch.mean(torch.abs(contact - ref.foot_contact), dim=-1)


# =============================================================================
# Success termination
# =============================================================================


class JumpSuccess(ManagerTermBase):
    """True once the success conditions (checks.py) have held for JUMP_SUCCESS_SETTLE_S.

    It counts, per env, how many policy steps in a row the conditions held; the count
    restarts at every reset.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.box = env.scene[cfg.params.get("box_name", "box")]
        self.feet_ids, _ = self.robot.find_bodies(list(FEET), preserve_order=True)
        self.foot_radius = go2_geometry().foot_radius
        self.ok_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.needed = settle_steps()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.ok_steps[:] = 0
        else:
            self.ok_steps[torch.as_tensor(env_ids, device=self.ok_steps.device, dtype=torch.long)] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        jump: str,
        min_feet: int = FEET_ON_TARGET_DEFAULT,
        asset_name: str = "robot",
        box_name: str = "box",
    ) -> torch.Tensor:
        origins = env.scene.env_origins
        feet = _t(self.robot.data.body_pos_w)[:, self.feet_ids, :] - origins.unsqueeze(1)
        box_pos = _t(self.box.data.root_pos_w) - origins
        on = feet_on_target(jump, feet, box_pos, _t(self.box.data.root_quat_w), self.foot_radius)
        ok = success_now(_t(self.robot.data.root_quat_w), _t(self.robot.data.root_lin_vel_w), on, min_feet)
        self.ok_steps = torch.where(ok, self.ok_steps + 1, torch.zeros_like(self.ok_steps))
        return self.ok_steps >= self.needed


# =============================================================================
# Open-loop replay of the reference (no policy): used to check a reference
# =============================================================================


def open_loop_actions(env: ManagerBasedRLEnv, command_name: str = "jump_reference") -> torch.Tensor:
    """Actions whose joint targets are the reference's joint angles at the next step.

    action = (q_ref(t + dt) - DEFAULT_JOINT_POS) / ACTION_SCALE, i.e. the PD motors simply try to
    follow the reference. Shows whether the reference works without any learning (usually not).
    """
    ref = reference_term(env, command_name).targets(lookahead=env.step_dt)
    q0 = torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=ref.joint_pos.dtype, device=ref.joint_pos.device)
    return (ref.joint_pos - q0) / C.ACTION_SCALE
