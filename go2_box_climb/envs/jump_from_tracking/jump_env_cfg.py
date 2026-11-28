"""Jump-tracking task: the Go2 jumps ground -> box or box -> platform, guided by a reference.

One environment config for both jumps; ``JumpTrackEnvCfg().configure(jump, reference_path)``
picks the jump and the reference file. Built on the shared basic scene (robot, box,
platform, contact sensor; envs/scene/box_climb_scene.py) and the shared robot contract
(common/constants.py). Every episode is ONE jump attempt from new random start
conditions (see mdp.JumpReferenceCommand).

    observation   77 numbers (skills/jump/jump_from_tracking/observation.py)
    action        12 joint targets: DEFAULT_JOINT_POS + ACTION_SCALE * action (PD, shared contract)
    rewards       follow the reference (base height / position / orientation, joints, foot
                  contacts) + feet on the target after the planned landing + success bonus
                  - fall penalty - effort and smoothness penalties
    episode end   success (held JUMP_SUCCESS_SETTLE_S), fall (trunk/head contact or tilt
                  > JUMP_MAX_TILT), or JUMP_TIME_LIMIT_S

Training aids (switched off by ``eval_mode()``, which play.py and eval.py use, so the
policy is judged with the shared jump rules only):
    * early stop when the base is more than FAR_FROM_REFERENCE_M from the reference
    * RSI_PROB of the attempts start part-way through the jump (reference state initialization)
    * curriculum: the start/box ranges grow from nominal to the full shared ranges between
      RANDOMIZATION_RAMP_ITERATIONS
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from go2_box_climb.common import constants as C
from go2_box_climb.envs.scene.box_climb_scene import BoxClimbSceneCfg
from go2_box_climb.skills.jump.jump_from_tracking.reference import check_jump

from . import mdp
from .agents import STEPS_PER_ITERATION

COMMAND = "jump_reference"  # name of the command term that holds the reference

# ---- Reward weights (starting values; tuned on the real references) -----------------
# Isaac Lab multiplies every term by the policy step (0.02 s). Tracking terms are in [0, 1]
# per step, so a weight w gives at most w per second of good tracking; the success bonus and
# fall penalty happen once, so their weights are large (250 * 0.02 = 5 once).
# The body's path is weighted above the joint angles: copying the leg motion alone (without
# leaving the ground) must not pay off.
WEIGHTS = {
    "track_base_height": 2.0,
    "track_base_xy": 2.0,
    "track_base_orientation": 1.0,
    "track_joint_pos": 1.0,
    "track_foot_contact": 0.5,
    "feet_on_target": 1.0,  # per step, from the planned landing on
    "success": 250.0,
    "fall": -250.0,
    "action_rate": -0.01,
    "joint_torques": -2.0e-4,
    "joint_acc": -2.5e-7,
    "joint_pos_limits": -5.0,
}
# Widths of the exp(-error^2 / std^2) tracking rewards: wide enough that being closer to the
# planned path is still rewarded when the robot is far from it
TRACK_STD = {
    "base_height": 0.15,  # [m]
    "base_xy": 0.3,  # [m]
    "base_orientation": 0.4,  # [rad]
    "joint_pos": 0.6,  # [rad], over the sum of the 12 squared joint errors
}
FOOT_CONTACT_FORCE = 1.0  # [N] a foot counts as touching above this force
FEET_SENSOR = SceneEntityCfg("contact_forces", body_names=list(mdp.FEET), preserve_order=True)
FEET_BODIES = SceneEntityCfg("robot", body_names=list(mdp.FEET), preserve_order=True)

# ---- Training aids (see the module docstring) ------------------------------------------
FAR_FROM_REFERENCE_M = 0.3  # early stop when the base is farther than this from the reference [m]
RSI_PROB = 0.5  # share of attempts that start part-way through the jump
RANDOMIZATION_RAMP_ITERATIONS = (200, 1000)  # start/box ranges: nominal before, full after
BASE_MASS_RANGE = (1 / 1.1, 1.1)  # base mass scale (TO's MuJoCo check: the plan is sensitive to mass)


@configclass
class CommandsCfg:
    jump_reference = mdp.JumpReferenceCommandCfg(jump="ground_to_box")


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(C.JOINT_NAMES),
        scale=C.ACTION_SCALE,
        use_default_offset=True,  # targets = default pose + scale * action (default pose = DEFAULT_JOINT_POS)
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        jump = ObsTerm(func=mdp.jump_obs, params={"command_name": COMMAND})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # startup: foot/body friction 0.4-1.25 (as for walking) and base mass (BASE_MASS_RANGE)
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.25),
            "dynamic_friction_range": (0.4, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": BASE_MASS_RANGE,
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # reset: done by the command term (mdp.JumpReferenceCommand places robot and box)


@configclass
class RewardsCfg:
    track_joint_pos = RewTerm(
        func=mdp.track_joint_pos_exp,
        weight=WEIGHTS["track_joint_pos"],
        params={"command_name": COMMAND, "std": TRACK_STD["joint_pos"]},
    )
    track_base_height = RewTerm(
        func=mdp.track_base_height_exp,
        weight=WEIGHTS["track_base_height"],
        params={"command_name": COMMAND, "std": TRACK_STD["base_height"]},
    )
    track_base_xy = RewTerm(
        func=mdp.track_base_xy_exp,
        weight=WEIGHTS["track_base_xy"],
        params={"command_name": COMMAND, "std": TRACK_STD["base_xy"]},
    )
    track_base_orientation = RewTerm(
        func=mdp.track_base_orientation_exp,
        weight=WEIGHTS["track_base_orientation"],
        params={"command_name": COMMAND, "std": TRACK_STD["base_orientation"]},
    )
    track_foot_contact = RewTerm(
        func=mdp.track_foot_contact,
        weight=WEIGHTS["track_foot_contact"],
        params={"command_name": COMMAND, "sensor_cfg": FEET_SENSOR, "threshold": FOOT_CONTACT_FORCE},
    )
    feet_on_target = RewTerm(
        func=mdp.feet_on_target_after_landing,
        weight=WEIGHTS["feet_on_target"],
        params={"command_name": COMMAND, "asset_cfg": FEET_BODIES},
    )
    success = RewTerm(func=base_mdp.is_terminated_term, weight=WEIGHTS["success"], params={"term_keys": "success"})
    fall = RewTerm(
        func=base_mdp.is_terminated_term, weight=WEIGHTS["fall"], params={"term_keys": ["fall_contact", "fall_tilt"]}
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=WEIGHTS["action_rate"])
    joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=WEIGHTS["joint_torques"])
    joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=WEIGHTS["joint_acc"])
    joint_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=WEIGHTS["joint_pos_limits"])


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)  # JUMP_TIME_LIMIT_S (episode_length_s)
    fall_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(C.JUMP_FAILURE_CONTACT_BODIES)),
            "threshold": 1.0,
        },
    )
    fall_tilt = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": float(C.JUMP_MAX_TILT)})
    success = DoneTerm(func=mdp.JumpSuccess, params={"jump": "ground_to_box"})
    # training only (removed by eval_mode)
    far_from_reference = DoneTerm(
        func=mdp.far_from_reference, params={"command_name": COMMAND, "max_distance": FAR_FROM_REFERENCE_M}
    )


@configclass
class JumpTrackEnvCfg(ManagerBasedRLEnvCfg):
    """Jump-tracking environment. Call ``configure(jump, reference_path)`` before use."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    scene: BoxClimbSceneCfg = BoxClimbSceneCfg(num_envs=4096, env_spacing=C.ENV_SPACING)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # --- Robot contract (single source of truth: go2_box_climb.common) ---
        self.decimation = C.DECIMATION
        self.sim.dt = C.SIM_DT
        self.sim.render_interval = C.DECIMATION
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = C.SIM_DT
        # --- One jump attempt per episode ---
        self.episode_length_s = C.JUMP_TIME_LIMIT_S
        # --- Training aids (eval_mode switches them off) ---
        self.commands.jump_reference.rsi_prob = RSI_PROB
        self.commands.jump_reference.randomization_ramp_steps = tuple(
            i * STEPS_PER_ITERATION for i in RANDOMIZATION_RAMP_ITERATIONS
        )
        # --- Camera (gui / stream / --video): side view of robot 0 ---
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.4, -2.6, 1.0)
        self.viewer.lookat = (0.4, 0.0, 0.3)

    def configure(self, jump: str, reference_path: str | None = None) -> "JumpTrackEnvCfg":
        """Pick the jump and its reference file (None = the final one in references/)."""
        check_jump(jump)
        self.commands.jump_reference.jump = jump
        self.commands.jump_reference.reference_path = str(reference_path) if reference_path else None
        self.terminations.success.params["jump"] = jump
        return self

    def eval_mode(self) -> "JumpTrackEnvCfg":
        """Judge with the shared jump rules only (eval.py, play.py): no early stop on drifting from the
        reference, every attempt from the start, full start/box ranges."""
        self.terminations.far_from_reference = None
        self.commands.jump_reference.rsi_prob = 0.0
        self.commands.jump_reference.randomization_ramp_steps = None
        return self

    def play_mode(self):
        """Fewer robots for watching (used by play.py), under evaluation conditions."""
        super().play_mode()
        self.eval_mode()
        self.scene.num_envs = 16
