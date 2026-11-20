"""Walking task: the Go2 follows velocity commands (vx, vy, wz) on flat ground.

Built on Isaac Lab's tested Go2 flat-ground recipe (``UnitreeGo2FlatEnvCfg``),
with our shared robot contract and the changes our project needs:

    * yaw rate wz is commanded directly (MPPI sends wz; Isaac Lab steers to a heading)
    * command ranges for pushing and standing still before jumps
    * random foot friction
    * the episode also ends when the head hits the ground
    * gait-shaping rewards for a clean trot (feet lift, diagonal pairs in sync,
      legs under the body); without them the policy shuffles
    * stage B only (ready to push the box):
        - a body-height command h (observation #49): tall for walking, low (~0.27 m)
          for pushing, so the head meets the box face instead of going over the box
        - a steady push on the head that ramps up to +-40 N (the box pushing back)
        - more practice standing still, at any height, with the feet planted

Two configs:
    WalkEnvCfg       stage A: normal walking (48 observations)
    WalkPushEnvCfg   stage B: + height command, steady push, standing still (49 observations;
                     continues from stage A's network). No boxes in training.
"""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.spot.mdp import rewards as gait_rewards
from isaaclab_tasks.core.velocity import mdp as vel_mdp
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from go2_box_climb.assets import GO2_CFG
from go2_box_climb.common import constants as C
from go2_box_climb.skills.locomotion.walking_policy import COMMAND_LIMITS, DEFAULT_BODY_HEIGHT

from . import mdp
from .agents import STEPS_PER_ITERATION

# ---- Gait-shaping rewards (weights; the speed-tracking reward has weight TRACK_LIN_VEL_WEIGHT) ----
GAIT_WEIGHTS = {
    "gait_trot": 1.0,  # diagonal feet pairs move together, opposite to the other pair
    "air_time": 0.5,  # steps of ~AIR_TIME_S in the air and on the ground (all feet planted when standing)
    "foot_clearance": 0.7,  # swinging feet reach FOOT_CLEARANCE_TARGET
    "foot_slip": -0.25,  # feet sliding while on the ground
    "hip_pose": -0.3,  # hips straying from their default angle (legs splayed)
    "stand_still": -0.5,  # when told to stand: joints away from the default standing pose
    "action_rate": -0.02,  # quick changes of the action (smoothness); Isaac Lab default -0.01
}
TRACK_LIN_VEL_WEIGHT = 2.0  # Isaac Lab Go2 default 1.5; raised so the extra gait terms do not slow the robot
TRACK_ANG_VEL_WEIGHT = 1.0  # Isaac Lab Go2 default 0.75
TRACK_STD = 0.25  # width of the exp(-error^2 / std^2) tracking rewards [m/s, rad/s]; Isaac Lab default 0.5 (stricter = less lag)
AIR_TIME_S = 0.3  # target swing (and stance) duration per step [s]
FOOT_CLEARANCE_TARGET = 0.08  # target foot-center height while swinging [m] (~6 cm clearance; foot center is 0.022 m when standing)
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# ---- Stage B push settings --------------------------------------------------
PUSH_MAX_FORCE = (40.0, 15.0)  # final (forward/back, sideways) limits [N]
PUSH_RAMP_ITERATIONS = 300  # training iterations to grow the limits from 0
PUSH_HOLD_TIME_S = (2.0, 5.0)  # each force stays on for 2-5 s, then is re-drawn
PUSH_ZERO_PROB = 0.25  # a quarter of the draws are "no force"

# ---- Stage B: height command and standing still ----------------------------
HEIGHT_RANGE = COMMAND_LIMITS["h"]  # commanded body height [m] (0.25-0.34; normal 0.32, pushing ~0.27)
B_WEIGHTS = {
    "height": 1.5,  # base height close to the commanded height
    "feet_planted": -1.0,  # per foot off the ground while told to stand still
}
STANDING_PROB = 0.2  # fraction of velocity commands that are "stand still" (stage A: 0.1)
COMMAND_RESAMPLE_S = (5.0, 5.0)  # new commands every 5 s (stage A: 10 s): more "walking, now stop" practice
HEIGHT_STD = 0.04  # width of the height-tracking reward [m]


@configclass
class WalkEnvCfg(UnitreeGo2FlatEnvCfg):
    """Stage A: normal walking on flat ground."""

    def __post_init__(self):
        super().__post_init__()

        # --- Robot contract (single source of truth: go2_box_climb.common) ---
        self.scene.robot = GO2_CFG.replace()  # a copy, so the shared config is never modified
        self.sim.dt = C.SIM_DT
        self.decimation = C.DECIMATION
        self.sim.render_interval = C.DECIMATION
        self.actions.joint_pos.scale = C.ACTION_SCALE
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.joint_names = list(C.JOINT_NAMES)
        self.scene.terrain.physics_material.static_friction = C.GROUND_FRICTION
        self.scene.terrain.physics_material.dynamic_friction = C.GROUND_FRICTION

        # --- Commands: direct yaw rate, ranges for pushing and standing ---
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1  # 10% of robots are told to stand still
        cmd.ranges.lin_vel_x = COMMAND_LIMITS["vx"]  # shared with the deployment wrapper
        cmd.ranges.lin_vel_y = COMMAND_LIMITS["vy"]
        cmd.ranges.ang_vel_z = COMMAND_LIMITS["wz"]
        cmd.ranges.heading = None

        # --- Randomization: foot friction 0.4-1.25 (ground friction is 1.0, "multiply") ---
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.25)
        self.events.physics_material.params["make_consistent"] = True  # dynamic <= static

        # --- Gait-shaping rewards (terms from Isaac Lab's Spot task) ---
        w = GAIT_WEIGHTS
        feet_sensor = SceneEntityCfg("contact_forces", body_names=list(FEET))
        feet_body = SceneEntityCfg("robot", body_names=list(FEET))
        self.rewards.feet_air_time = None  # replaced by air_time below (its 0.5 s target is too long for a Go2)
        self.rewards.air_time = RewTerm(
            func=gait_rewards.air_time_reward,
            weight=w["air_time"],
            params={
                "mode_time": AIR_TIME_S,
                "velocity_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": feet_sensor,
            },
        )
        self.rewards.gait_trot = RewTerm(
            func=gait_rewards.GaitReward,
            weight=w["gait_trot"],
            params={
                "std": 0.1,
                "max_err": 0.2,
                "velocity_threshold": 0.5,
                "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=gait_rewards.foot_clearance_reward,
            weight=w["foot_clearance"],
            params={"std": 0.05, "tanh_mult": 2.0, "target_height": FOOT_CLEARANCE_TARGET, "asset_cfg": feet_body},
        )
        self.rewards.foot_slip = RewTerm(
            func=gait_rewards.foot_slip_penalty,
            weight=w["foot_slip"],
            params={"asset_cfg": feet_body, "sensor_cfg": feet_sensor, "threshold": 1.0},
        )
        self.rewards.hip_pose = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=w["hip_pose"],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
        )
        self.rewards.stand_still = RewTerm(
            func=vel_mdp.stand_still_joint_deviation_l1,
            weight=w["stand_still"],
            params={"command_name": "base_velocity"},
        )
        self.rewards.action_rate_l2.weight = w["action_rate"]
        self.rewards.track_lin_vel_xy_exp.weight = TRACK_LIN_VEL_WEIGHT
        self.rewards.track_ang_vel_z_exp.weight = TRACK_ANG_VEL_WEIGHT
        self.rewards.track_lin_vel_xy_exp.params["std"] = TRACK_STD
        self.rewards.track_ang_vel_z_exp.params["std"] = TRACK_STD

        # --- Camera (gui / stream): follow robot 0 from behind-right ---
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.0, -2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # --- Termination: trunk OR head touches the ground = fall ---
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["base", "Head_upper", "Head_lower"]
        )

    def play_mode(self):
        """Fewer robots, no random disturbances (used by play.py)."""
        super().play_mode()
        self.scene.num_envs = 16


@configclass
class WalkPushEnvCfg(WalkEnvCfg):
    """Stage B: stage A + body-height command + a steady push on the head + standing still."""

    def __post_init__(self):
        super().__post_init__()

        # --- Commands: more "stand still" commands; body-height command ---
        self.commands.base_velocity.rel_standing_envs = STANDING_PROB
        self.commands.base_velocity.resampling_time_range = COMMAND_RESAMPLE_S
        self.commands.body_height = mdp.BodyHeightCommandCfg(
            resampling_time_range=self.commands.base_velocity.resampling_time_range,
            range=HEIGHT_RANGE,
            default=DEFAULT_BODY_HEIGHT,
        )

        # --- Observation #49 (appended LAST, so stage A's 48 inputs keep their places) ---
        self.observations.policy.height_command = ObsTerm(
            func=mdp.height_command_obs, params={"command_name": "body_height"}
        )

        # --- Steady push on the head, ramping up from 0 to +-40 N ---
        self.events.steady_push = EventTerm(
            func=mdp.steady_push,
            mode="interval",
            interval_range_s=PUSH_HOLD_TIME_S,
            params={
                "max_force": (0.0, 0.0),  # set by the curriculum below
                "zero_prob": PUSH_ZERO_PROB,
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        self.curriculum.push_force = CurrTerm(
            func=mdp.push_force_ramp,
            params={
                "event_name": "steady_push",
                "final_max_force": PUSH_MAX_FORCE,
                "ramp_iterations": PUSH_RAMP_ITERATIONS,
                "steps_per_iteration": STEPS_PER_ITERATION,
            },
        )

        # --- Rewards ---
        w = B_WEIGHTS
        self.rewards.height = RewTerm(
            func=mdp.height_tracking_exp,
            weight=w["height"],
            params={"command_name": "body_height", "std": HEIGHT_STD},
        )
        self.rewards.stand_still = RewTerm(  # replaces stage A's version: standing pose for the commanded height
            func=mdp.stand_still_pose,
            weight=GAIT_WEIGHTS["stand_still"],
            params={"command_name": "base_velocity", "height_command_name": "body_height"},
        )
        self.rewards.feet_planted = RewTerm(
            func=mdp.feet_off_ground_when_standing,
            weight=w["feet_planted"],
            params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FEET))},
        )

    def play_mode(self):
        super().play_mode()
        self.events.steady_push = None
        self.curriculum.push_force = None
