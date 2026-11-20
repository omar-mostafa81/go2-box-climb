"""Single source of truth for robot settings and scene geometry.

Every part of the project (walking, jumping, MPPI, TO, sequencer) imports these
values instead of writing its own numbers. If two policies used different
settings, they would behave differently when chained in the full task.

Conventions used everywhere in the project:
    * Units: meters, seconds, radians, kilograms, newtons.
    * World frame: z up. In each environment, the robot starts at the origin
      facing +x, the box is ahead of it on +x, and the platform is beyond the box.
    * Quaternions: (x, y, z, w) order. This is Isaac Lab 3.0's convention.
      (Older Isaac Lab versions used (w, x, y, z); convert if you use other tools.)

Only change this file with the team leader's approval.
"""

import numpy as np

# =============================================================================
# Robot contract: Unitree Go2
# (values match Isaac Lab's UNITREE_GO2_CFG and Go2 velocity task)
# =============================================================================

# --- Timing ---
SIM_DT = 0.005  # physics step [s]  (200 Hz)
DECIMATION = 4  # physics steps per policy step
POLICY_DT = SIM_DT * DECIMATION  # policy step [s]  (50 Hz)

# --- Joint order ---
# Isaac Lab orders the Go2 joints BY TYPE (all hips, then all thighs, then all
# calves), not by leg. Every observation/action vector over joints uses this order.
# Legs: FL = front-left, FR = front-right, RL = rear-left, RR = rear-right.
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)  # fmt: skip
NUM_JOINTS = len(JOINT_NAMES)  # 12

# --- Default (standing) joint pose [rad], in JOINT_NAMES order ---
DEFAULT_JOINT_POS = np.array(
    [
        0.1, -0.1, 0.1, -0.1,  # hips:   left +0.1, right -0.1
        0.8, 0.8, 1.0, 1.0,  # thighs: front 0.8, rear 1.0
        -1.5, -1.5, -1.5, -1.5,  # calves
    ]
)  # fmt: skip

# --- Actions ---
# Every policy outputs 12 numbers a (one per joint). The joint target sent to the
# PD controller is:   q_target = DEFAULT_JOINT_POS + ACTION_SCALE * a
ACTION_SCALE = 0.25

# --- Low-level PD controller (runs every physics step) ---
# torque = KP * (q_target - q) - KD * q_dot, clipped by the DC motor model.
KP = 25.0  # stiffness [N*m/rad]
KD = 0.5  # damping   [N*m*s/rad]
EFFORT_LIMIT = 23.5  # max torque [N*m]
VELOCITY_LIMIT = 30.0  # max joint speed [rad/s]

# --- Spawn ---
ROBOT_START_POS = (0.0, 0.0, 0.4)  # base position at reset, in env frame [m]

# =============================================================================
# Scene geometry (env frame)
# =============================================================================

# --- Ground ---
GROUND_FRICTION = 1.0  # static and dynamic, same as Isaac Lab's locomotion tasks

# --- Movable box (the step): a cardboard delivery box ---
# Its root position is always the box CENTER (all scenes keep this).
BOX_SIZE = (0.6, 0.6, 0.25)  # (x length, y width, z height) [m]
BOX_MASS = 5.0  # [kg] (a typical parcel)
BOX_FRICTION = 0.5  # cardboard on concrete; static and dynamic; effective box-ground friction = 0.5 * 1.0
BOX_START_POS = (1.0, 0.0, BOX_SIZE[2] / 2)  # box center at reset [m]

# --- Fixed platform (the goal): a warehouse staging platform ---
# Two equal steps: ground -> box top (0.25 m) -> platform top (0.5 m).
PLATFORM_SIZE = (2.0, 3.0, 0.5)  # (x depth, y width, z height) [m]
PLATFORM_FRONT_X = 2.0  # x of the platform face the box is pushed against [m]
PLATFORM_POS = (PLATFORM_FRONT_X + PLATFORM_SIZE[0] / 2, 0.0, PLATFORM_SIZE[2] / 2)  # center [m]
# The first PLATFORM_CLEAR_DEPTH meters behind the front edge (full width) are
# kept free of obstacles in every scene: this is the landing zone of the last jump.
PLATFORM_CLEAR_DEPTH = 1.0  # [m]

# Box position when it is pushed flush against the platform (the push target).
BOX_TARGET_POS = (PLATFORM_FRONT_X - BOX_SIZE[0] / 2, 0.0, BOX_SIZE[2] / 2)

# Space between environments when many are simulated in parallel [m]
ENV_SPACING = 6.0

# =============================================================================
# Jumps (shared by jump_from_tracking and jump_from_scratch, so both policies
# are trained and judged by the same rules). Starting values: the jump members
# may propose changes, which go through the team leader like any contract change.
# =============================================================================

# --- Failure: ends a training episode or an evaluation attempt ---
JUMP_TIME_LIMIT_S = 3.0  # from the jump start until the robot must have landed and settled [s]
JUMP_MAX_TILT = np.deg2rad(60.0)  # body tilt (roll or pitch) beyond this = fall [rad]
JUMP_FAILURE_CONTACT_BODIES = ("base", "Head_upper", "Head_lower")  # trunk/head touching anything = fall

# --- Success (the jump evaluation in eval/) ---
JUMP_SUCCESS_SETTLE_S = 1.0  # all success conditions must hold this long [s]
JUMP_SUCCESS_MAX_TILT = np.deg2rad(20.0)  # upright: tilt below this [rad]
JUMP_SUCCESS_MAX_SPEED = 0.2  # settled: base speed below this [m/s]
JUMP_FOOT_ON_SURFACE_TOL = 0.03  # a foot counts as on the target surface within this height [m]

# --- Starting conditions (training randomization AND the box-placement study) ---
# Jump 1 (ground -> box): robot base this far behind the box's front face. After the push
# the head touches the box, which puts the base about 0.35 m behind the face.
JUMP1_START_DISTANCE = (0.35, 0.60)  # [m]
JUMP_START_YAW = (-np.deg2rad(10.0), np.deg2rad(10.0))  # robot heading relative to the jump direction [rad]
# Box placement errors around the push target (BOX_TARGET_POS: flush, centered, square).
# These ranges must cover the box-placement study, so the policies are trained on every
# case the study tests.
BOX_GAP_RANGE = (0.0, 0.15)  # gap between the box and the platform face [m]
BOX_LATERAL_RANGE = (-0.15, 0.15)  # sideways offset of the box [m]
BOX_YAW_RANGE = (-np.deg2rad(15.0), np.deg2rad(15.0))  # box rotation [rad]
