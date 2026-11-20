"""The Go2 robot configuration used by every environment in the project.

It starts from Isaac Lab's ``UNITREE_GO2_CFG`` (robot model file and physics
settings) and sets the start pose and motor settings explicitly from
``go2_box_climb.common.constants``, so the robot contract has one source of truth.
"""

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from go2_box_climb.common import constants as C

GO2_CFG: ArticulationCfg = UNITREE_GO2_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=C.ROBOT_START_POS,
        joint_pos=dict(zip(C.JOINT_NAMES, C.DEFAULT_JOINT_POS.tolist())),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=C.KP,
            damping=C.KD,
            effort_limit=C.EFFORT_LIMIT,
            saturation_effort=C.EFFORT_LIMIT,
            velocity_limit=C.VELOCITY_LIMIT,
            friction=0.0,
        ),
    },
)
