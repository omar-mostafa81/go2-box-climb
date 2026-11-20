"""The movable box and the fixed platform."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from go2_box_climb.common import constants as C

# Movable box: a dynamic rigid body (has mass, can be pushed).
BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    spawn=sim_utils.CuboidCfg(
        size=C.BOX_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=C.BOX_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            static_friction=C.BOX_FRICTION,
            dynamic_friction=C.BOX_FRICTION,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.6, 0.25)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=C.BOX_START_POS),
)

# Fixed platform: collision shape only, no rigid body => static (never moves).
PLATFORM_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Platform",
    spawn=sim_utils.CuboidCfg(
        size=C.PLATFORM_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            static_friction=C.GROUND_FRICTION,
            dynamic_friction=C.GROUND_FRICTION,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.5)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=C.PLATFORM_POS),
)
