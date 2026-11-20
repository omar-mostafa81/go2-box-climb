"""Warehouse look: delivery box, staging platform, decorations and the building.

All 3D models come from NVIDIA's Isaac Sim asset library ("Simple_Warehouse").
They are downloaded the first time they are used.

Physics rule: everything the robot interacts with (box, platform) keeps the exact
cuboid shape and values from ``go2_box_climb.common.constants``; the warehouse
models only change how things LOOK. The building itself is visual only.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from go2_box_climb.common import constants as C

from .spawners import CuboidWithVisualCfg

WAREHOUSE_DIR = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse"
PROPS_DIR = f"{WAREHOUSE_DIR}/Props"

# Cardboard box model "SM_CardBoxC_01": 0.50 x 0.50 x 0.25 m, origin at its bottom face.
_CARDBOX_C_SIZE = (0.5, 0.5, 0.25)

# Quaternions are (x, y, z, w), as everywhere in the project.
_YAW_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


# -----------------------------------------------------------------------------
# Delivery box: exact cuboid physics (same as the basic scene), cardboard look
# -----------------------------------------------------------------------------
DELIVERY_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    spawn=CuboidWithVisualCfg(
        size=C.BOX_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=C.BOX_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            static_friction=C.BOX_FRICTION,
            dynamic_friction=C.BOX_FRICTION,
        ),
        visual_usd_path=f"{PROPS_DIR}/SM_CardBoxC_01.usd",
        # stretch the 0.5 x 0.5 x 0.25 model to the contract size (0.6 x 0.6 x 0.25)
        visual_scale=tuple(s / m for s, m in zip(C.BOX_SIZE, _CARDBOX_C_SIZE)),
        # model origin is its bottom face; the cuboid origin is its center
        visual_offset=(0.0, 0.0, -C.BOX_SIZE[2] / 2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=C.BOX_START_POS),
)

# -----------------------------------------------------------------------------
# Staging platform: same static cuboid as the basic scene, concrete look,
# plus a yellow safety band on the front face (visual only, no collider)
# -----------------------------------------------------------------------------
STAGING_PLATFORM_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Platform",
    spawn=sim_utils.CuboidCfg(
        size=C.PLATFORM_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            static_friction=C.GROUND_FRICTION,
            dynamic_friction=C.GROUND_FRICTION,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.29), roughness=0.9),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=C.PLATFORM_POS),
)

_BAND_HEIGHT, _BAND_THICKNESS = 0.08, 0.002  # [m]
PLATFORM_SAFETY_BAND_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/PlatformSafetyBand",
    spawn=sim_utils.CuboidCfg(
        size=(_BAND_THICKNESS, C.PLATFORM_SIZE[1], _BAND_HEIGHT),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.05)),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(C.PLATFORM_FRONT_X - _BAND_THICKNESS / 2, 0.0, C.PLATFORM_SIZE[2] - _BAND_HEIGHT / 2)
    ),
)

# -----------------------------------------------------------------------------
# Decorations: pallets and stacked boxes. They are solid (static colliders) so the
# robot cannot walk through them. On the platform they stay behind the landing
# zone (x >= PLATFORM_FRONT_X + PLATFORM_CLEAR_DEPTH).
# -----------------------------------------------------------------------------
_TOP = C.PLATFORM_SIZE[2]  # platform top height
_BACK = C.PLATFORM_FRONT_X + C.PLATFORM_SIZE[0]  # platform back edge x
_PALLET_H = 0.211  # SM_PaletteA_01 height
_CARDBOX_A_H = 0.5  # SM_CardBoxA_01 height (0.70 x 0.50 x 0.50)
_CARDBOX_C_H = 0.25  # SM_CardBoxC_01 height


def _prop(name: str, model: str, pos, rot=(0.0, 0.0, 0.0, 1.0)) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Decor/{name}",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{PROPS_DIR}/{model}.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


_px = _BACK - 0.5  # pallet center x on the platform (pallet is 1.0 m deep once rotated)
_sx = _BACK - 0.36  # x of the corner stacks
DECORATIONS: dict[str, AssetBaseCfg] = {
    # On the platform, back center: pallet + two big boxes + one box on top
    "pallet_top": _prop("pallet_top", "SM_PaletteA_01", (_px, 0.0, _TOP), _YAW_90),
    "pallet_top_box_l": _prop("pallet_top_box_l", "SM_CardBoxA_01", (_px, 0.26, _TOP + _PALLET_H)),
    "pallet_top_box_r": _prop("pallet_top_box_r", "SM_CardBoxA_01", (_px, -0.26, _TOP + _PALLET_H)),
    "pallet_top_box_up": _prop(
        "pallet_top_box_up", "SM_CardBoxC_01", (_px, 0.0, _TOP + _PALLET_H + _CARDBOX_A_H), _yaw_quat(0.3)
    ),
    # On the platform, back corners
    "corner_l_big": _prop("corner_l_big", "SM_CardBoxA_01", (_sx, 1.1, _TOP)),
    "corner_l_small": _prop("corner_l_small", "SM_CardBoxD_04", (_sx, 1.1, _TOP + _CARDBOX_A_H), _yaw_quat(-0.4)),
    "corner_r_mid": _prop("corner_r_mid", "SM_CardBoxC_01", (_sx - 0.1, -1.1, _TOP)),
    "corner_r_small": _prop("corner_r_small", "SM_CardBoxD_04", (_sx - 0.1, -1.1, _TOP + _CARDBOX_C_H)),
    # On the floor, beside the platform (away from the robot's path)
    "pallet_floor": _prop("pallet_floor", "SM_PaletteA_01", (3.0, -2.6, 0.0)),
    "pallet_floor_box_1": _prop("pallet_floor_box_1", "SM_CardBoxA_01", (2.75, -2.6, _PALLET_H)),
    "pallet_floor_box_2": _prop("pallet_floor_box_2", "SM_CardBoxC_01", (3.35, -2.45, _PALLET_H), _yaw_quat(0.2)),
}

# -----------------------------------------------------------------------------
# The building: NVIDIA's "full_warehouse" (racks full of boxes, forklift, signs).
# Placed so the robot stands on the open floor and looks down a rack aisle; the
# racks start about 3 m behind the platform. Visual only: its colliders are
# switched off, the robot stands on our own invisible ground plane instead.
# -----------------------------------------------------------------------------
WAREHOUSE_SPOT = (-8.0, 2.0)  # point of the building (its own frame) placed at the env origin [m]
WAREHOUSE_YAW = -math.pi / 2  # rotates the building so its rack aisles run along our +x


def _warehouse_translation() -> tuple[float, float, float]:
    c, s = math.cos(WAREHOUSE_YAW), math.sin(WAREHOUSE_YAW)
    x, y = WAREHOUSE_SPOT
    return (-(c * x - s * y), -(s * x + c * y), 0.0)


WAREHOUSE_CFG = AssetBaseCfg(
    prim_path="/World/Warehouse",  # one building for the whole world (not cloned per env)
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WAREHOUSE_DIR}/full_warehouse.usd",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=_warehouse_translation(), rot=_yaw_quat(WAREHOUSE_YAW)),
)
