"""The warehouse scene: the final test / demo world.

Same task geometry and physics as the basic scene (box and platform sizes, masses,
frictions and positions all come from ``go2_box_climb.common.constants``), but
everything looks like a real warehouse:
    * the box is a cardboard delivery box,
    * the platform is a concrete staging platform with pallets and boxes on it,
    * the whole thing stands inside NVIDIA's full warehouse (racks, forklift, ...).

Meant for one environment (num_envs = 1): the building surrounds env 0 only.
Training uses the lighter basic scene (or its own scenes); policies do not see
the building, because they observe joint states and object poses, not images.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from go2_box_climb.assets.warehouse import (
    DECORATIONS,
    DELIVERY_BOX_CFG,
    PLATFORM_SAFETY_BAND_CFG,
    STAGING_PLATFORM_CFG,
    WAREHOUSE_CFG,
)
from go2_box_climb.common import constants as C

from .box_climb_scene import BoxClimbSceneCfg


@configclass
class WarehouseSceneCfg(BoxClimbSceneCfg):
    """Go2 + delivery box + staging platform inside a warehouse."""

    # The basic scene's grid floor is replaced by an INVISIBLE collision plane with
    # the same friction, so the warehouse's own floor is what you see.
    terrain = None
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            visible=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=C.GROUND_FRICTION,
                dynamic_friction=C.GROUND_FRICTION,
            ),
        ),
    )

    box = DELIVERY_BOX_CFG
    platform = STAGING_PLATFORM_CFG
    platform_safety_band = PLATFORM_SAFETY_BAND_CFG
    warehouse = WAREHOUSE_CFG

    def __post_init__(self):
        # Add every decoration (pallets, stacked boxes) as its own scene entry.
        for name, cfg in DECORATIONS.items():
            setattr(self, f"decor_{name}", cfg)
