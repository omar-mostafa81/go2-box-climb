"""The shared scene: ground, Go2, movable box, fixed platform, lights.

Every environment (walking, jumping, full task, studies) builds on this scene so
that all parts see exactly the same world.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from go2_box_climb.assets import BOX_CFG, GO2_CFG, PLATFORM_CFG
from go2_box_climb.common import constants as C


@configclass
class BoxClimbSceneCfg(InteractiveSceneCfg):
    """Go2 + box + platform. One copy of this is created per parallel environment."""

    num_envs: int = 1
    env_spacing: float = C.ENV_SPACING

    # Flat ground with the same friction as Isaac Lab's locomotion tasks
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=C.GROUND_FRICTION,
            dynamic_friction=C.GROUND_FRICTION,
        ),
    )

    robot: ArticulationCfg = GO2_CFG
    box: RigidObjectCfg = BOX_CFG
    platform: AssetBaseCfg = PLATFORM_CFG

    # Contact forces on every robot body (feet contacts, falls, touching the box)
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    )
    sun_light = AssetBaseCfg(prim_path="/World/SunLight", spawn=sim_utils.DistantLightCfg(intensity=2000.0))
