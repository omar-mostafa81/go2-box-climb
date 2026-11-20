"""Custom spawner: a simple cuboid for physics, drawn with a detailed 3D model.

Why: the warehouse box model (a cardboard box) has its origin at its bottom face,
but our contract says the box's root position is its CENTER, and the MPPI / TO
models assume a perfect cuboid. So the physics body stays an exact, invisible
cuboid (center origin, clean contacts), and the cardboard model is attached under
it as a child that is only drawn, never collides, and adds no mass.
"""

from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils.configclass import configclass
from pxr import UsdGeom


@clone
def spawn_cuboid_with_visual(prim_path: str, cfg: "CuboidWithVisualCfg", translation=None, orientation=None, **kwargs):
    stage = get_current_stage()
    # 1) The physics body: a normal Isaac Lab cuboid (rigid body, mass, collider, friction)
    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation)
    # 2) Hide the cuboid's own mesh. Hidden prims still collide in PhysX.
    UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")).MakeInvisible()
    # 3) Attach the detailed model as a visual-only child
    visual_path = f"{prim_path}/visual"
    create_prim(
        visual_path, usd_path=cfg.visual_usd_path, translation=cfg.visual_offset, scale=cfg.visual_scale, stage=stage
    )
    schemas.modify_collision_properties(visual_path, sim_utils.CollisionPropertiesCfg(collision_enabled=False), stage)
    return prim


@configclass
class CuboidWithVisualCfg(sim_utils.CuboidCfg):
    """A :class:`CuboidCfg` whose visible shape is replaced by a USD model."""

    func: Callable = spawn_cuboid_with_visual

    visual_usd_path: str = MISSING
    """USD file of the model that is drawn instead of the cuboid."""

    visual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scale applied to the model so it matches the cuboid size."""

    visual_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the model's origin relative to the cuboid center [m]."""
