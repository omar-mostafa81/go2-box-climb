"""Shared scenes used by all environments. Part: scene (team leader).

Scenes are picked by name (scripts expose this as ``--scene <name>``):
    basic      box + platform on a plain grid floor; cloneable into many robots, so the
               jump policies are TRAINED here (same geometry and physics as the warehouse)
    warehouse  same geometry and physics, inside a full warehouse (final test / demos)

``SCENES`` and ``get_scene_cfg_class`` can be imported BEFORE Isaac Sim starts
(e.g. to build the ``--scene`` argument); the Isaac Lab scene classes are only
imported when a scene is actually requested.
"""

SCENES = ("basic", "warehouse")


def get_scene_cfg_class(name: str):
    """Return the scene config class for ``name`` (call only after launching the app)."""
    if name == "basic":
        from .box_climb_scene import BoxClimbSceneCfg

        return BoxClimbSceneCfg
    if name == "warehouse":
        from .warehouse_scene import WarehouseSceneCfg

        return WarehouseSceneCfg
    raise ValueError(f"Unknown scene {name!r}; expected one of {SCENES}")


def __getattr__(attr):
    # Keeps `from go2_box_climb.envs.scene import BoxClimbSceneCfg` working (lazy import).
    if attr == "BoxClimbSceneCfg":
        return get_scene_cfg_class("basic")
    if attr == "WarehouseSceneCfg":
        return get_scene_cfg_class("warehouse")
    raise AttributeError(attr)
