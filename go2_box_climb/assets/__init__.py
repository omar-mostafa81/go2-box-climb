"""Asset configurations (Isaac-dependent: import only after launching the app)."""

from .objects import BOX_CFG, PLATFORM_CFG
from .robot import GO2_CFG

__all__ = ["BOX_CFG", "GO2_CFG", "PLATFORM_CFG"]
