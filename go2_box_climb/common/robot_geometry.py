"""Go2 robot geometry, read from the SAME USD file the simulator loads (rule 13).

Every part (walking, trajectory optimization, IK, MPPI) gets link lengths, joint
offsets and the foot size from here, never typed in by hand. The file path comes
from Isaac Lab's own Go2 configuration (UNITREE_GO2_CFG, which our GO2_CFG is
built from), and the file is read with ``pxr`` (the USD library). Neither starts
Isaac Sim, so this takes about a second.

The USD is downloaded once into ~/.cache/go2_box_climb/ (outside the repository)
and read from there afterwards.

Definitions (all in meters, in the base/body frame at the default zero joint angles:
x forward, y left, z up):
    hip_joint_pos[leg]    position of the hip (ab/adduction) joint relative to the base origin
    thigh_joint_offset[leg]  position of the thigh joint relative to the hip joint
    thigh_length          thigh joint -> knee (calf) joint   (same for all legs)
    calf_length           knee joint -> foot CENTER          (same for all legs)
    foot_radius           radius of the foot's collision sphere (foot center is this
                          high above the ground when the foot touches it)
    head_upper_pos / head_lower_pos   head frames relative to the base origin
Legs: "FL", "FR", "RL", "RR".
"""

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LEGS = ("FL", "FR", "RL", "RR")
CACHE_DIR = Path.home() / ".cache" / "go2_box_climb" / "go2_usd"
# Files of the Go2 model we read (the main file, and the one holding the collision shapes)
_FILES = ("go2.usd", "Props/instanceable_meshes.usd")


@dataclass(frozen=True)
class Go2Geometry:
    hip_joint_pos: dict[str, tuple[float, float, float]]
    thigh_joint_offset: dict[str, tuple[float, float, float]]
    thigh_length: float
    calf_length: float
    foot_radius: float
    head_upper_pos: tuple[float, float, float]
    head_lower_pos: tuple[float, float, float]
    usd_url: str


def go2_usd_url() -> str:
    """The Go2 USD the simulator loads (Isaac Lab's UNITREE_GO2_CFG; no simulator start)."""
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    return UNITREE_GO2_CFG.spawn.usd_path


def _local_copy(url: str) -> Path:
    """Download the Go2 USD files once into CACHE_DIR; return the local path of go2.usd."""
    base = url.rsplit("/", 1)[0]
    for name in _FILES:
        dst = CACHE_DIR / name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            urllib.request.urlretrieve(f"{base}/{name}", tmp)
            tmp.rename(dst)
    return CACHE_DIR / "go2.usd"


@lru_cache(maxsize=1)
def go2_geometry() -> Go2Geometry:
    """Read the Go2 geometry from its USD (cached after the first call)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    url = go2_usd_url()
    path = _local_copy(url)
    stage = Usd.Stage.Open(str(path))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Go2 USD is not in meters")
    root = stage.GetDefaultPrim().GetPath()

    def find_joint(name: str):
        return next(p for p in stage.Traverse() if p.GetName() == name)

    def pos0(name: str) -> tuple[float, float, float]:
        return tuple(float(v) for v in UsdPhysics.Joint(find_joint(name)).GetLocalPos0Attr().Get())

    hip = {leg: pos0(f"{leg}_hip_joint") for leg in LEGS}
    thigh_off = {leg: pos0(f"{leg}_thigh_joint") for leg in LEGS}
    thigh = {leg: abs(pos0(f"{leg}_calf_joint")[2]) for leg in LEGS}
    calf = {leg: abs(pos0(f"{leg}_foot_joint")[2]) for leg in LEGS}
    if len({round(v, 6) for v in thigh.values()}) != 1 or len({round(v, 6) for v in calf.values()}) != 1:
        raise ValueError(f"Legs have different lengths: thigh {thigh}, calf {calf}")

    # foot radius: the collision sphere of the foot (stored in the referenced mesh file)
    meshes = Usd.Stage.Open(str(CACHE_DIR / "Props" / "instanceable_meshes.usd"))
    radii = {
        float(p.GetAttribute("radius").Get())
        for p in meshes.Traverse()
        if p.GetTypeName() == "Sphere" and "_foot/collisions" in str(p.GetPath())
    }
    if len(radii) != 1:
        raise ValueError(f"Could not find one foot collision radius: {radii}")

    head_upper = pos0("Head_upper_joint")
    head_lower_rel = pos0("Head_lower_joint")
    head_lower = tuple(a + b for a, b in zip(head_upper, head_lower_rel))
    return Go2Geometry(
        hip_joint_pos=hip,
        thigh_joint_offset=thigh_off,
        thigh_length=thigh["FL"],
        calf_length=calf["FL"],
        foot_radius=radii.pop(),
        head_upper_pos=head_upper,
        head_lower_pos=head_lower,
        usd_url=url,
    )
