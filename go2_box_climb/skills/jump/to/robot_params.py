"""Robot numbers the jump optimizer needs, all read from the simulator's Go2 USD (rule 13).

Geometry (leg lengths, hip positions, foot radius, head position) comes from the shared
``go2_geometry()``. The optimizer also needs things ``go2_geometry()`` does not provide
yet, read here from the same USD files:
    * mass of every link -> total mass, center of mass and pitch inertia of the robot,
      and the inertia each leg motor has to swing,
    * the collision shapes (trunk box, head, hips, thighs, calves), so the plan keeps
      the real robot body out of the box and the platform,
    * the joint limits.

TODO (team leader approval, rules 10 and 13): move ``_read_usd_extras`` into
``go2_box_climb/common/robot_geometry.py`` so every part shares it.

Laptop use (no Isaac Lab): ``go2_geometry()`` asks Isaac Lab for the USD's web address.
When Isaac Lab is missing we use the address Isaac Lab 3.0 / Isaac Sim 6.0 resolves to
(``FALLBACK_GO2_USD_URL``). On a machine with Isaac Lab the real lookup is used.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from go2_box_climb.common import constants as C
from go2_box_climb.common import robot_geometry as rg

from . import planar_model as pm

# Where Isaac Lab (Isaac Sim 6.0 assets) keeps the Go2 USD. Only used when Isaac Lab
# is not installed (for example on a laptop without an NVIDIA GPU).
FALLBACK_GO2_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"
)


@dataclass(frozen=True)
class JumpRobotParams:
    """Everything the side-view (sagittal) jump model needs. Units: m, kg, kg m^2, rad.

    The side-view model has two "legs": the front pair (FL + FR) and the rear pair
    (RL + RR). Each pair moves together, so a pair is drawn and optimized as one leg.
    Body-frame points are (x forward, z up) relative to the base origin.
    """

    mass: float  # whole robot [kg]
    com_offset: tuple[float, float]  # whole-robot center of mass in the body frame, default pose [m]
    pitch_inertia: float  # whole robot about its center of mass, body y axis, default pose [kg m^2]
    thigh_swing_inertia: float  # one leg about its thigh joint, leg straight (upper bound, for torque limits) [kg m^2]
    thigh_swing_inertia_nominal: float  # one leg about its thigh joint, default knee angle (for momentum) [kg m^2]
    calf_swing_inertia: float  # one calf + foot about its knee [kg m^2]
    thigh_length: float  # [m]
    calf_length: float  # [m]
    foot_radius: float  # [m]
    front_hip_x: float  # x of the front hips in the body frame [m] (> 0)
    rear_hip_x: float  # x of the rear hips in the body frame [m] (< 0)
    # The default pose turns each hip 0.1 rad sideways (DEFAULT_JOINT_POS). The thigh joint sits
    # 9.55 cm to the side of the hip axis, so this lifts it by 0.0955 sin(0.1) = 9.5 mm, and the
    # leg's plane leans outward, so the leg reaches cos(0.1) as far down.
    thigh_joint_z: float  # height of the thigh joints above the hip axis in the body frame [m]
    leg_tilt_cos: float  # cos(default hip sideways angle)
    hip_y: float  # sideways distance of the feet from the body center line, default pose [m]
    trunk_half_size: tuple[float, float]  # trunk collision box, half length (x) and half height (z) [m]
    head_upper: tuple[float, float, float]  # (x, z, radius) of the upper head collision shape [m]
    head_lower: tuple[float, float, float]  # (x, z, radius) of the lower head collision sphere [m]
    hip_radius: float  # hip motor collision cylinder [m]
    thigh_half_thickness: float  # thigh collision box, half thickness in the side view [m]
    calf_radius: float  # thickest calf collision cylinder [m]
    thigh_limits_front: tuple[float, float]  # [rad]
    thigh_limits_rear: tuple[float, float]  # [rad]
    calf_limits: tuple[float, float]  # [rad]
    usd_url: str


def _geometry() -> rg.Go2Geometry:
    """go2_geometry(), with the fallback address when Isaac Lab is not installed."""
    try:
        return rg.go2_geometry()
    except ModuleNotFoundError:  # isaaclab_assets missing: laptop without Isaac Lab
        original = rg.go2_usd_url
        rg.go2_usd_url = lambda: FALLBACK_GO2_USD_URL
        try:
            rg.go2_geometry.cache_clear()
            return rg.go2_geometry()
        finally:
            rg.go2_usd_url = original


def _read_usd_extras():
    """Link masses / centers of mass / inertias, collision shapes and joint limits from the USD.

    Returns:
        links: {link name: (mass, (com_x, com_z) in the link frame, Iyy)}
        shapes: {link name: list of (type, attributes, scale)} for the collision shapes
        limits: {joint name: (lower, upper)} [rad]
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(rg.CACHE_DIR / "go2.usd"))
    links, limits = {}, {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            com = api.GetCenterOfMassAttr().Get()
            links[name] = (float(api.GetMassAttr().Get()), (float(com[0]), float(com[2])),
                           float(api.GetDiagonalInertiaAttr().Get()[1]))  # fmt: skip
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            limits[name] = (math.radians(j.GetLowerLimitAttr().Get()), math.radians(j.GetUpperLimitAttr().Get()))

    meshes = Usd.Stage.Open(str(rg.CACHE_DIR / "Props" / "instanceable_meshes.usd"))
    shapes: dict[str, list] = {}
    for prim in meshes.Traverse():
        path = str(prim.GetPath())
        kind = prim.GetTypeName()
        if "/collisions/" not in path or kind not in ("Cube", "Cylinder", "Sphere"):
            continue
        link = path.split("/collisions/")[0].rsplit("/", 1)[-1]
        attrs = {a: prim.GetAttribute(a).Get() for a in ("size", "radius", "height") if prim.GetAttribute(a)}
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        scale = tuple(float(v) for v in Gf.Transform(matrix).GetScale())
        shapes.setdefault(link, []).append((kind, attrs, scale))
    return links, shapes, limits


def _mass_properties(links, g: rg.Go2Geometry, L1: float, L2: float):
    """Center of mass and pitch inertia of the whole robot in the default pose (side view).

    Each link's own inertia plus its mass times its squared distance to the robot's
    center of mass (parallel-axis theorem). Link frames: hip link at the hip joint,
    thigh link at the thigh joint turned by the thigh angle, calf link at the knee
    turned by thigh + calf angle, foot at the foot center.
    """
    q = C.DEFAULT_JOINT_POS
    points = []  # (mass, x, z, own Iyy) in the body frame

    def add(name, x0, z0, angle):
        m, (cx, cz), iyy = links[name]
        wx, wz = pm.body_to_world(cx, cz, angle)  # rotate the link's center of mass by the link angle
        points.append((m, x0 + wx, z0 + wz, iyy))

    add("base", 0.0, 0.0, 0.0)
    for name in ("Head_upper", "Head_lower"):
        pos = g.head_upper_pos if name == "Head_upper" else g.head_lower_pos
        add(name, pos[0], pos[2], 0.0)
    for i, leg in enumerate(rg.LEGS):
        hx, _, hz = g.hip_joint_pos[leg]
        t, c = q[4 + i], q[8 + i]
        add(f"{leg}_hip", hx, hz, 0.0)
        add(f"{leg}_thigh", hx, hz, t)
        kx, kz = hx - L1 * math.sin(t), hz - L1 * math.cos(t)
        add(f"{leg}_calf", kx, kz, t + c)
        fx, fz = pm.leg_fk(t, c, L1, L2)
        add(f"{leg}_foot", hx + fx, hz + fz, t + c)
    m = np.array([p[0] for p in points])
    xz = np.array([[p[1], p[2]] for p in points])
    total = m.sum()
    com = (m[:, None] * xz).sum(axis=0) / total
    inertia = sum(p[3] for p in points) + float((m * ((xz - com) ** 2).sum(axis=1)).sum())
    return float(total), (float(com[0]), float(com[1])), inertia


def _swing_inertias(links, leg: str, L1: float, L2: float, calf_angle: float) -> tuple[float, float, float]:
    """Inertia the thigh motor and the knee motor have to swing (one leg).

    Returns:
        thigh_max: thigh + calf + foot about the thigh joint with the leg straight (the
            largest it can be, so torque limits stay safe).
        thigh_nominal: the same at the default knee angle (for the legs' momentum).
        knee: calf + foot about the knee.
    """
    mt, (tx, tz), it = links[f"{leg}_thigh"]
    mc, (cx, cz), ic = links[f"{leg}_calf"]
    mf, _, i_f = links[f"{leg}_foot"]
    knee = ic + mc * (cx**2 + cz**2) + i_f + mf * L2**2
    own = it + mt * (tx**2 + tz**2) + ic + i_f  # parts that do not depend on the knee angle

    def about_thigh_joint(c: float) -> float:
        # calf center of mass and foot, relative to the thigh joint, for a thigh angle of 0
        ccx, ccz = pm.body_to_world(cx, cz, c)
        calf = (ccx, -L1 + ccz)
        fx, fz = pm.leg_fk(0.0, c, L1, L2)
        return own + mc * (calf[0] ** 2 + calf[1] ** 2) + mf * (fx**2 + fz**2)

    return about_thigh_joint(0.0), about_thigh_joint(calf_angle), knee


def _shape_sizes(shapes) -> dict[str, float]:
    """Sizes of the collision shapes the jump planner keeps clear of the scene."""
    (_, _, trunk_scale), = [s for s in shapes["base"] if s[0] == "Cube"]
    thigh = [s for s in shapes["FL_thigh"] if s[0] == "Cube"][0][2]
    return {
        "trunk_half_x": trunk_scale[0] / 2,
        "trunk_half_z": trunk_scale[2] / 2,
        "head_upper_r": float(shapes["Head_upper"][0][1]["radius"]),
        "head_lower_r": float(shapes["Head_lower"][0][1]["radius"]),
        "hip_r": max(float(s[1]["radius"]) for s in shapes["FL_hip"] if s[0] == "Cylinder"),
        # thigh box (0.213 x 0.0245 x 0.034) is turned 90 deg about y: its 0.213 side runs along
        # the thigh, 0.0245 is its sideways width, and 0.034 is its thickness seen from the side
        "thigh_half_thickness": thigh[2] / 2,
        "calf_r": max(float(s[1]["radius"]) for s in shapes["FL_calf"] if s[0] == "Cylinder"),
    }


@lru_cache(maxsize=1)
def jump_robot_params() -> JumpRobotParams:
    """Read the jump model's numbers from the Go2 USD (downloaded once, then cached)."""
    g = _geometry()
    links, shapes, limits = _read_usd_extras()
    L1, L2 = g.thigh_length, g.calf_length
    mass, com, inertia = _mass_properties(links, g, L1, L2)
    thigh_i, thigh_nominal, calf_i = _swing_inertias(links, "FL", L1, L2, float(C.DEFAULT_JOINT_POS[8]))
    size = _shape_sizes(shapes)
    hu, hl = g.head_upper_pos, g.head_lower_pos
    hip_angle = abs(float(C.DEFAULT_JOINT_POS[0]))  # same size for all four hips
    offset = abs(g.thigh_joint_offset["FL"][1])
    _, reach = pm.leg_fk(float(C.DEFAULT_JOINT_POS[4]), float(C.DEFAULT_JOINT_POS[8]), L1, L2)
    return JumpRobotParams(
        mass=mass,
        com_offset=com,
        pitch_inertia=inertia,
        thigh_swing_inertia=thigh_i,
        thigh_swing_inertia_nominal=thigh_nominal,
        calf_swing_inertia=calf_i,
        thigh_length=L1,
        calf_length=L2,
        foot_radius=g.foot_radius,
        front_hip_x=g.hip_joint_pos["FL"][0],
        rear_hip_x=g.hip_joint_pos["RL"][0],
        thigh_joint_z=offset * math.sin(hip_angle),
        leg_tilt_cos=math.cos(hip_angle),
        hip_y=g.hip_joint_pos["FL"][1] + offset * math.cos(hip_angle) + abs(reach) * math.sin(hip_angle),
        trunk_half_size=(size["trunk_half_x"], size["trunk_half_z"]),
        head_upper=(hu[0], hu[2], size["head_upper_r"]),
        head_lower=(hl[0], hl[2], size["head_lower_r"]),
        hip_radius=size["hip_r"],
        thigh_half_thickness=size["thigh_half_thickness"],
        calf_radius=size["calf_r"],
        thigh_limits_front=limits["FL_thigh_joint"],
        thigh_limits_rear=limits["RL_thigh_joint"],
        calf_limits=limits["FL_calf_joint"],
        usd_url=g.usd_url,
    )
