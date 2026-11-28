"""MuJoCo model of the Go2, the box and the platform, built from the simulator's own files.

A second physics engine to test the planned jump on a laptop without Isaac Sim. To
compare like with like, nothing is typed in by hand (rule 13):
    * the robot comes from the same Go2 USD Isaac Sim loads: every link's mass, center
      of mass and inertia, every joint's position, axis and limits, and every
      collision shape (trunk box, head, hip motors, thigh boxes, calf cylinders, feet),
    * the box and the platform come from common/constants.py (size, mass, friction),
    * the physics step is SIM_DT (0.005 s), like Isaac Lab.

Matching Isaac Lab's choices:
    * no collisions between the robot's own parts (Isaac's Go2 has self-collisions off),
    * friction: Isaac multiplies the two surfaces' values. MuJoCo normally takes the
      larger one; giving the box a higher "priority" makes every box contact use the
      box's 0.5, which is what multiplying gives with feet, ground and platform at 1.0,
    * contacts: an elliptic friction cone with impratio 100, so feet stick like in PhysX.
      With MuJoCo's defaults the feet slowly creep sideways; with these settings the
      robot holding its default pose settles at 0.279 m, as Isaac reports (~0.28 m),
    * the motors are plain torque inputs: the PD controller and the DC-motor limit are
      computed outside, exactly like Isaac Lab's DCMotor actuator (see mujoco_sim.py).
"""

import math

import numpy as np

from go2_box_climb.common import constants as C
from go2_box_climb.common import robot_geometry as rg

from .robot_params import _geometry

LINK_COLORS = {"F": "0.16 0.47 0.84 1", "R": "0.92 0.41 0.20 1"}  # front legs blue, rear legs orange
TRUNK_COLOR = "0.80 0.80 0.78 1"


def _usd_robot():
    """Links, joints and collision shapes of the Go2, read from the cached USD files."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    _geometry()  # makes sure the USD is downloaded (same file Isaac Sim loads)
    stage = Usd.Stage.Open(str(rg.CACHE_DIR / "go2.usd"))
    links, joints = {}, {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            q = api.GetPrincipalAxesAttr().Get()
            links[name] = {
                "mass": float(api.GetMassAttr().Get()),
                "com": tuple(float(v) for v in api.GetCenterOfMassAttr().Get()),
                "inertia": tuple(float(v) for v in api.GetDiagonalInertiaAttr().Get()),
                "quat": (float(q.GetReal()), *(float(v) for v in q.GetImaginary())),
            }
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            parent = str(j.GetBody0Rel().GetTargets()[0]).rsplit("/", 1)[-1]
            child = str(j.GetBody1Rel().GetTargets()[0]).rsplit("/", 1)[-1]
            rot0, rot1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            if abs(Gf.Dot(rot0, rot1)) < 1 - 1e-6 or any(abs(v) > 1e-9 for v in j.GetLocalPos1Attr().Get()):
                raise ValueError(f"{name}: child frame is turned or moved relative to the joint; not supported")
            entry = {"name": name, "parent": parent, "pos": tuple(float(v) for v in j.GetLocalPos0Attr().Get())}
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rj = UsdPhysics.RevoluteJoint(prim)
                axis_name = rj.GetAxisAttr().Get()
                unit = Gf.Vec3d(*(1.0 if a == axis_name else 0.0 for a in "XYZ"))
                axis = Gf.Rotation(Gf.Quatd(rot0)).TransformDir(unit)  # joint axis in the parent link frame
                entry["axis"] = tuple(round(float(v), 9) for v in axis)
                entry["range"] = (math.radians(rj.GetLowerLimitAttr().Get()), math.radians(rj.GetUpperLimitAttr().Get()))
            joints[child] = entry

    meshes = Usd.Stage.Open(str(rg.CACHE_DIR / "Props" / "instanceable_meshes.usd"))
    shapes: dict[str, list] = {}
    for prim in meshes.Traverse():
        path = str(prim.GetPath())
        kind = prim.GetTypeName()
        if "/collisions/" not in path or kind not in ("Cube", "Cylinder", "Sphere"):
            continue
        link = path.split("/collisions/")[0].rsplit("/", 1)[-1]
        xf = Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        q = xf.GetRotation().GetQuat()
        shape = {
            "kind": kind,
            "pos": tuple(float(v) for v in xf.GetTranslation()),
            "quat": (float(q.GetReal()), *(float(v) for v in q.GetImaginary())),
        }
        scale = np.array(xf.GetScale(), dtype=float)
        if kind == "Cube":
            shape["size"] = tuple(float(prim.GetAttribute("size").Get()) * scale / 2)  # MuJoCo boxes use half sizes
        elif kind == "Cylinder":
            if prim.GetAttribute("axis").Get() != "Z":
                raise ValueError(f"{path}: only Z-axis cylinders are supported")
            shape["size"] = (float(prim.GetAttribute("radius").Get()), float(prim.GetAttribute("height").Get()) / 2)
        else:
            shape["size"] = (float(prim.GetAttribute("radius").Get()),)
        shapes.setdefault(link, []).append(shape)
    return links, joints, shapes


MESH_CACHE = rg.CACHE_DIR.parent / "go2_visual_meshes"  # outside the repository, next to the USD cache


def _write_stl(path, triangles: np.ndarray):
    """Binary STL file from (F, 3, 3) triangle corners."""
    tri = np.asarray(triangles, dtype=np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    record = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    record["n"], record["v"] = normals, tri
    with open(path, "wb") as f:
        f.write(b"Go2 visual mesh from the Isaac Sim USD".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(record.tobytes())


def export_visual_meshes() -> dict[str, list[tuple[str, tuple[float, float, float]]]]:
    """Write the Go2's real visual meshes (from the Isaac Sim USD) as STL files, one per colored part.

    Every link's visual mesh is split by its material subsets; each part keeps the
    material's color (UsdPreviewSurface diffuse color). Files are written once into
    MESH_CACHE and reused.

    Returns:
        {link name: [(STL file path, (r, g, b)), ...]}
    """
    from pxr import Usd, UsdGeom, UsdShade

    _geometry()  # makes sure the USD files are downloaded
    stage = Usd.Stage.Open(str(rg.CACHE_DIR / "Props" / "instanceable_meshes.usd"))
    MESH_CACHE.mkdir(parents=True, exist_ok=True)

    def color_of(prim) -> tuple[float, float, float]:
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            for shader in material.GetPrim().GetChildren():
                for name in ("diffuse_color_constant", "diffuseColor"):
                    value = UsdShade.Shader(shader).GetInput(name)
                    if value and value.Get() is not None:
                        return tuple(float(c) for c in value.Get())
        return (0.6, 0.6, 0.6)

    parts: dict[str, list] = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() != "Mesh" or "/visuals/" not in path:
            continue
        link = path.split("/visuals/")[0].rsplit("/", 1)[-1]
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        if set(mesh.GetFaceVertexCountsAttr().Get()) != {3}:
            raise ValueError(f"{path}: only triangle meshes are supported")
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
        subsets = [c for c in prim.GetChildren() if c.GetTypeName() == "GeomSubset"]
        groups = [(np.array(UsdGeom.Subset(s).GetIndicesAttr().Get(), dtype=int), color_of(s)) for s in subsets]
        if not groups:
            groups = [(np.arange(len(faces)), color_of(prim))]
        for i, (face_ids, color) in enumerate(groups):
            file = MESH_CACHE / f"{link}_{i}.stl"
            if not file.exists():
                _write_stl(file, points[faces[face_ids]])
            parts.setdefault(link, []).append((str(file), color))
    return parts


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def go2_scene_xml(foot_friction: float = 1.0, visual_meshes: dict | None = None) -> str:
    """MJCF text of the jump-1 scene: ground, Go2, the box flush against the platform.

    Args:
        visual_meshes: ``export_visual_meshes()`` to draw the robot with its real Isaac Sim
            meshes. The collision shapes are then kept but hidden (geom group 3).
    """
    links, joints, shapes = _usd_robot()
    children: dict[str, list] = {}
    for child, j in joints.items():
        children.setdefault(j["parent"], []).append(child)

    def body_xml(name: str, indent: int) -> str:
        pad = " " * indent
        link = links[name]
        color = LINK_COLORS.get(name[0], TRUNK_COLOR) if name[1:2] in ("L", "R") else TRUNK_COLOR
        parts = [f'{pad}  <inertial pos="{_fmt(link["com"])}" quat="{_fmt(link["quat"])}" mass="{link["mass"]:.6g}" '
                 f'diaginertia="{_fmt(link["inertia"])}"/>']  # fmt: skip
        j = joints.get(name)
        if j is not None and "axis" in j:
            parts.append(f'{pad}  <joint name="{j["name"]}" axis="{_fmt(j["axis"])}" range="{_fmt(j["range"])}"/>')
        hidden = ' group="3"' if visual_meshes else ""
        for i, s in enumerate(shapes.get(name, [])):
            kind = {"Cube": "box", "Cylinder": "cylinder", "Sphere": "sphere"}[s["kind"]]
            parts.append(f'{pad}  <geom name="{name}_{i}" type="{kind}" size="{_fmt(s["size"])}" pos="{_fmt(s["pos"])}" '
                         f'quat="{_fmt(s["quat"])}" rgba="{color}" class="robot"{hidden}/>')  # fmt: skip
        for i, (_file, rgb) in enumerate((visual_meshes or {}).get(name, [])):
            parts.append(f'{pad}  <geom name="{name}_visual_{i}" type="mesh" mesh="{name}_visual_{i}" '
                         f'rgba="{_fmt(rgb)} 1" contype="0" conaffinity="0" group="1" mass="0"/>')  # fmt: skip
        if name.endswith("_foot"):
            parts.append(f'{pad}  <site name="{name}_center" size="0.005"/>')
        for child in sorted(children.get(name, [])):
            parts.append(body_xml(child, indent + 2))
        pos = joints[name]["pos"] if name in joints else (0, 0, 0)
        head = f'{pad}<body name="{name}" pos="{_fmt(pos)}">' if name != "base" else f'{pad}<body name="base" pos="0 0 0.4">'
        root = f"{pad}  <freejoint name=\"root\"/>\n" if name == "base" else ""
        return head + "\n" + root + "\n".join(parts) + f"\n{pad}</body>"

    bx, by, bz = C.BOX_SIZE
    px, py, pz = C.PLATFORM_SIZE
    actuators = "\n".join(f'    <motor name="{n}" joint="{n}" ctrlrange="{-C.EFFORT_LIMIT} {C.EFFORT_LIMIT}"/>'
                          for n in C.JOINT_NAMES)  # fmt: skip
    mesh_assets = "\n".join(f'    <mesh name="{link}_visual_{i}" file="{file}" inertia="shell"/>'
                            for link, items in (visual_meshes or {}).items()
                            for i, (file, _rgb) in enumerate(items))  # fmt: skip
    return f"""<mujoco model="go2_box_climb_jump1">
  <compiler angle="radian"/>
  <!-- elliptic friction cone + high impratio: feet stick like in PhysX instead of slowly creeping -->
  <option timestep="{C.SIM_DT}" integrator="implicitfast" cone="elliptic" impratio="100"/>
  <visual>
    <global offwidth="1600" offheight="800"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.96 0.95" rgb2="0.86 0.88 0.90" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.88 0.88 0.86" rgb2="0.82 0.82 0.80"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0"/>
{mesh_assets}
  </asset>
  <default>
    <default class="robot">
      <geom contype="1" conaffinity="0" friction="{foot_friction} 0.005 0.0001"/>
    </default>
  </default>
  <worldbody>
    <light pos="1.2 -1.5 3" dir="-0.2 0.4 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light pos="-1 1 2.5" dir="0.3 -0.3 -1" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <camera name="side" pos="1.3 -1.75 0.62" xyaxes="1 0 0 0 0.3 1"/>
    <geom name="ground" type="plane" size="6 6 0.1" material="grid" friction="{C.GROUND_FRICTION} 0.005 0.0001"/>
    <geom name="platform" type="box" size="{px / 2} {py / 2} {pz / 2}" pos="{_fmt(C.PLATFORM_POS)}"
          rgba="0.72 0.72 0.70 1" friction="{C.GROUND_FRICTION} 0.005 0.0001"/>
    <camera name="three_quarter" pos="0.35 -1.45 0.95" mode="targetbody" target="box"/>
    <body name="box" pos="{_fmt(C.BOX_TARGET_POS)}">
      <freejoint name="box_root"/>
      <geom name="box" type="box" size="{bx / 2} {by / 2} {bz / 2}" mass="{C.BOX_MASS}" rgba="0.78 0.60 0.40 1"
            friction="{C.BOX_FRICTION} 0.005 0.0001" priority="1"/>
    </body>
{body_xml("base", 4)}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def load_scene(foot_friction: float = 1.0, real_meshes: bool = False):
    """Compile the scene. Returns (mujoco.MjModel, mujoco.MjData).

    Args:
        real_meshes: also draw the robot with its real Isaac Sim meshes. They are visual
            only (no mass, no collisions), so the physics is exactly the same.
    """
    import mujoco

    meshes = export_visual_meshes() if real_meshes else None
    model = mujoco.MjModel.from_xml_string(go2_scene_xml(foot_friction, visual_meshes=meshes))
    return model, mujoco.MjData(model)
