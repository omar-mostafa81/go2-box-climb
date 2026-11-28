"""Jump references: loading, looking them up in time, and a placeholder (Isaac-free, torch).

A reference (``JumpReference``, common/formats.py) is the jump planned by trajectory
optimization + IK (part TO), one file per jump in references/. The tracking policy
is trained to follow it.

``ReferenceTrack`` holds a reference as torch tensors and returns its values at any
time (linear interpolation between samples; the orientation is interpolated and
re-normalized; foot contacts are taken from the nearest sample). Before the start
it returns the first sample, after the end the last one (the robot should stand
still in the final pose).

Anchoring. The reference is planned for the nominal scene: robot at the nominal start,
box flush against the platform. In every attempt the robot starts somewhere else and
the box is somewhere else (random ranges in common/constants.py), so the reference's
horizontal path is moved onto the actual scene (``Anchor``):
    * at the start it begins where the robot actually stands (shifted and turned about z
      to the robot's start position and heading),
    * from the landing on, it is placed relative to the actual landing target: the box
      for ground_to_box (position and rotation of the box), the platform for
      box_to_platform (fixed, so the planned landing spot itself),
    * in between it blends smoothly from the first to the second (smoothstep in
      time / landing time, where the landing time is the first touch-down after the
      flight, found from the foot contacts).
The heading stays the robot's own start heading (shifted like the start). Heights are
NOT shifted: they are measured from the ground, so the box and platform heights stay
correct. Why: the planned jump has little room on the box (rear feet 7 cm inside its
front edge, head 1 cm from the platform face), so shifting the whole jump with the
start (up to 15 cm) would ask the robot to land off the box or into the platform.

Placeholder. Until the TO member delivers the real files, ``placeholder_reference``
makes a simple, smooth, kinematically plausible jump in the same format (feet on the
start surface, crouch, push off, flight, land on the target surface, settle). It is
NOT physically planned: it only lets the whole pipeline be built and tested. It is
saved to the git-ignored outputs/ folder, never to references/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from go2_box_climb.common import constants as C
from go2_box_climb.common.formats import REFERENCE_FILES, JumpReference
from go2_box_climb.utils.runs import REPO_ROOT

from .jump_math import quat_from_yaw, quat_mul, yaw_of

#: The two jumps, named like the keys of REFERENCE_FILES.
JUMPS = tuple(REFERENCE_FILES)  # ("ground_to_box", "box_to_platform")

#: Where make_placeholder_reference.py writes the placeholders (git-ignored folder).
PLACEHOLDER_DIR = REPO_ROOT / "outputs" / "jump_from_tracking"


def placeholder_path(jump: str) -> Path:
    return PLACEHOLDER_DIR / f"placeholder_jump_{jump}.npz"


def check_jump(jump: str) -> str:
    if jump not in JUMPS:
        raise ValueError(f"Unknown jump {jump!r}; expected one of {JUMPS}")
    return jump


def reference_file(jump: str, path: str | Path | None = None) -> Path:
    """The reference file to use: ``path`` if given, else the final one in references/."""
    check_jump(jump)
    p = Path(path) if path else REFERENCE_FILES[jump]
    if not p.exists():
        raise FileNotFoundError(
            f"Jump reference not found: {p}\n"
            "The final references come from the TO part (references/). Until then, create a placeholder with\n"
            "    python scripts/jump_from_tracking/make_placeholder_reference.py\n"
            f"and pass  --reference {placeholder_path(jump).relative_to(REPO_ROOT)}"
        )
    return p


@dataclass
class ReferenceSample:
    """Reference values at given times (batch of N)."""

    base_pos: torch.Tensor  # (N, 3) [m]
    base_quat: torch.Tensor  # (N, 4) (x, y, z, w)
    joint_pos: torch.Tensor  # (N, 12) [rad], JOINT_NAMES order
    foot_contact: torch.Tensor  # (N, 4) 1.0 = foot on a surface, FEET order


@dataclass
class Anchor:
    """Where the reference is placed in one attempt (batch of N, env frame; see "Anchoring")."""

    start_xy: torch.Tensor  # (N, 2) robot base position at the start [m]
    start_yaw: torch.Tensor  # (N,) robot heading at the start [rad]
    target_xy: torch.Tensor  # (N, 2) actual position of the landing target's reference point [m]
    target_yaw: torch.Tensor  # (N,) actual rotation of the landing target [rad]


def nominal_target(jump: str) -> tuple[float, float]:
    """Reference point of the landing target in the nominal scene (env frame) [m].

    ground_to_box: the box center at the push target (BOX_TARGET_POS).
    box_to_platform: the middle of the platform's front edge (the platform never moves).
    """
    check_jump(jump)
    if jump == "ground_to_box":
        return (C.BOX_TARGET_POS[0], C.BOX_TARGET_POS[1])
    return (C.PLATFORM_FRONT_X, C.PLATFORM_POS[1])


def target_anchor(jump: str, box_pos: torch.Tensor, box_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Actual position (N, 2) and rotation (N,) of the landing target, from the box pose at the start."""
    if check_jump(jump) == "ground_to_box":
        return box_pos[:, :2].clone(), yaw_of(box_quat)
    n = box_pos.shape[0]
    xy = torch.tensor(nominal_target(jump), dtype=box_pos.dtype, device=box_pos.device).expand(n, 2).clone()
    return xy, torch.zeros(n, dtype=box_pos.dtype, device=box_pos.device)


def landing_time(t: np.ndarray, foot_contact: np.ndarray) -> float:
    """First time after the flight when all feet touch a surface again [s] (end time if there is no flight)."""
    airborne = ~np.asarray(foot_contact, dtype=bool).any(axis=1)
    if not airborne.any():
        return float(t[-1])
    first_air = int(np.argmax(airborne))
    down = np.asarray(foot_contact, dtype=bool).all(axis=1)
    after = np.nonzero(down[first_air:])[0]
    return float(t[first_air + after[0]]) if len(after) else float(t[-1])


def _rotate_xy(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]), dim=-1)


class ReferenceTrack:
    """A JumpReference as torch tensors, looked up by time."""

    def __init__(self, ref: JumpReference, device: str | torch.device = "cpu", jump: str | None = None):
        """``jump`` sets the nominal landing target (``nominal_target``); without it the target point is (0, 0)."""
        t = np.asarray(ref.t, dtype=np.float64)
        if t.ndim != 1 or len(t) < 2:
            raise ValueError("A jump reference needs at least 2 time samples.")
        if abs(t[0]) > 1e-6:
            raise ValueError(f"A jump reference must start at t = 0 (it starts at {t[0]}).")
        if np.any(np.diff(t) <= 0):
            raise ValueError("The time stamps of a jump reference must be strictly increasing.")
        q = np.asarray(ref.base_quat, dtype=np.float64).copy()
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            raise ValueError("The jump reference contains a zero quaternion.")
        q /= norms
        for i in range(1, len(q)):  # q and -q are the same rotation: keep neighbours on the same side
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]

        def f(a) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

        self.device = device
        self.t = f(t)
        self.base_pos = f(ref.base_pos)
        self.base_quat = f(q)
        self.joint_pos = f(ref.joint_pos)
        self.foot_contact = f(np.asarray(ref.foot_contact, dtype=bool))
        self.duration = float(t[-1])
        self.num_samples = len(t)
        self.start_pos = self.base_pos[0]  # (3,)
        self.start_yaw = yaw_of(self.base_quat[0])  # scalar tensor
        self.landing_time = landing_time(t, ref.foot_contact)
        self.target_xy_nominal = torch.tensor(nominal_target(jump) if jump else (0.0, 0.0), device=device)

    def at(self, time: torch.Tensor) -> ReferenceSample:
        """Reference values at times ``time`` (N,) [s], as planned (nominal start)."""
        time = torch.as_tensor(time, dtype=torch.float32, device=self.device).reshape(-1).clamp(0.0, self.duration)
        idx = (torch.searchsorted(self.t, time.contiguous(), right=True) - 1).clamp(0, self.num_samples - 2)
        t0, t1 = self.t[idx], self.t[idx + 1]
        a = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0).unsqueeze(-1)

        def lerp(x: torch.Tensor) -> torch.Tensor:
            return x[idx] + a * (x[idx + 1] - x[idx])

        quat = lerp(self.base_quat)
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
        contact = torch.where(a < 0.5, self.foot_contact[idx], self.foot_contact[idx + 1])
        return ReferenceSample(lerp(self.base_pos), quat, lerp(self.joint_pos), contact)

    def blend(self, time: torch.Tensor) -> torch.Tensor:
        """(N,) weight of the target anchor: 0 at the start, smoothly up to 1 at the landing time."""
        t_land = max(self.landing_time, 1e-6)
        x = (torch.as_tensor(time, dtype=torch.float32, device=self.device).reshape(-1) / t_land).clamp(0, 1)
        return x * x * (3 - 2 * x)

    def at_anchored(self, time: torch.Tensor, anchor: Anchor) -> ReferenceSample:
        """Reference values placed in the actual scene (see "Anchoring" above).

        Args:
            time: (N,) time since the jump started [s].
            anchor: the robot's start and the landing target's actual pose (env frame).
        """
        s = self.at(time)
        p = s.base_pos[:, :2]
        dyaw = anchor.start_yaw - self.start_yaw
        from_start = anchor.start_xy + _rotate_xy(dyaw, p - self.start_pos[:2])
        from_target = anchor.target_xy + _rotate_xy(anchor.target_yaw, p - self.target_xy_nominal)
        w = self.blend(time).unsqueeze(-1)
        pos = torch.cat(((1 - w) * from_start + w * from_target, s.base_pos[:, 2:3]), dim=-1)
        quat = quat_mul(quat_from_yaw(dyaw), s.base_quat)
        return ReferenceSample(pos, quat, s.joint_pos, s.foot_contact)


def load_reference(jump: str, path: str | Path | None = None, device: str | torch.device = "cpu") -> ReferenceTrack:
    """Load the reference of ``jump`` (from ``path``, or the final file in references/)."""
    return ReferenceTrack(JumpReference.load(str(reference_file(jump, path))), device=device, jump=jump)


# =============================================================================
# Placeholder reference (for building and testing the pipeline only)
# =============================================================================


def jump_surfaces(jump: str) -> tuple[float, float]:
    """Heights of the start and target surfaces of a jump [m] (from the scene geometry)."""
    check_jump(jump)
    if jump == "ground_to_box":
        return 0.0, C.BOX_SIZE[2]
    return C.BOX_SIZE[2], C.PLATFORM_SIZE[2]


def placeholder_reference(jump: str, dt: float = C.POLICY_DT) -> JumpReference:
    """A smooth fake jump in the JumpReference format (NOT physically planned; see module docstring).

    Nominal start (box flush against the platform, centered, square):
        ground_to_box:   base JUMP1_START_DISTANCE[0] behind the box face -> above the box center
        box_to_platform: above the box center -> middle of the platform's clear landing zone
    Feet touch the start surface at the start and the target surface at the end (via
    forward kinematics), the body stays level and faces +x.
    """
    # Imported here: standing_pose belongs to the walking part; we only use its function.
    from go2_box_climb.envs.locomotion.standing_pose import standing_joint_pos

    from .kinematics import stance_height

    s0, s1 = jump_surfaces(jump)
    if jump == "ground_to_box":
        x0 = C.BOX_TARGET_POS[0] - C.BOX_SIZE[0] / 2 - C.JUMP1_START_DISTANCE[0]
        x1 = C.BOX_TARGET_POS[0]
    else:
        x0 = C.BOX_TARGET_POS[0]
        x1 = C.PLATFORM_FRONT_X + C.PLATFORM_CLEAR_DEPTH / 2

    def pose(h: float | None) -> torch.Tensor:  # standing pose for body height h (None = default pose)
        if h is None:
            return torch.as_tensor(C.DEFAULT_JOINT_POS, dtype=torch.float64)
        return standing_joint_pos(torch.tensor([h], dtype=torch.float64))[0]

    stand, crouch, extend, absorb = pose(None), pose(0.24), pose(0.36), pose(0.28)

    def h(q: torch.Tensor) -> float:
        return float(stance_height(q.unsqueeze(0))[0])

    apex = max(s0, s1) + h(crouch) + 0.10
    # keyframes: (time [s], base x [m], base z [m], joint pose, all feet on a surface?)
    keys = [
        (0.00, x0, s0 + h(stand), stand, True),  # standing on the start surface
        (0.40, x0, s0 + h(crouch), crouch, True),  # crouch
        (0.55, x0 + 0.08, s0 + h(extend), extend, True),  # push off (take-off at the end)
        (0.80, x0 + 0.6 * (x1 - x0), apex, crouch, False),  # flight, legs tucked
        (1.00, x1, s1 + h(stand), stand, True),  # touch down on the target surface
        (1.15, x1, s1 + h(absorb), absorb, True),  # absorb the landing
        (1.60, x1, s1 + h(stand), stand, True),  # stand still
    ]
    key_t = np.array([k[0] for k in keys])
    t = np.round(np.arange(0.0, key_t[-1] + 1e-9, dt), 6)
    base_pos = np.zeros((len(t), 3))
    joint_pos = np.zeros((len(t), C.NUM_JOINTS))
    contact = np.zeros((len(t), 4), dtype=bool)
    for n, tn in enumerate(t):
        k = min(int(np.searchsorted(key_t, tn, side="right")) - 1, len(keys) - 2)
        (ta, xa, za, qa, ca), (tb, xb, zb, qb, cb) = keys[k], keys[k + 1]
        s = float((tn - ta) / (tb - ta))
        s = s * s * (3 - 2 * s)  # smooth start and stop of every segment
        base_pos[n] = (xa + s * (xb - xa), 0.0, za + s * (zb - za))
        joint_pos[n] = (qa + (qb - qa) * s).numpy()
        on_key = [i for i, kt in enumerate(key_t) if abs(kt - tn) < 1e-9]
        contact[n] = keys[on_key[0]][4] if on_key else (ca and cb)
    base_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    return JumpReference(t=t, base_pos=base_pos, base_quat=base_quat, joint_pos=joint_pos, foot_contact=contact)
