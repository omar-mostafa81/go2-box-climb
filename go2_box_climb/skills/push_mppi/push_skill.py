"""The push skill: MPPI plans walking commands that push the box flush against the platform.

Per policy step (POLICY_DT = 0.02 s):
    every ``replan_every`` steps: state -> planar state -> MPPI -> (vx, vy, wz)
    body height h from a fixed rule (low near the push point, so the head hits the box face)
    (vx, vy, wz, h) -> WalkSkill (the walking policy) -> 12 joint actions
Once the box has settled against the platform, the robot is commanded to stand still.

End of the skill (``status``; the reason is kept in ``end_reason``):
    fell            the robot fell
    success         box settled flush (gap < 1 cm, speed < 1 cm/s for 1 s), y and yaw within tolerance
    box_off_center  box settled flush, but y or yaw outside the tolerance
    no_contact      the head never touched the box within 15 s
    timeout         none of the above within 30 s

The box is square and any side may be pushed and end flush (see cost.py, model.py):
    - box yaw errors are taken mod 90 deg; contact uses the side nearest to the head
    - the box only counts as settled (so success / box_off_center can only happen) after the
      robot has pushed from the push point: head within push_reach of it, and the box moved
      push_moved since then. A box that starts flush is therefore never "done" before the robot acts.
"""

from dataclasses import dataclass, field

import numpy as np

from go2_box_climb.common import constants as C
from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.common.skill import Skill, SkillStatus
from go2_box_climb.common.state import TaskState
from go2_box_climb.skills.locomotion import DEFAULT_BODY_HEIGHT, PUSH_BODY_HEIGHT, WalkSkill, has_fallen

from .cost import GOAL, push_point
from .model import face_gap, head_point, wrap_yaw90
from .mppi import MPPIConfig, MPPIPlanner


@dataclass
class PushConfig:
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    # Head contact point = head_lower frame x (from the Go2 USD) + this margin. The USD gives
    # the frame, not the head's collision size, so the margin is tuned in simulation
    # (0.05 m: the real head touches the box ~5 cm before the frame; 0 m left the box 4-6 cm off center).
    contact_margin: float = 0.05  # [m]
    # Settled: box front within settle_gap of the platform and slower than settle_speed for settle_time.
    settle_gap: float = 0.01  # [m]
    settle_speed: float = 0.01  # [m/s]
    settle_time: float = 1.0  # [s]
    # Success tolerances: provisional = the box placements the jump policies are trained on
    # (common/constants.py), until the box-placement study gives real values. After settling,
    # only y and yaw decide success; tol_gap is used for reporting (see placement_within).
    tol_gap: float = C.BOX_GAP_RANGE[1]  # box front face to platform face [m]
    tol_y: float = C.BOX_LATERAL_RANGE[1]  # [m]
    tol_yaw: float = C.BOX_YAW_RANGE[1]  # [rad]
    # "Tight" levels: the target the cost weights are tuned for (not used by status()).
    tight_gap: float = 0.02  # [m]
    tight_y: float = 0.03  # [m]
    tight_yaw: float = np.deg2rad(3.0)  # [rad]
    # Failure checks
    contact_gap: float = 0.01  # head within this of the nearest box side counts as contact [m]
    contact_timeout: float = 15.0  # no contact by then -> no_contact [s]
    timeout: float = 30.0  # [s]
    # Push point rules (see the module docstring)
    push_low: float = 0.80  # crouch (PUSH_BODY_HEIGHT) with the head within this of the push point [m]
    push_reach: float = 0.05  # head within this of the push point = at the push point [m]
    push_moved: float = 0.02  # box moved this far since then = "pushed from the push point" [m]

    @property
    def replan_every(self) -> int:
        """Policy steps between replans (MPPI dt / POLICY_DT, e.g. 0.1 / 0.02 = 5)."""
        return max(1, round(self.mppi.dt / C.POLICY_DT))


def yaw_of(q_xyzw: np.ndarray) -> float:
    """Heading (rotation about z) of a quaternion (x, y, z, w)."""
    x, y, z, w = q_xyzw
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def planar_state(state: TaskState) -> np.ndarray:
    """TaskState -> [rx, ry, ryaw, bx, by, byaw] used by the planar model."""
    return np.array(
        [*state.base_pos[:2], yaw_of(state.base_quat), *state.box_pos[:2], yaw_of(state.box_quat)], dtype=float
    )


def box_errors_planar(x: np.ndarray) -> tuple[float, float, float]:
    """(gap, y, yaw) of the box relative to the goal, from a planar state (6,).

    gap: platform face minus the box's front-most point (yaw-aware) [m]; y: sideways offset
    from the goal [m]; yaw: box yaw mod 90 deg, in [-45, 45) deg [rad].
    """
    byaw = float(wrap_yaw90(x[5]))
    front = x[3] + C.BOX_SIZE[0] / 2 * abs(np.cos(byaw)) + C.BOX_SIZE[1] / 2 * abs(np.sin(byaw))
    return float(C.PLATFORM_FRONT_X - front), float(x[4] - GOAL[1]), byaw


def box_errors(state: TaskState) -> tuple[float, float, float]:
    """box_errors_planar for a TaskState."""
    return box_errors_planar(planar_state(state))


def placement_within(state: TaskState, gap: float, y: float, yaw: float) -> bool:
    """Box gap, |y| and |yaw| all within the given limits (e.g. cfg.tol_* or cfg.tight_*)."""
    g, dy, dyaw = box_errors(state)
    return g <= gap and abs(dy) <= y and abs(dyaw) <= yaw


def push_point_progress(x: np.ndarray, p) -> tuple[bool, float]:
    """(push point valid, head distance to it [m]) for one planar state (6,).
    Not valid = box within 2 cm of the goal (no push point; distance is then inf)."""
    pp, _, valid = push_point(x[None], p)
    return bool(valid[0]), float(np.linalg.norm(head_point(x[None], p.head_x)[0] - pp[0])) if valid[0] else np.inf


class PushLatch:
    """Has the robot pushed from the push point? (head within push_reach of it, then the box moved
    push_moved). A box already at the goal (no push point) counts as pushed."""

    def __init__(self, cfg: PushConfig):
        self.cfg, self.start, self.pushed = cfg, None, False

    def update(self, x: np.ndarray, p) -> bool:
        valid, dist = push_point_progress(x, p)
        if not valid:
            self.pushed = True
        if self.start is None and dist <= self.cfg.push_reach:
            self.start = x[3:5].copy()
        if self.start is not None and np.linalg.norm(x[3:5] - self.start) >= self.cfg.push_moved:
            self.pushed = True
        return self.pushed


class PushMPPISkill(Skill):
    name = "push"

    def __init__(self, walk: WalkSkill | None = None, config: PushConfig | None = None, head_x: float | None = None):
        """``walk``: the walking skill to drive (default: the released walking policy).
        ``head_x``: head contact point ahead of the base [m]; default from the Go2 USD (rule 13)."""
        self.cfg = config or PushConfig()
        self.walk = walk or WalkSkill()
        if head_x is None:
            head_x = go2_geometry().head_lower_pos[0] + self.cfg.contact_margin
        self.planner = MPPIPlanner(head_x, self.cfg.mppi)
        self.command = np.zeros(4)  # last (vx, vy, wz, h) sent to the walking policy
        self._steps = 0
        self._low = False
        self._settle_steps = 0  # consecutive policy steps with the box flush and still
        self._contact = False  # the head has touched the box
        self._latch = PushLatch(self.cfg)  # pushed from the push point yet?
        self.end_reason: str | None = None  # set by status() once the skill ends

    @property
    def last_cost_terms(self) -> dict[str, float]:
        """Cost breakdown of the plan being followed (for logging and tuning)."""
        return self.planner.last_terms

    def reset(self, state: TaskState) -> None:
        self.walk.reset(state)
        self.planner.reset()
        self.command = np.array([0.0, 0.0, 0.0, DEFAULT_BODY_HEIGHT])
        self._steps = 0
        self._low = False
        self._settle_steps = 0
        self._contact = False
        self._latch = PushLatch(self.cfg)
        self.end_reason = None

    @property
    def settled(self) -> bool:
        """Box flush against the platform and still for settle_time."""
        return self._settle_steps * C.POLICY_DT >= self.cfg.settle_time

    def body_height(self, x: np.ndarray) -> float:
        """Fixed height rule: PUSH_BODY_HEIGHT with the head within push_low of the push point,
        DEFAULT_BODY_HEIGHT otherwise. No push point (box at the goal): keep the current height."""
        valid, dist = push_point_progress(x, self.planner.params)
        if valid:
            self._low = dist < self.cfg.push_low
        return PUSH_BODY_HEIGHT if self._low else DEFAULT_BODY_HEIGHT

    def act(self, state: TaskState) -> np.ndarray:
        x = planar_state(state)
        h = self.body_height(x)
        # bookkeeping for status(): contact and settling
        head_gap, _ = face_gap(x[None], self.planner.params)
        self._contact |= bool(head_gap[0] <= self.cfg.contact_gap)
        gap, _, _ = box_errors(state)
        still = np.linalg.norm(state.box_lin_vel[:2]) < self.cfg.settle_speed
        flush = gap < self.cfg.settle_gap and self._latch.update(x, self.planner.params)
        self._settle_steps = self._settle_steps + 1 if (flush and still) else 0
        if self.settled:
            vel = np.zeros(3)  # done: stand still
        elif self._steps % self.cfg.replan_every == 0:
            vel = self.planner.plan(x)
        else:
            vel = self.command[:3]  # hold the command between replans
        self.command = self.walk.set_command(*vel, h)
        self._steps += 1
        return self.walk.act(state)

    def status(self, state: TaskState) -> SkillStatus:
        self.end_reason = self._end_reason(state)
        if self.end_reason is None:
            return SkillStatus.RUNNING
        return SkillStatus.SUCCESS if self.end_reason == "success" else SkillStatus.FAILURE

    def _end_reason(self, state: TaskState) -> str | None:
        t = self._steps * C.POLICY_DT
        if has_fallen(state):
            return "fell"
        if self.settled:
            # flush and still: only y and yaw decide success
            _, dy, dyaw = box_errors(state)
            return "success" if abs(dy) <= self.cfg.tol_y and abs(dyaw) <= self.cfg.tol_yaw else "box_off_center"
        if not self._contact and t >= self.cfg.contact_timeout:
            return "no_contact"
        if t > self.cfg.timeout:
            return "timeout"
        return None
