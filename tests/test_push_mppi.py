"""Fast tests of the MPPI push skill (planar model, cost, push point, planner, skill). No simulator."""

import numpy as np
import pytest

from go2_box_climb.common import constants as C
from go2_box_climb.common.skill import SkillStatus
from go2_box_climb.common.state import TaskState
from go2_box_climb.skills.locomotion import DEFAULT_BODY_HEIGHT, PUSH_BODY_HEIGHT, WalkSkill
from go2_box_climb.skills.push_mppi import (
    GOAL,
    CostWeights,
    MPPIConfig,
    MPPIPlanner,
    PlanarParams,
    PushConfig,
    PushMPPISkill,
)
from go2_box_climb.skills.push_mppi.cost import push_point, running_cost
from go2_box_climb.skills.push_mppi.model import face_gap, nearest_side, step, wrap_yaw90
from go2_box_climb.skills.push_mppi.push_skill import PushLatch

HEAD_X = 0.3  # a plausible head offset; the skill reads the real one from the Go2 USD
P = PlanarParams(dt=0.1, head_x=HEAD_X)
BOX = (C.BOX_START_POS[0], 0.0, 0.0)
HALF = C.BOX_SIZE[0] / 2
SIDE_OUT = {0: 0.0, 1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}  # outward normal angle of each side (box frame)


def _touching(side=2, lateral=0.0, box=BOX, depth=0.0):
    """Robot facing into `side` (0 front, 1 left, 2 back, 3 right), head `depth` m inside it,
    `lateral` m to the left of the push direction. Returns a planar state (1, 6)."""
    bx, by, byaw = box
    out = byaw + SIDE_OUT[side]
    n = np.array([np.cos(out), np.sin(out)])
    left = np.array([n[1], -n[0]])  # left of the push direction -n
    head = np.array([bx, by]) + (HALF - depth) * n + lateral * left
    ryaw = np.arctan2(np.sin(out + np.pi), np.cos(out + np.pi))
    base = head - HEAD_X * np.array([np.cos(ryaw), np.sin(ryaw)])
    return np.array([[*base, ryaw, bx, by, byaw]])


def _forward(v=0.3):
    return np.array([[v, 0.0, 0.0]])


# ---------------- model ----------------
@pytest.mark.parametrize("deg, expected", [(0, 0), (90, 0), (-90, 0), (180, 0), (44, 44), (46, -44), (-46, 44)])
def test_yaw_wrap(deg, expected):
    assert np.isclose(np.degrees(wrap_yaw90(np.deg2rad(deg))), expected, atol=1e-9)


def test_box_still_without_contact():
    x = np.array([[0.0, 0.0, 0.0, *BOX]])
    x1, info = step(x, _forward(), P)
    assert np.allclose(x1[0, 3:], x[0, 3:])
    assert not info["contact"][0]
    assert np.isclose(x1[0, 0], 0.03)


@pytest.mark.parametrize("side, direction", [(0, (-1, 0)), (1, (0, -1)), (2, (1, 0)), (3, (0, 1))])
def test_push_each_side_moves_box_inward(side, direction):
    x1, info = step(_touching(side), _forward(0.3), P)
    assert info["contact"][0] and nearest_side(_touching(side, depth=0.01), P)["side"][0] == side
    assert np.allclose(x1[0, 3:5] - BOX[:2], 0.03 * np.array(direction), atol=1e-9)
    assert np.isclose(x1[0, 5], 0.0)  # centered push: no rotation


@pytest.mark.parametrize("side", [0, 1, 2, 3])
def test_off_center_push_rotation_sign(side):
    # pushing left of the push direction turns the box clockwise (negative yaw), and vice versa
    xl, _ = step(_touching(side, lateral=+0.2), _forward(), P)
    xr, _ = step(_touching(side, lateral=-0.2), _forward(), P)
    assert xl[0, 5] < 0 < xr[0, 5]


@pytest.mark.parametrize("side", [0, 1, 2, 3])
def test_no_pass_through_any_side(side):
    x = _touching(side, box=(1.0, 0.0, np.deg2rad(20.0)))
    for _ in range(20):
        x, info = step(x, _forward(0.4), P)
        assert info["gap"][0] >= -1e-9  # head never left inside the box


def test_box_stops_at_wall():
    x = _touching(box=(GOAL[0] - 0.01, 0.0, 0.0))
    for _ in range(5):
        x, _ = step(x, _forward(0.4), P)
    assert np.isclose(x[0, 3], GOAL[0])  # flush, not past the platform face
    assert face_gap(x, P)[0][0] >= -1e-9  # head not inside the box


@pytest.mark.parametrize("start_deg, square_deg", [(5.0, 0.0), (85.0, 90.0)])
def test_wall_squares_to_nearest_90(start_deg, square_deg):
    x = _touching(box=(GOAL[0] - 0.01, 0.0, np.deg2rad(start_deg)))
    for _ in range(10):
        x, _ = step(x, _forward(0.3), P)
    assert abs(np.degrees(x[0, 5]) - square_deg) < abs(start_deg - square_deg)


# ---------------- push point ----------------
def _pp(box):
    pp, d, valid = push_point(np.array([[0.0, 0.0, 0.0, *box]]), P)
    return pp[0], d[0], valid[0]


def test_push_point_normal_start_is_back_center():
    pp, d, valid = _pp((1.0, 0.0, 0.0))
    assert valid and np.allclose(pp, (0.7, 0.0)) and np.allclose(d, (1.0, 0.0))


def test_push_point_flush_off_center_is_left_side():
    pp, d, valid = _pp((GOAL[0], 0.10, 0.0))
    assert valid and np.allclose(d, (0.0, -1.0)) and np.allclose(pp, (GOAL[0], 0.40))


def test_push_point_rotated_45_is_back_corner():
    pp, _, valid = _pp((1.0, 0.0, np.deg2rad(45.0)))
    assert valid and np.allclose(pp, (1.0 - 0.3 * np.sqrt(2), 0.0))


def test_box_at_goal_has_no_push_point():
    _, _, valid = _pp((GOAL[0], 0.0, 0.0))
    assert not valid
    _, terms = _cost(np.array([[0.5, 0.3, 1.0, GOAL[0], 0.0, 0.0]]))
    assert terms["approach"][0] == 0 and terms["heading"][0] == 0


# ---------------- cost ----------------
def _cost(x, u=None, x_prev=None):
    u = np.zeros((len(x), 3)) if u is None else u
    x_prev = x if x_prev is None else x_prev
    _, info = step(x, np.zeros((len(x), 3)), PlanarParams(dt=0.0, head_x=HEAD_X))  # info without moving
    return running_cost(x, x_prev, u, u, info, P, CostWeights())


def test_terms_sum_to_total_and_are_nonnegative():
    x = np.array([[0.0, 0.1, 0.2, 1.2, -0.1, 0.1], [0.5, 0.0, 0.0, 1.5, 0.0, 0.0]])
    total, terms = _cost(x, u=np.array([[0.3, 0.1, 0.2], [0.0, 0.0, 0.0]]))
    assert np.allclose(total, sum(terms.values()))
    assert all((v >= 0).all() for v in terms.values())


def test_approach_and_heading_zero_at_push_point():
    _, terms = _cost(_touching(2))  # head on the back side center facing +x: the normal-start push point
    assert terms["approach"][0] < 1e-9 and terms["heading"][0] < 1e-9


def test_cost_is_symmetric_under_90_deg_box_rotation():
    # a square box turned by 90 deg is the same box: same push point, same wrapped yaw, same cost
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.uniform(-0.2, 0.8, 50), rng.uniform(-0.8, 0.8, 50), rng.uniform(-np.pi, np.pi, 50),
                         rng.uniform(0.9, 1.6, 50), rng.uniform(-0.3, 0.3, 50), rng.uniform(-0.5, 0.5, 50)])  # fmt: skip
    u = rng.uniform(-0.3, 0.3, (50, 3))
    total, terms = _cost(x, u)
    for k in (1, 2, 3):
        xr = x.copy()
        xr[:, 5] += k * np.pi / 2
        total_r, terms_r = _cost(xr, u)
        for name in terms:
            assert np.allclose(terms[name], terms_r[name]), name
        assert np.allclose(total, total_r)


def test_platform_penalty_only_past_face():
    w = CostWeights()
    at = np.array([[C.PLATFORM_FRONT_X - HEAD_X + 0.01, 0.0, 0.0, GOAL[0], 0.0, 0.0]])
    before = at.copy()
    before[0, 0] -= 0.05
    assert _cost(at)[1]["platform"][0] == w.platform
    assert _cost(before)[1]["platform"][0] == 0


def test_forward_effort_not_dearer_than_sideways():
    _, terms = _cost(np.zeros((2, 6)), u=np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0]]))
    assert terms["effort"][0] <= terms["effort"][1]


def test_box_speed_only_near_goal_in_xy():
    # flush (x at the goal) but 10 cm off center, sliding sideways: not "near the goal", no box_speed cost
    x_prev = np.array([[1.0, 1.0, 0.0, GOAL[0], 0.10, 0.0]])
    x = x_prev.copy()
    x[0, 4] -= 0.02
    assert _cost(x, x_prev=x_prev)[1]["box_speed"][0] == 0
    at = np.array([[1.0, 1.0, 0.0, GOAL[0], 0.0, 0.0]])
    assert _cost(at, x_prev=at - np.array([[0, 0, 0, 0.005, 0, 0]]))[1]["box_speed"][0] > 0


# ---------------- planner ----------------
def test_commands_stay_in_limits_and_plan_shifts():
    cfg = MPPIConfig(num_samples=64, horizon=10)
    pl = MPPIPlanner(HEAD_X, cfg)
    lo, hi = cfg.bounds()
    u = pl.plan(np.array([0.0, 0.0, 0.0, *BOX]))
    assert (u >= lo - 1e-12).all() and (u <= hi + 1e-12).all()
    assert hi[0] <= 0.4  # slow push
    assert np.allclose(pl.U[-1], pl.U[-2])  # warm start: last step repeated after the shift
    assert set(pl.last_terms) >= {"box_pos", "approach", "terminal"}


def test_closed_loop_pushes_box_to_goal():
    pl = MPPIPlanner(HEAD_X, MPPIConfig(num_samples=256))
    x = np.array([[0.0, 0.0, 0.0, *BOX]])
    for _ in range(60):  # 6 s
        x, _ = step(x, pl.plan(x[0])[None], P)
    assert abs(x[0, 3] - GOAL[0]) < 0.02
    assert abs(x[0, 4]) < 0.03
    assert abs(wrap_yaw90(x[0, 5])) < np.deg2rad(3.0)


def test_closed_loop_rotated_90_matches_normal():
    # a square box turned 90 deg is the same box: the planar push should be identical
    def run(byaw):
        pl = MPPIPlanner(HEAD_X, MPPIConfig(num_samples=128))
        x = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, byaw]])
        for _ in range(40):
            x, _ = step(x, pl.plan(x[0])[None], P)
        return x[0]

    a, b = run(0.0), run(np.pi / 2)
    assert np.allclose(a[:5], b[:5], atol=1e-6)


# ---------------- skill ----------------
def _yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def _state(robot=(0.0, 0.0, 0.0), box=BOX, box_vel=(0.0, 0.0), base_z=0.32):
    return TaskState(
        base_pos=np.array([robot[0], robot[1], base_z]),
        base_quat=_yaw_quat(robot[2]),
        joint_pos=C.DEFAULT_JOINT_POS.copy(),
        box_pos=np.array([box[0], box[1], C.BOX_SIZE[2] / 2]),
        box_quat=_yaw_quat(box[2]),
        box_lin_vel=np.array([*box_vel, 0.0]),
    )


def _state_from(x, box_vel=(0.0, 0.0)):
    """TaskState from a planar state (1, 6) or (6,)."""
    x = np.asarray(x).reshape(-1)
    return _state(robot=tuple(x[:3]), box=tuple(x[3:6]), box_vel=box_vel)


@pytest.fixture(scope="module")
def walk():
    return WalkSkill()  # the released walking policy (policies/walking/policy.pt)


def _skill(walk, **cfg):
    return PushMPPISkill(walk=walk, config=PushConfig(mppi=MPPIConfig(num_samples=64), **cfg), head_x=HEAD_X)


def _run(sk, s, seconds):
    """Call act() for this long on a fixed state; return the status after."""
    for _ in range(round(seconds / C.POLICY_DT)):
        sk.act(s)
    return sk.status(s)


def test_body_height_rule(walk):
    sk = _skill(walk)
    sk.reset(_state())
    push_x = BOX[0] - HALF  # push point of the normal start (back side center)
    sk.act(_state(robot=(push_x - HEAD_X - sk.cfg.push_low - 0.1, 0.0, 0.0)))  # far from the push point: tall
    assert sk.command[3] == DEFAULT_BODY_HEIGHT
    sk.act(_state(robot=(push_x - HEAD_X - sk.cfg.push_low + 0.1, 0.0, 0.0)))  # within push_low: crouch
    assert sk.command[3] == PUSH_BODY_HEIGHT


def test_act_returns_12_actions_and_replans_every_0p1s(walk):
    sk = _skill(walk)
    s = _state()
    sk.reset(s)
    a = sk.act(s)
    assert a.shape == (C.NUM_JOINTS,)
    assert sk.cfg.replan_every == 5
    first = sk.command.copy()
    for _ in range(4):
        sk.act(s)
        assert np.allclose(sk.command, first)  # held between replans


def test_latch_needs_push_from_push_point():
    latch = PushLatch(PushConfig())
    box = (GOAL[0], 0.10, 0.0)  # scenario D: flush, 10 cm off
    assert not latch.update(np.array([0.0, 0.0, 0.0, *box]), P)  # robot far away: not pushed
    at_pp = _touching(1, box=box)[0]  # head on the +y side center = the push point
    assert not latch.update(at_pp, P)  # at the push point, box not moved yet
    moved = at_pp.copy()
    moved[4] -= 0.03
    assert latch.update(moved, P)  # box moved 3 cm since: pushed


def test_settled_at_goal_is_success(walk):
    # box at the goal (no push point): settles after settle_time and succeeds; the robot stands still
    s = _state_from(_touching(box=(GOAL[0], 0.0, 0.0)))
    sk = _skill(walk)
    sk.reset(s)
    assert _run(sk, s, 0.9) == SkillStatus.RUNNING  # not settled for 1 s yet
    assert _run(sk, s, 0.2) == SkillStatus.SUCCESS
    assert sk.end_reason == "success"
    assert np.allclose(sk.command[:3], 0.0)


def test_moving_box_does_not_settle(walk):
    s = _state_from(_touching(box=(GOAL[0], 0.0, 0.0)), box_vel=(0.05, 0.0))
    sk = _skill(walk)
    sk.reset(s)
    assert _run(sk, s, 1.5) == SkillStatus.RUNNING


def test_flush_off_center_box_not_done_before_a_push(walk):
    # scenario D: box starts flush and still, 10 cm off; without a push from the push point the run goes on
    s = _state(box=(GOAL[0], 0.10, 0.0))
    sk = _skill(walk)
    sk.reset(s)
    assert _run(sk, s, 2.0) == SkillStatus.RUNNING


def test_settled_off_center_after_push_fails(walk):
    sk = _skill(walk, tol_y=0.05)
    box0 = (GOAL[0], 0.10, 0.0)
    at_pp = _touching(1, box=box0)[0]  # head at the +y push point
    sk.reset(_state_from(at_pp))
    sk.act(_state_from(at_pp))
    pushed = at_pp.copy()
    pushed[4] -= 0.02  # the box moved 2 cm toward the goal, then stopped 8 cm off center
    assert _run(sk, _state_from(pushed), 1.1) == SkillStatus.FAILURE
    assert sk.end_reason == "box_off_center"


def test_no_contact_and_timeout(walk):
    sk = _skill(walk, contact_timeout=0.5, timeout=1.0)
    sk.reset(_state())
    assert _run(sk, _state(), 0.6) == SkillStatus.FAILURE  # robot never reaches the box
    assert sk.end_reason == "no_contact"
    sk = _skill(walk, contact_timeout=0.5, timeout=1.0)
    s = _state_from(_touching())  # touching, box far from the platform
    sk.reset(s)
    assert _run(sk, s, 0.6) == SkillStatus.RUNNING
    assert _run(sk, s, 0.5) == SkillStatus.FAILURE
    assert sk.end_reason == "timeout"


def test_fall_ends_skill(walk):
    sk = _skill(walk)
    sk.reset(_state())
    assert sk.status(_state(base_z=0.1)) == SkillStatus.FAILURE
    assert sk.end_reason == "fell"


def test_default_config():
    cfg = PushConfig()
    assert (cfg.tol_gap, cfg.tol_y, cfg.tol_yaw) == (C.BOX_GAP_RANGE[1], C.BOX_LATERAL_RANGE[1], C.BOX_YAW_RANGE[1])
    assert (cfg.tight_gap, cfg.tight_y, cfg.tight_yaw) == (0.02, 0.03, np.deg2rad(3.0))
    assert (cfg.contact_timeout, cfg.timeout) == (15.0, 30.0)
