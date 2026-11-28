"""Jump trajectory optimization: leg math, scene, reference format, and one full solve."""

import copy

import numpy as np
import pytest

from go2_box_climb.common import constants as C
from go2_box_climb.common.formats import JumpReference
from go2_box_climb.skills.jump.to import planar_model as pm
from go2_box_climb.skills.jump.to.reference import joints_12

# Made-up link lengths: these tests check the math, not the Go2's measurements.
L1, L2 = 0.2, 0.25


def test_ik_inverts_fk():
    rng = np.random.default_rng(0)
    t = rng.uniform(-1.0, 2.0, 200)
    c = rng.uniform(-2.6, -0.3, 200)  # knee bent backward, as on the Go2
    x, z = pm.leg_fk(t, c, L1, L2)
    t2, c2 = pm.leg_ik(x, z, L1, L2)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    np.testing.assert_allclose(c2, c, atol=1e-9)


def test_jacobian_matches_finite_differences():
    t, c, h = 0.7, -1.4, 1e-6
    (dx_dt, dx_dc), (dz_dt, dz_dc) = pm.leg_jacobian(t, c, L1, L2)
    x0, z0 = pm.leg_fk(t, c, L1, L2)
    xt, zt = pm.leg_fk(t + h, c, L1, L2)
    xc, zc = pm.leg_fk(t, c + h, L1, L2)
    np.testing.assert_allclose([dx_dt, dz_dt], [(xt - x0) / h, (zt - z0) / h], atol=1e-5)
    np.testing.assert_allclose([dx_dc, dz_dc], [(xc - x0) / h, (zc - z0) / h], atol=1e-5)


def test_torques_balance_the_foot_force():
    """Virtual work: motor power tau . qdot equals minus the force's power on the foot."""
    t, c, theta, fx, fz = 0.8, -1.5, 0.2, 30.0, 120.0
    tau = np.array(pm.leg_torques(t, c, theta, fx, fz, L1, L2))
    qdot = np.array([0.3, -0.7])
    (a, b), (d, e) = pm.leg_jacobian(t, c, L1, L2)
    foot_vel_body = np.array([[a, b], [d, e]]) @ qdot
    foot_vel_world = np.array(pm.body_to_world(*foot_vel_body, theta))
    assert tau @ qdot == pytest.approx(-np.array([fx, fz]) @ foot_vel_world)


def test_rotations_and_quaternion():
    theta = 0.3
    v = np.array(pm.body_to_world(1.0, 0.0, theta))
    np.testing.assert_allclose(pm.world_to_body(*v, theta), [1.0, 0.0], atol=1e-12)
    assert v[1] < 0  # positive pitch = nose down
    qx, qy, qz, qw = pm.pitch_to_quat(theta)
    assert (qx, qz) == (0.0, 0.0) and qy == pytest.approx(np.sin(theta / 2)) and qw == pytest.approx(np.cos(theta / 2))


def test_joints_12_uses_the_contract_order():
    q = C.DEFAULT_JOINT_POS
    side = np.array([[q[4], q[8], q[6], q[10]]])  # front thigh, front calf, rear thigh, rear calf
    np.testing.assert_allclose(joints_12(side)[0], q)


def test_scene_height_profile():
    """Ground, box top and platform top, with each step almost fully up at its edge."""
    from go2_box_climb.skills.jump.to.optimizer import ground_to_box_task, terrain_height

    task = ground_to_box_task()
    box_front, box_top = task.steps[0]
    platform_front, platform_top = task.steps[1]
    assert (box_front, box_top) == (1.4, C.BOX_SIZE[2]) and (platform_front, platform_top) == (2.0, C.PLATFORM_SIZE[2])
    h = lambda x: float(terrain_height(x, task, xp=np))  # noqa: E731
    assert h(1.0) == pytest.approx(0.0, abs=1e-6)
    assert h(box_front) > 0.95 * box_top  # already 95.8% up at the box edge
    assert h(1.7) == pytest.approx(box_top, abs=1e-6)
    assert h(platform_front) > box_top + 0.95 * (platform_top - box_top)


# ---------------------------------------------------------------------------------------
# Tests that need the Go2 USD (pxr; internet the first time) and one full solve (~30 s)
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robot():
    pytest.importorskip("pxr")
    from go2_box_climb.skills.jump.to.robot_params import jump_robot_params

    return jump_robot_params()


@pytest.fixture(scope="module")
def solution(robot):
    pytest.importorskip("casadi")
    from go2_box_climb.skills.jump.to.optimizer import ground_to_box_task, solve_jump

    return solve_jump(ground_to_box_task(), robot)


def test_robot_numbers_come_from_the_usd(robot):
    assert robot.mass == pytest.approx(15.019, abs=1e-3)  # sum of all 17 links in the USD
    assert robot.trunk_half_size == pytest.approx((0.3762 / 2, 0.114 / 2), abs=1e-4)  # trunk collision box
    assert robot.head_lower[2] == pytest.approx(0.047, abs=1e-4)  # lower head collision sphere
    assert robot.thigh_half_thickness == pytest.approx(0.017, abs=1e-4)
    assert abs(robot.com_offset[0]) < 0.02 and -0.04 < robot.com_offset[1] < 0.0  # COM just below the base origin
    assert 0.3 < robot.pitch_inertia < 0.7


def test_ground_to_box_jump_is_physical(robot, solution):
    from go2_box_climb.skills.jump.to.physics_check import check_jump

    sol = solution
    task = sol.task
    assert sol.status == "solved"
    # ends standing still on the box, all feet on its top surface and inside its edges
    np.testing.assert_allclose(sol.com_vel[-1], 0.0, atol=1e-6)
    np.testing.assert_allclose(sol.feet[-1, :, 1], task.target_height + robot.foot_radius, atol=1e-6)
    assert np.all((sol.feet[-1, :, 0] > task.target_x[0]) & (sol.feet[-1, :, 0] < task.target_x[1]))
    # independent check with the exact scene and robot shapes
    report = check_jump(sol, robot)
    assert report.worst_overlap <= 0.0, report.worst_overlap_part  # never inside the ground, box or platform
    assert report.torque_curve_use <= 0.9 + 1e-3, report.torque_curve_where  # inside the DC-motor curve
    assert report.max_joint_speed < C.VELOCITY_LIMIT
    assert report.friction_use <= 0.8 + 1e-3


def test_physics_check_catches_a_collision(robot, solution):
    from go2_box_climb.skills.jump.to.physics_check import check_jump

    bad = copy.deepcopy(solution)
    bad.base[:, 1] -= 0.15  # sink the whole robot 15 cm
    assert check_jump(bad, robot).worst_overlap > 0.05


def test_dc_motor_limit_matches_isaac_lab():
    from go2_box_climb.skills.jump.to.mujoco_sim import motor_torque

    big = np.full(12, 10.0)  # a target far away: the motor asks for 250 N m and gets clipped
    assert motor_torque(big, np.zeros(12), np.zeros(12))[0] == pytest.approx(C.EFFORT_LIMIT)  # standing still: 23.5
    assert motor_torque(big, np.zeros(12), np.full(12, 15.0))[0] == pytest.approx(11.75)  # half speed: half torque
    assert motor_torque(-big, np.zeros(12), np.full(12, 15.0))[0] == pytest.approx(-C.EFFORT_LIMIT)  # braking: full


@pytest.fixture(scope="module")
def mujoco_scene(robot):
    pytest.importorskip("mujoco")
    from go2_box_climb.skills.jump.to.mujoco_model import load_scene

    return load_scene()


def test_mujoco_model_comes_from_the_usd(mujoco_scene):
    import mujoco

    model, _ = mujoco_scene
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert float(model.body_mass[base:].sum()) == pytest.approx(15.019, abs=1e-3)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == list(C.JOINT_NAMES)  # motors in the contract's joint order
    calf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "FL_calf_joint")
    np.testing.assert_allclose(model.jnt_range[calf], np.radians([-156.0, -48.0]), atol=1e-3)  # USD limits, radians
    assert model.opt.timestep == C.SIM_DT


def test_mujoco_standing_matches_isaac(mujoco_scene):
    """Holding the default pose, Isaac Sim's Go2 settles at about 0.28 m (report 4.1)."""
    from go2_box_climb.skills.jump.to.mujoco_sim import stand_test

    assert stand_test(*mujoco_scene) == pytest.approx(0.28, abs=0.01)


def test_mujoco_feedback_lands_the_jump(mujoco_scene, solution):
    from go2_box_climb.skills.jump.to.mujoco_sim import simulate_jump

    result = simulate_jump(*mujoco_scene, solution, mode="feedback")
    assert result.success, result.reason


@pytest.fixture(scope="module")
def replay_scene(robot):
    pytest.importorskip("mujoco")
    from go2_box_climb.skills.jump.to.mujoco_replay import load_replay_scene

    return load_replay_scene()


def test_replay_uses_the_isaac_sim_meshes(replay_scene):
    model, _ = replay_scene
    assert model.nmesh == 33  # every link's visual mesh from the USD, split by material
    assert int(model.mesh_facenum.sum()) > 300_000  # the detailed Isaac Sim meshes, not simple shapes


def test_replayed_plan_fits_the_3d_robot(replay_scene, solution):
    """Kinematic replay with the 3D Go2: stance feet on their surfaces, nothing inside the scene."""
    from go2_box_climb.skills.jump.to.mujoco_replay import check_replay

    check = check_replay(*replay_scene, solution)
    assert check.stance_foot_error < 0.002, check.lines()  # within 2 mm (the side-view plan includes the hip tilt)
    assert check.min_clearance > 0.0, check.min_clearance_where


def test_reference_has_the_shared_format(solution):
    from go2_box_climb.skills.jump.to.reference import to_jump_reference

    ref = to_jump_reference(solution)
    assert isinstance(ref, JumpReference)
    np.testing.assert_allclose(np.diff(ref.t)[:-1], C.POLICY_DT)
    assert ref.foot_contact[0].all() and ref.foot_contact[-1].all()
    np.testing.assert_allclose(ref.joint_pos[0], C.DEFAULT_JOINT_POS, atol=1e-9)  # starts in the default pose
