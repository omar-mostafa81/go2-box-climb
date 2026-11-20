"""The Go2 geometry is read from the simulator's own USD (rule 13)."""

import pytest

pytest.importorskip("pxr")
pytest.importorskip("isaaclab_assets")

from go2_box_climb.common.robot_geometry import LEGS, go2_geometry  # noqa: E402


def test_go2_geometry_from_usd():
    g = go2_geometry()
    assert g.usd_url.endswith("/Robots/Unitree/Go2/go2.usd")
    assert g.thigh_length == pytest.approx(0.213, abs=1e-4)
    assert g.calf_length == pytest.approx(0.213, abs=1e-4)
    assert g.foot_radius == pytest.approx(0.022, abs=1e-4)
    # hips: front legs ahead of the body center, left legs on the +y side
    for leg in LEGS:
        x, y, _ = g.hip_joint_pos[leg]
        assert (x > 0) == leg.startswith("F")
        assert (y > 0) == leg.endswith("L")
    # the lower head is ahead of the front hips (it is what touches the box)
    assert g.head_lower_pos[0] > g.hip_joint_pos["FL"][0]
