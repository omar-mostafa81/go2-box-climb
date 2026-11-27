"""Write placeholder jump references, to build and test the pipeline before the real ones exist.

The real references come from the TO part (trajectory optimization + IK) into references/.
Until then this writes a smooth fake jump per jump, in the same JumpReference format, to
the git-ignored folder

    outputs/jump_from_tracking/placeholder_jump_<jump>.npz

It is NOT physically planned (see placeholder_reference in
go2_box_climb/skills/jump/jump_from_tracking/reference.py). Use it with --reference in
train.py / play.py / eval.py. Does not start Isaac Sim (the Go2 geometry is read from its
USD with pxr, like the tests).

Example (from the repository root):
    python scripts/jump_from_tracking/make_placeholder_reference.py
"""

import argparse

import torch

from go2_box_climb.common.robot_geometry import go2_geometry
from go2_box_climb.skills.jump.jump_from_tracking.kinematics import foot_positions
from go2_box_climb.skills.jump.jump_from_tracking.reference import (
    JUMPS,
    ReferenceTrack,
    jump_surfaces,
    placeholder_path,
    placeholder_reference,
)
from go2_box_climb.utils.runs import REPO_ROOT

parser = argparse.ArgumentParser(description="Write placeholder jump references to outputs/.")
parser.add_argument("--jump", choices=(*JUMPS, "all"), default="all")
args = parser.parse_args()

for jump in JUMPS if args.jump == "all" else (args.jump,):
    ref = placeholder_reference(jump)
    path = placeholder_path(jump)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref.save(str(path))

    # summary, and a check that the feet stand on the right surfaces at the start and the end
    track = ReferenceTrack(ref)
    r = go2_geometry().foot_radius
    ends = track.at(torch.tensor([0.0, track.duration]))
    feet_bottom = foot_positions(ends.base_pos, ends.base_quat, ends.joint_pos)[..., 2] - r
    s0, s1 = jump_surfaces(jump)
    airborne = (~ref.foot_contact.any(axis=1)).sum() * (ref.t[1] - ref.t[0])
    print(f"[{jump}] {path.relative_to(REPO_ROOT)}")
    print(f"  {len(ref.t)} samples, {track.duration:.2f} s, about {airborne:.2f} s with no foot on a surface")
    print(f"  base start {ref.base_pos[0].round(3).tolist()}  ->  end {ref.base_pos[-1].round(3).tolist()}")
    print(f"  feet bottom at start {feet_bottom[0].numpy().round(3).tolist()} (surface {s0} m), "
          f"at end {feet_bottom[1].numpy().round(3).tolist()} (surface {s1} m)")
