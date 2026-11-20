# Jump reference trajectories

Output of trajectory optimization + IK (`go2_box_climb/skills/jump/to/`), used as
training input by the TO-guided jump policy (`go2_box_climb/envs/jump_from_tracking/`).

- One file per jump, computed **once, offline**, for the nominal box placement
  (box flush against the platform, centered, square):
  - `jump_ground_to_box.npz`: ground → box
  - `jump_box_to_platform.npz`: box → platform
- Format: `JumpReference` (`go2_box_climb/common/formats.py`): time stamps, base
  position and orientation, 12 joint angles and foot contacts on a time grid.
  Load with `JumpReference.load(REFERENCE_FILES["ground_to_box"])`.
- Only the final references are committed here (like `policies/`). Drafts and
  intermediate TO runs go to an ignored folder such as `outputs/`.
- Owner: the TO member. Changing a reference changes what the tracking policy is
  trained on, so tell the jump-from-tracking member.
