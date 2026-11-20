"""Training environment for the TO-guided jump policy. Part: jump from tracking.

The policy learns to jump (ground -> box, box -> platform) guided by a reference
trajectory from skills/jump/to (a JumpReference file, common/formats.py): it is
rewarded for following the reference and for landing.

Rules:
    * Train in the BASIC scene (envs/scene, ``BoxClimbSceneCfg``): plain box +
      platform, same geometry and physics as the warehouse, and it can be cloned
      into thousands of parallel robots (the warehouse supports only one).
    * The reference is computed once, offline (TO is not run at deployment). After
      training, the policy jumps on its own.
    * Physical failure checks (fall, flip, time limit, ...) use the jump limits in
      common/constants.py, the same ones as jump_from_scratch, so the comparison is
      fair. This method may add its own "too far from the reference" termination.
    * No code shared with jump_from_scratch: do not import from it.
"""
