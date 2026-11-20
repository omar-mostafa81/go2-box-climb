"""Training environment for the jump policy learned from scratch. Part: jump from scratch.

The policy learns to jump (ground -> box, box -> platform) with RL only, without a
reference trajectory. It is the baseline for the TO-guided policy.

Rules:
    * Train in the BASIC scene (envs/scene, ``BoxClimbSceneCfg``): plain box +
      platform, same geometry and physics as the warehouse, and it can be cloned
      into thousands of parallel robots (the warehouse supports only one).
    * Physical failure checks (fall, flip, time limit, ...) use the jump limits in
      common/constants.py, the same ones as jump_from_tracking, so the comparison is
      fair.
    * No code shared with jump_from_tracking: do not import from it.
"""
