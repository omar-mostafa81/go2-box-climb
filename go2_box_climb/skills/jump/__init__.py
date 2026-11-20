"""Jump skills (ground -> box, box -> platform).

    to/                  trajectory optimization + IK -> JumpReference files. Part: TO.
    jump_from_tracking/  runs the trained TO-guided jump policy. Part: jump from tracking.
    jump_from_scratch/   runs the trained from-scratch jump policy. Part: jump from scratch.

Both jump skills implement the same Skill interface (common/skill.py), so the
sequencer can use either one. They are trained in envs/jump_from_tracking and
envs/jump_from_scratch, and compared by the jump evaluation in eval/.
"""
