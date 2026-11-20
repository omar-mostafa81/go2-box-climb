"""Runs the trained TO-guided jump policy (Skill interface). Part: jump from tracking.

Loads the released policy from policies/, builds its observation from TaskState
exactly as during training (envs/jump_from_tracking), and reports SUCCESS / FAILURE.
No trajectory optimization at runtime.
"""
