"""Trajectory optimization (CasADi) + inverse kinematics for the jumps. Part: TO.

Runs ONCE, offline, per jump (ground -> box, box -> platform) for the nominal box
placement, and writes a JumpReference file (common/formats.py) to references/.
It is not run at deployment: the TO-guided policy (jump_from_tracking) only uses the
reference during training.

Robot geometry (link lengths, joint offsets, foot radius) comes from
common/robot_geometry.py, read from the simulator's own Go2 USD (CLAUDE.md rule 13).
Keep the math in plain functions so it can be tested without the simulator.
"""
