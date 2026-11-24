"""Named box start poses for testing the push skill (run_push.py --scenario).

Test scenarios only: the skill itself does not know about them.
Box center x [m], y [m], yaw [deg]; the robot always starts at its default start pose.
"""

SCENARIOS = {
    "A": (1.0, 0.0, 0.0),  # normal start
    "B": (1.4, 0.0, 60.0),  # rotated 60 deg
    "C": (1.0, 0.0, 45.0),  # rotated 45 deg 
    "D": (1.3, 0.10, 0.0),  # already flush against the platform, 10 cm off center
    "E": (1.1, 2.40, 0.0),  # 

}
