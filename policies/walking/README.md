# Walking policy (released)

`policy.pt` — TorchScript actor, 49 inputs → 12 joint actions, 50 Hz.
Default policy of `go2_box_climb.skills.locomotion` (`WalkingPolicy`, `WalkSkill`).

## Use

```python
from go2_box_climb.skills.locomotion import WalkSkill, PUSH_BODY_HEIGHT

walk = WalkSkill()                    # loads policies/walking/policy.pt
walk.reset(state)
walk.set_command(vx, vy, wz)          # walking: body height 0.32 m (default)
walk.set_command(0.3, 0.0, 0.0, PUSH_BODY_HEIGHT)   # pushing the box: 0.27 m
action = walk.act(state)              # joint targets = DEFAULT_JOINT_POS + ACTION_SCALE * action
```

Command ranges: vx ±1.0 m/s, vy ±0.6 m/s, wz ±1.0 rad/s, body height 0.25–0.34 m.
Observation layout and frames: `go2_box_climb/skills/locomotion/walking_policy.py`.

## Training

Two PPO stages (rsl_rl, 4096 robots, flat ground, no boxes), `go2_box_climb/envs/locomotion/`:

| Stage | Run | Iterations | Adds |
| --- | --- | --- | --- |
| A | `go2_walk/2026-11-17_04-53-24` (`model_999.pt`) | 1000 | velocity tracking with trot-shaping rewards |
| B | `go2_walk_push/2026-11-17_05-04-18` (`model_2498.pt`) | 1500 (from A) | body-height command, steady push on the head up to ±40 N, standing still at any height |

Reproduce:

```bash
python scripts/locomotion/train.py --stage a --mode headless
python scripts/locomotion/train.py --stage b --mode headless --load logs/rsl_rl/go2_walk/<stage A run>/model_999.pt
```

## Evaluation (`scripts/locomotion/eval.py --stage b`, 64 robots, 6 s per case)

| Command | Actual | Body height |
| --- | --- | --- |
| forward 0.3 / 0.6 / 1.0 m/s | 0.30 / 0.61 / 1.00 m/s | 0.32 m |
| backward 0.5 m/s | 0.49 m/s | 0.32 m |
| sideways 0.4 m/s | 0.40 m/s | 0.32 m |
| turn 0.8 rad/s (+0.3 m/s) | 0.79 rad/s | 0.32 m |
| forward 0.3 m/s at h 0.27 m | 0.30 m/s | 0.27 m |

- No falls. Trot with about 2 cm foot clearance.
- Standing still (from rest or right after walking, at 0.25–0.32 m): feet planted
  (≤0.1% of the time off the ground), joints still.
