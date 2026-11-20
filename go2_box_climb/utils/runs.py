"""Where training runs are stored (Isaac-free).

Every training run writes to   <repo>/logs/rsl_rl/<experiment>/<YYYY-MM-DD_HH-MM-SS>[_<name>]/
with checkpoints  model_<iteration>.pt  inside, no matter which folder the script is
started from. ``logs/`` is ignored by git; only released policies in ``policies/``
are committed.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_LOGS = REPO_ROOT / "logs" / "rsl_rl"
