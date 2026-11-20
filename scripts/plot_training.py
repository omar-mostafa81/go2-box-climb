"""Draw training plots for any run (no simulator needed).

Examples (from the repository root):
    # One run -> <run>/plots/training.png and rewards.png
    python scripts/plot_training.py logs/rsl_rl/go2_walk/<run>

    # Keep redrawing while that run trains (Ctrl+C to stop)
    python scripts/plot_training.py logs/rsl_rl/go2_walk/<run> --watch

    # Compare runs on the same axes
    python scripts/plot_training.py logs/rsl_rl/go2_walk/<run1> logs/rsl_rl/go2_walk/<run2> \
        --labels "before" "after" --out logs/plots/compare_a
"""

import argparse
import time

from go2_box_climb.utils.training_plots import PLOT_EVERY_S, plot_runs

parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs.")
parser.add_argument("runs", nargs="+", help="Run folder(s).")
parser.add_argument("--labels", nargs="+", default=None, help="Legend names, one per run.")
parser.add_argument("--out", default=None, help="Output folder (default: <first run>/plots).")
parser.add_argument("--watch", action="store_true", help=f"Redraw every {PLOT_EVERY_S:.0f} s until Ctrl+C.")
args = parser.parse_args()

while True:
    for p in plot_runs(args.runs, labels=args.labels, out_dir=args.out):
        print(f"[INFO] Saved {p}")
    if not args.watch:
        break
    try:
        time.sleep(PLOT_EVERY_S)
    except KeyboardInterrupt:
        break
