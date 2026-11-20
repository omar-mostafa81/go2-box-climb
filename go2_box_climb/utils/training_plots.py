"""Training plots from a run's TensorBoard log (Isaac-free).

rsl_rl writes every metric to a TensorBoard event file in the run folder. These
functions read it and draw PNG figures, so a run can be watched without opening
TensorBoard:

    <run>/plots/training.png   main curves: reward, episode length, losses, noise,
                               learning rate, tracking error, falls (+ push force in stage B)
    <run>/plots/rewards.png    every reward term separately

Used by scripts/locomotion/train.py (redraws every PLOT_EVERY_S seconds while
training) and by scripts/plot_training.py (any run, or several runs compared).
"""

import os
import threading
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PLOT_EVERY_S = 30.0  # rsl_rl flushes its log every 10 s

# (title, [tags], y-axis log scale?)
MAIN_PANELS = [
    ("Mean episode reward", ["Train/mean_reward"], False),
    ("Mean episode length [steps] (max 1000 = 20 s)", ["Train/mean_episode_length"], False),
    ("Value loss (critic)", ["Loss/value"], True),
    ("Surrogate loss (actor)", ["Loss/surrogate"], False),
    ("Entropy", ["Loss/entropy"], False),
    ("Action noise std", ["Policy/mean_std"], False),
    ("Learning rate", ["Loss/learning_rate"], True),
    ("Velocity tracking error", ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw"], False),
    ("Fraction of episodes ending in a fall", ["Episode_Termination/base_contact"], False),
    ("Steady push limit [N] (stage B)", ["Curriculum/push_force"], False),
]


def read_scalars(run_dir) -> dict[str, tuple[list[int], list[float]]]:
    """All scalar curves of a run: {tag: (iterations, values)}."""
    events = sorted(Path(run_dir).glob("events.out.tfevents.*"))
    if not events:
        return {}
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        pts = acc.Scalars(tag)
        out[tag] = ([p.step for p in pts], [p.value for p in pts])
    return out


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    FigureCanvasAgg(fig).print_png(str(tmp))
    os.replace(tmp, path)  # atomic: an image viewer never sees a half-written file


def _grid(n: int, cols: int = 3):
    rows = (n + cols - 1) // cols
    return Figure(figsize=(5.2 * cols, 3.4 * rows), layout="constrained"), rows, cols


def plot_runs(run_dirs, labels=None, out_dir=None) -> list[Path]:
    """Draw training.png and rewards.png for one run, or overlay several runs.

    Args:
        run_dirs: one run folder, or a list of them (curves are overlaid).
        labels: legend names (default: folder names).
        out_dir: where to save (default: <first run>/plots).
    """
    run_dirs = [Path(run_dirs)] if isinstance(run_dirs, (str, Path)) else [Path(d) for d in run_dirs]
    labels = labels or [d.name for d in run_dirs]
    data = [read_scalars(d) for d in run_dirs]
    out_dir = Path(out_dir) if out_dir else run_dirs[0] / "plots"
    multi = len(run_dirs) > 1
    title = run_dirs[0].name if not multi else "  vs  ".join(labels)

    # --- main curves ---
    panels = [p for p in MAIN_PANELS if any(t in d for d in data for t in p[1])]
    fig, rows, cols = _grid(len(panels))
    fig.suptitle(title)
    for i, (name, tags, log) in enumerate(panels):
        ax = fig.add_subplot(rows, cols, i + 1)
        for d, lab in zip(data, labels):
            for tag in tags:
                if tag in d:
                    x, y = d[tag]
                    short = tag.split("/")[-1]
                    ax.plot(x, y, lw=1.2, label=(f"{lab}: {short}" if multi else short) if (multi or len(tags) > 1) else None)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if log:
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    paths = [out_dir / "training.png"]
    _save(fig, paths[0])

    # --- every reward term ---
    reward_tags = sorted({t for d in data for t in d if t.startswith("Episode_Reward/")})
    if reward_tags:
        fig, rows, cols = _grid(len(reward_tags), cols=4)
        fig.suptitle(f"{title}: reward terms (per episode, weighted)")
        for i, tag in enumerate(reward_tags):
            ax = fig.add_subplot(rows, cols, i + 1)
            for d, lab in zip(data, labels):
                if tag in d:
                    ax.plot(*d[tag], lw=1.2, label=lab if multi else None)
            ax.set_title(tag.split("/")[-1], fontsize=9)
            ax.grid(alpha=0.3)
            if multi:
                ax.legend(fontsize=7)
        paths.append(out_dir / "rewards.png")
        _save(fig, paths[1])
    return paths


class LivePlotter:
    """Redraws a run's plots every ``every_s`` seconds in a background thread."""

    def __init__(self, run_dir, every_s: float = PLOT_EVERY_S):
        self.run_dir, self.every_s = Path(run_dir), every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.every_s):
            self.draw()

    def draw(self):
        try:
            plot_runs(self.run_dir)
        except Exception as e:  # never let plotting break training
            print(f"[WARN] Could not draw training plots: {e}")

    def stop(self):
        """Stop the thread and draw one final time."""
        self._stop.set()
        self._thread.join(timeout=5)
        self.draw()
