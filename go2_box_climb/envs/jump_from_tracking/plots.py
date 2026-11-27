"""Jump-specific training curves (Isaac-free): how the attempts end and how well the reference is followed.

Draws <run>/plots/jump.png, next to the general training.png and rewards.png that
go2_box_climb/utils/training_plots.py draws for every run. ``JumpLivePlotter`` redraws
all of them while training runs.
"""

from pathlib import Path

from matplotlib.figure import Figure

from go2_box_climb.utils.training_plots import LivePlotter, read_scalars

# (title, [TensorBoard tags])
JUMP_PANELS = [
    (
        "How attempts end (fraction of episodes)",
        [
            "Episode_Termination/success",
            "Episode_Termination/fall_contact",
            "Episode_Termination/fall_tilt",
            "Episode_Termination/time_out",
        ],
    ),
    ("Joint tracking error [rad] (per step, mean over the attempt)", ["Metrics/jump_reference/error_joint_pos"]),
    ("Base position tracking error [m]", ["Metrics/jump_reference/error_base_pos"]),
    ("Mean episode length [steps] (max 150 = 3 s)", ["Train/mean_episode_length"]),
]


def plot_jump_run(run_dir) -> Path | None:
    """Draw <run_dir>/plots/jump.png; returns its path (None if nothing is logged yet)."""
    run_dir = Path(run_dir)
    data = read_scalars(run_dir)
    panels = [p for p in JUMP_PANELS if any(t in data for t in p[1])]
    if not panels:
        return None
    fig = Figure(figsize=(6 * len(panels), 4.2))
    fig.suptitle(f"{run_dir.name}: jump attempts")
    for i, (name, tags) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        for tag in tags:
            if tag in data:
                ax.plot(*data[tag], lw=1.2, label=tag.split("/")[-1])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if len(tags) > 1:
            ax.legend(fontsize=7)
    out = run_dir / "plots" / "jump.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return out


class JumpLivePlotter(LivePlotter):
    """The shared live plots (training.png, rewards.png) plus jump.png."""

    def draw(self):
        super().draw()
        try:
            plot_jump_run(self.run_dir)
        except Exception as e:  # never let plotting break training
            print(f"[WARN] Could not draw the jump plot: {e}")
