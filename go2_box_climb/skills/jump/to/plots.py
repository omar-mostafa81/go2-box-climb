"""Plots and an animation of an optimized jump. Part: TO.

Figures (all saved as PNG):
    foot_paths      side view: the path of every foot over the box, with the robot (its real
                    collision shapes) drawn at the start, at the top of the flight and at the end
    foot_timeline   foot height, foot forward position and foot contacts over time
    feet_3d         the four foot paths in 3D (left and right feet side by side)
    jump_dynamics   trunk height and pitch, ground forces, motor torques over time, every
                    motor's torque against its speed with the DC-motor limit, joint angles vs limits
And ``jump_animation.gif`` / ``.mp4``: the robot jumping, drawn from the optimized joint angles.

Colors: front feet (FL, FR) blue, rear feet (RL, RR) orange (first two slots of a
color-blind-safe palette). Left feet are solid lines, right feet dashed, thighs solid and
calves dashed, so color is never the only cue. Jump phases are gray bands with their names.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from go2_box_climb.common import constants as C  # noqa: E402

from . import planar_model as pm  # noqa: E402
from .optimizer import PHASES, JumpSolution  # noqa: E402
from .robot_params import JumpRobotParams  # noqa: E402

# --- colors (validated palette slots 1-2) and chart ink ---
FRONT = "#2a78d6"  # front feet / front legs
REAR = "#eb6834"  # rear feet / rear legs
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BOX_FILL = "#e1e0d9"
PLATFORM_FILL = "#c3c2b7"
TRUNK_FILL = "#d9d8d2"
PHASE_BANDS = ("#f0efec", "#fcfcfb")  # alternating light gray bands

FOOT_STYLE = {  # foot: (side-view leg, color, line style, label)
    "FL": (0, FRONT, "-", "FL (front left)"),
    "FR": (0, FRONT, "--", "FR (front right)"),
    "RL": (1, REAR, "-", "RL (rear left)"),
    "RR": (1, REAR, "--", "RR (rear right)"),
}
JOINTS = ((FRONT, "-", "front thigh"), (FRONT, "--", "front calf"), (REAR, "-", "rear thigh"), (REAR, "--", "rear calf"))


def _style(ax):
    """Light, recessive axes: hairline grid, muted spines and ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _events(sol: JumpSolution) -> dict[str, int]:
    """Node index of the key moments of the jump."""
    first = {p: int(np.argmax(sol.phase == p)) for p in range(len(PHASES))}
    flight = np.where(sol.phase == 2)[0]
    return {
        "start": 0,
        "front lift-off": first[1],
        "take-off": first[2],
        "apex": int(flight[np.argmax(sol.base[flight, 1])]),
        "touch-down": first[3],
        "end": len(sol.t) - 1,
    }


def _x_range(sol: JumpSolution) -> tuple[float, float]:
    return sol.task.start_base_x - 0.45, sol.task.steps[-1][0] + 0.35


def _draw_scene(ax, sol: JumpSolution, x_max: float):
    """Ground and the steps (for jump 1: the box and the platform behind it), side view."""
    task = sol.task
    ax.axhline(task.start_height, color=AXIS, linewidth=1.2, zorder=1)
    names = ("box", "platform") if task.name == "ground_to_box" else ("",) * len(task.steps)
    fills = (BOX_FILL, PLATFORM_FILL)
    for i, (x0, top) in enumerate(task.steps):
        x1 = task.steps[i + 1][0] if i + 1 < len(task.steps) else x_max
        ax.add_patch(Rectangle((x0, 0.0), x1 - x0, top, facecolor=fills[i % 2], edgecolor=MUTED, linewidth=1, zorder=1))
        ax.text(0.5 * (x0 + x1), 0.5 * top, names[i], ha="center", va="center", color=INK_2, fontsize=9, zorder=1)


def _robot_shapes(x, z, th, joints, robot: JumpRobotParams):
    """World coordinates of the robot's collision shapes in the side view."""
    L1, L2 = robot.thigh_length, robot.calf_length
    hx, hz = robot.trunk_half_size
    trunk = np.array([pm.point_world(x, z, th, bx, bz) for bx, bz in ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))])
    heads = [(pm.point_world(x, z, th, cx, cz), rad) for cx, cz, rad in (robot.head_upper, robot.head_lower)]
    legs = []
    for leg, hip_x in enumerate((robot.front_hip_x, robot.rear_hip_x)):
        t, c = joints[2 * leg], joints[2 * leg + 1]
        hip = pm.point_world(x, z, th, hip_x, 0.0)
        hz, tilt = robot.thigh_joint_z, robot.leg_tilt_cos
        knee = pm.knee_world(x, z, th, hip_x, t, L1, np, hz, tilt)
        foot = pm.foot_world(x, z, th, hip_x, t, c, L1, L2, np, hz, tilt)
        legs.append(np.array([hip, knee, foot]))
    return trunk, heads, legs


def _draw_robot(ax, x, z, th, joints, robot: JumpRobotParams, alpha=1.0, ghost=False):
    """Trunk box, head, hip motors, thighs, calves and feet, at their real sizes."""
    trunk, heads, legs = _robot_shapes(x, z, th, joints, robot)
    edge = MUTED if ghost else INK_2
    face = GRID if ghost else TRUNK_FILL
    for leg in (1, 0):  # rear leg first, so the front leg is drawn on top
        pts = legs[leg]
        color = MUTED if ghost else (FRONT, REAR)[leg]
        ax.plot(pts[:2, 0], pts[:2, 1], color=color, linewidth=3.5 if not ghost else 2.5, alpha=alpha, zorder=4,
                solid_capstyle="round")  # thigh  # fmt: skip
        ax.plot(pts[1:, 0], pts[1:, 1], color=color, linewidth=2.0 if not ghost else 1.5, alpha=alpha, zorder=4,
                solid_capstyle="round")  # calf  # fmt: skip
        ax.add_patch(Circle(pts[0], robot.hip_radius, facecolor=face, edgecolor=edge, linewidth=1, alpha=alpha, zorder=5))
        if not ghost:  # ghosts skip the feet so they are not mistaken for data points
            ax.add_patch(Circle(pts[2], robot.foot_radius, color=color, alpha=alpha, zorder=5))
    ax.add_patch(Polygon(trunk, closed=True, facecolor=face, edgecolor=edge, linewidth=1, alpha=alpha, zorder=6))
    for center, rad in heads:
        ax.add_patch(Circle(center, rad, facecolor=face, edgecolor=edge, linewidth=1, alpha=alpha, zorder=6))


def _phase_bands(ax, sol: JumpSolution, labels: bool):
    """Gray bands behind a time axis, one per phase, with the phase name on top."""
    for p, (name, *_rest) in enumerate(PHASES):
        idx = np.where(sol.phase == p)[0]
        t0 = sol.t[idx[0]]
        t1 = sol.t[min(idx[-1] + 1, len(sol.t) - 1)]
        ax.axvspan(t0, t1, color=PHASE_BANDS[p % 2], zorder=0, linewidth=0)
        if labels:  # alternate two heights so names of short phases do not overlap
            ax.text(0.5 * (t0 + t1), 1.02 + 0.07 * (p % 2), name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=INK_2)  # fmt: skip


def _side_view_axes(ax, sol: JumpSolution):
    x_min, x_max = _x_range(sol)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.03, max(0.95, sol.base[:, 1].max() + 0.3))
    ax.set_aspect("equal")
    ax.set_xlabel("forward position x [m]")
    ax.set_ylabel("height z [m]")


def plot_foot_paths(sol: JumpSolution, robot: JumpRobotParams, path: Path) -> Path:
    """Side view: path of every foot, the center of mass, and the robot at three key moments."""
    ev = _events(sol)
    fig, ax = _figure(figsize=(10, 4.8))
    _draw_scene(ax, sol, _x_range(sol)[1])
    for name in ("start", "apex", "end"):
        k = ev[name]
        _draw_robot(ax, *sol.base[k], sol.joints[k], robot, alpha=0.5, ghost=True)
    ax.plot(sol.com[:, 0], sol.com[:, 1], color=INK_2, linestyle=":", linewidth=1.5, label="center of mass", zorder=7)
    for leg, (color, name) in enumerate(((FRONT, "front feet (FL, FR)"), (REAR, "rear feet (RL, RR)"))):
        ax.plot(sol.feet[:, leg, 0], sol.feet[:, leg, 1], color=color, linewidth=2, label=name, zorder=8)
        lift = np.where(sol.contact[:-1, leg] & ~sol.contact[1:, leg])[0]
        land = np.where(~sol.contact[:-1, leg] & sol.contact[1:, leg])[0] + 1
        ax.scatter(sol.feet[lift, leg, 0], sol.feet[lift, leg, 1], s=60, facecolor=SURFACE, edgecolor=color,
                   linewidth=2, zorder=9)  # fmt: skip
        ax.scatter(sol.feet[land, leg, 0], sol.feet[land, leg, 1], s=60, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=9)  # fmt: skip
    k = ev["apex"]
    ax.annotate(f"top of the flight, t = {sol.t[k]:.2f} s", xy=(sol.com[k, 0], sol.com[k, 1]),
                xytext=(sol.com[k, 0] - 0.45, sol.com[k, 1] + 0.2), color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))  # fmt: skip
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK_2, linewidth=2, label="lift-off")
    ax.scatter([], [], s=60, color=INK_2, label="touch-down")
    _side_view_axes(ax, sol)
    ax.set_title(f"Planned foot paths, {sol.task.name.replace('_', ' ')} jump (side view; gray = robot at its real "
                 "size at start, top of flight, end)", color=INK, fontsize=10.5, loc="left")  # fmt: skip
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    return _save(fig, path)


def plot_foot_timeline(sol: JumpSolution, path: Path) -> Path:
    """Foot height, foot forward position and the contact pattern, against time."""
    fig, axes = _figure(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.6]})
    ax_z, ax_x, ax_c = axes
    for ax in axes:
        _phase_bands(ax, sol, labels=ax is ax_z)
    for foot, (leg, color, ls, label) in FOOT_STYLE.items():
        ax_z.plot(sol.t, sol.feet[:, leg, 1], color=color, linestyle=ls, linewidth=2, label=label)
        ax_x.plot(sol.t, sol.feet[:, leg, 0], color=color, linestyle=ls, linewidth=2, label=label)
    task = sol.task
    ax_z.axhline(task.target_height, color=MUTED, linestyle="--", linewidth=1)
    ax_z.text(sol.t[-1], task.target_height, " box top", va="bottom", ha="right", color=INK_2, fontsize=9)
    for x_line, text in ((task.target_x[0], " box front edge"), (task.target_x[1], " box back edge")):
        ax_x.axhline(x_line, color=MUTED, linestyle="--", linewidth=1)
        ax_x.text(sol.t[0], x_line, text, va="bottom", ha="left", color=INK_2, fontsize=9)
    ax_z.set_ylabel("foot center height [m]")
    ax_x.set_ylabel("foot forward position [m]")
    ax_z.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    # contact (gait) diagram: one row per foot, a bar where the foot touches a surface
    for row, (foot, (leg, color, _ls, _label)) in enumerate(FOOT_STYLE.items()):
        on = sol.contact[:, leg]
        for s in np.where(on & np.r_[True, ~on[:-1]])[0]:
            e = s
            while e + 1 < len(on) and on[e + 1]:
                e += 1
            ax_c.barh(row, sol.t[e] - sol.t[s], left=sol.t[s], height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
    ax_c.set_yticks(range(4), list(FOOT_STYLE))
    ax_c.invert_yaxis()
    ax_c.set_ylabel("contact")
    ax_c.grid(False, axis="y")
    ax_c.set_xlabel("time [s]")
    fig.suptitle("Planned foot motion over time (left and right feet move together in this side-view plan)",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    return _save(fig, path)


def plot_feet_3d(sol: JumpSolution, robot: JumpRobotParams, path: Path) -> Path:
    """The four foot paths in 3D: left feet at +y, right feet at -y."""
    fig = plt.figure(figsize=(9, 5.5))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(SURFACE)
    task = sol.task

    def box_edges(x0, x1, y0, y1, z0, z1, color):
        for zz in (z0, z1):
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zz] * 5, color=color, linewidth=1)
        for xx in (x0, x1):
            for yy in (y0, y1):
                ax.plot([xx, xx], [yy, yy], [z0, z1], color=color, linewidth=1)

    y_half = C.BOX_SIZE[1] / 2
    box_edges(task.target_x[0], task.target_x[1], -y_half, y_half, task.start_height, task.target_height, MUTED)
    if len(task.steps) > 1:  # the front part of the platform, for context
        px, ph = task.steps[1]
        box_edges(px, px + 0.3, -0.5, 0.5, 0.0, ph, AXIS)
    for foot, (leg, color, ls, label) in FOOT_STYLE.items():
        y = robot.hip_y if foot.endswith("L") else -robot.hip_y
        ax.plot(sol.feet[:, leg, 0], np.full(len(sol.t), y), sol.feet[:, leg, 1], color=color, linestyle=ls,
                linewidth=2, label=label)  # fmt: skip
        ax.scatter([sol.feet[0, leg, 0]], [y], [sol.feet[0, leg, 1]], color=color, s=25)
        ax.scatter([sol.feet[-1, leg, 0]], [y], [sol.feet[-1, leg, 1]], color=color, s=25, marker="s")
    ax.plot(sol.com[:, 0], np.zeros(len(sol.t)), sol.com[:, 1], color=INK_2, linestyle=":", linewidth=1.5,
            label="center of mass")  # fmt: skip
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_ylim(-0.5, 0.5)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect((np.ptp(ax.get_xlim()), 1.0, np.ptp(ax.get_zlim())))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title("Planned paths of all four feet (circle = start, square = end; box and platform outlined)",
                 color=INK, fontsize=10.5, loc="left")  # fmt: skip
    return _save(fig, path)


def plot_jump_dynamics(sol: JumpSolution, robot: JumpRobotParams, path: Path) -> Path:
    """Trunk height and pitch, ground forces, torques over time, torque vs speed, joint angles."""
    fig, axes = _figure(2, 3, figsize=(16, 8.5))
    (ax_h, ax_p, ax_q), (ax_f, ax_t, ax_ts) = axes
    for ax in (ax_h, ax_p, ax_q, ax_f, ax_t):
        _phase_bands(ax, sol, labels=ax in (ax_h, ax_p, ax_q))
        ax.set_xlabel("time [s]")

    ax_h.plot(sol.t, sol.com[:, 1], color=INK_2, linewidth=2)
    ax_h.set_ylabel("center of mass height [m]")
    ax_p.plot(sol.t, np.degrees(sol.base[:, 2]), color=INK_2, linewidth=2)
    ax_p.axhline(0.0, color=AXIS, linewidth=1)
    ax_p.set_ylabel("pitch [deg] (+ = nose down)")

    limits = (robot.thigh_limits_front, robot.calf_limits, robot.thigh_limits_rear, robot.calf_limits)
    for j, (color, ls, name) in enumerate(JOINTS):
        ax_q.plot(sol.t, sol.joints[:, j], color=color, linestyle=ls, linewidth=2, label=name)
    for lo, hi in {limits[1]}:  # both calves share their limits; thigh limits are far outside this range
        for v in (lo, hi):
            ax_q.axhline(v, color=MUTED, linestyle=":", linewidth=1)
        ax_q.text(sol.t[0], hi, " calf limits", va="bottom", ha="left", color=INK_2, fontsize=8)
    ax_q.set_ylabel("joint angle [rad]")
    ax_q.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))

    weight = robot.mass * 9.81
    for leg, (color, name) in enumerate(((FRONT, "front pair"), (REAR, "rear pair"))):
        ax_f.step(sol.t, sol.forces[:, 2 * leg + 1], where="post", color=color, linewidth=2, label=f"{name}, vertical")
        ax_f.step(sol.t, sol.forces[:, 2 * leg], where="post", color=color, linewidth=1.5, linestyle="--",
                  label=f"{name}, forward")  # fmt: skip
    ax_f.axhline(weight, color=MUTED, linestyle=":", linewidth=1)
    ax_f.text(0.0, weight, " robot weight", va="bottom", ha="left", color=INK_2, fontsize=8)
    ax_f.set_ylabel("ground force on the pair [N]")
    ax_f.legend(fontsize=8, frameon=False, loc="upper right")

    for j, (color, ls, name) in enumerate(JOINTS):
        ax_t.plot(sol.t, sol.torques[:, j], color=color, linestyle=ls, linewidth=2, label=name)
    for s in (-1, 1):
        ax_t.axhline(s * C.EFFORT_LIMIT, color=MUTED, linestyle=":", linewidth=1)
    ax_t.text(sol.t[-1], C.EFFORT_LIMIT, " motor limit 23.5 N m", va="bottom", ha="right", color=INK_2, fontsize=8)
    ax_t.set_ylabel("motor torque per leg [N m]")
    ax_t.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)

    # torque vs speed with the DC-motor limit: tau in [23.5 (-1 - w/30), 23.5 (1 - w/30)], |tau| <= 23.5
    w = np.linspace(-C.VELOCITY_LIMIT, C.VELOCITY_LIMIT, 200)
    up = np.minimum(C.EFFORT_LIMIT, C.EFFORT_LIMIT * (1 - w / C.VELOCITY_LIMIT))
    down = np.maximum(-C.EFFORT_LIMIT, C.EFFORT_LIMIT * (-1 - w / C.VELOCITY_LIMIT))
    ax_ts.fill_between(w, down, up, color=PHASE_BANDS[0], zorder=0)
    ax_ts.plot(w, up, color=MUTED, linewidth=1.2)
    ax_ts.plot(w, down, color=MUTED, linewidth=1.2)
    ax_ts.text(-29, 21.5, "what the motor can do (DC-motor model)", color=INK_2, fontsize=8, va="top")
    markers = ("o", "o", "s", "s")
    for j, (color, ls, name) in enumerate(JOINTS):
        filled = ls == "-"
        ax_ts.scatter(sol.joint_vel[:, j], sol.torques[:, j], s=16, marker=markers[j], label=name,
                      facecolor=color if filled else SURFACE, edgecolor=color, linewidth=1, zorder=3)  # fmt: skip
    ax_ts.set_xlabel("joint speed [rad/s]")
    ax_ts.set_ylabel("motor torque [N m]")
    ax_ts.legend(fontsize=8, frameon=False, loc="lower left")
    ax_ts.set_title("every time step of every motor", color=INK_2, fontsize=9, loc="left")
    fig.suptitle("Trunk motion, ground forces, motor torques and joint angles of the planned jump", color=INK,
                 fontsize=11, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def animate_jump(sol: JumpSolution, robot: JumpRobotParams, path: Path, dt: float = 0.01, slowdown: float = 3.0,
                 hold_s: float = 0.6, dpi: int = 90) -> Path:  # fmt: skip
    """Animation of the robot doing the jump (``slowdown`` x slower than real time), with foot trails.

    The file type follows the name: ``.gif`` (Pillow) or ``.mp4`` (H.264 video that plays on
    phones and in WhatsApp; needs the ``imageio-ffmpeg`` package, which brings its own ffmpeg).
    """
    t = np.arange(0.0, sol.t[-1] + 1e-9, dt)
    base = np.stack([np.interp(t, sol.t, sol.base[:, j]) for j in range(3)], axis=1)
    joints = np.stack([np.interp(t, sol.t, sol.joints[:, j]) for j in range(4)], axis=1)
    node = np.clip(np.searchsorted(sol.t, t, side="right") - 1, 0, len(sol.t) - 1)
    feet = np.array([[pm.foot_world(*base[i], hx, joints[i, 2 * leg], joints[i, 2 * leg + 1], robot.thigh_length,
                                    robot.calf_length, np, robot.thigh_joint_z, robot.leg_tilt_cos)
                      for leg, hx in enumerate((robot.front_hip_x, robot.rear_hip_x))]
                     for i in range(len(t))])  # fmt: skip
    n_hold = int(round(hold_s / dt))  # frames that repeat the first / final pose
    frames = [0] * n_hold + list(range(len(t))) + [len(t) - 1] * n_hold

    fig, ax = _figure(figsize=(8, 4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14)

    def draw(i):
        ax.clear()
        _style(ax)
        _draw_scene(ax, sol, _x_range(sol)[1])
        for leg, color in enumerate((FRONT, REAR)):
            ax.plot(sol.feet[:, leg, 0], sol.feet[:, leg, 1], color=color, linewidth=1, alpha=0.25)  # full plan
            ax.plot(feet[: i + 1, leg, 0], feet[: i + 1, leg, 1], color=color, linewidth=2)  # trail so far
        _draw_robot(ax, *base[i], joints[i], robot)
        _side_view_axes(ax, sol)
        ax.set_title(f"t = {t[i]:.2f} s   phase: {PHASES[sol.phase[node[i]]][0]}   ({slowdown:g}x slower)",
                     color=INK, fontsize=10, loc="left")  # fmt: skip

    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 * dt * slowdown)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / (dt * slowdown)
    if path.suffix.lower() == ".mp4":
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        # H.264 + yuv420p + faststart: the format phones and WhatsApp play without converting
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=4000,
                                        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])  # fmt: skip
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path


SIM_COLORS = {"feedback": "#2a78d6", "feedforward": "#eb6834", "replay": "#1baf7a"}  # palette slots 1-3, fixed order


def plot_mujoco_comparison(sol: JumpSolution, results: dict, path: Path) -> Path:
    """The plan against MuJoCo runs (one per controller): trunk path, x, height, pitch, motor use.

    Args:
        results: {controller name: mujoco_sim.SimResult}
    """
    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=(1.1, 1.0))
    ax_side = fig.add_subplot(gs[0, :2])
    ax_ts = fig.add_subplot(gs[0, 2])
    ax_x, ax_z, ax_p = (fig.add_subplot(gs[1, i]) for i in range(3))
    for ax in (ax_side, ax_ts, ax_x, ax_z, ax_p):
        _style(ax)

    # side view: trunk (base origin) paths
    x_max = _x_range(sol)[1]
    _draw_scene(ax_side, sol, x_max)
    ax_side.plot(sol.base[:, 0], sol.base[:, 1], color=INK, linestyle="--", linewidth=2, label="plan", zorder=5)
    for mode, res in results.items():
        color = SIM_COLORS.get(mode, MUTED)
        label = f"MuJoCo, {mode}: {'success' if res.success else 'fail'}"
        ax_side.plot(res.base_pos[:, 0], res.base_pos[:, 2], color=color, linewidth=2, label=label, zorder=6)
        if not res.success:
            k = len(res.t) - 1 if res.fail_contact_time is None else int(np.searchsorted(res.t, res.fail_contact_time))
            ax_side.scatter(res.base_pos[k, 0], res.base_pos[k, 2], marker="X", s=120, color=color, edgecolor=SURFACE,
                            linewidth=1.5, zorder=7)  # fmt: skip
    ax_side.set_xlim(sol.task.start_base_x - 0.55, x_max)
    ax_side.set_ylim(-0.03, 0.8)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("forward position x [m]")
    ax_side.set_ylabel("height z [m]")
    ax_side.set_title("Trunk path (base origin): plan vs. MuJoCo (X = where a run failed)", color=INK_2, fontsize=10,
                      loc="left")  # fmt: skip
    ax_side.legend(loc="upper left", fontsize=9, frameon=False)

    # over time
    for ax, plan_y, sim_y, label in (
        (ax_x, sol.base[:, 0], lambda r: r.base_pos[:, 0], "forward position x [m]"),
        (ax_z, sol.base[:, 1], lambda r: r.base_pos[:, 2], "height z [m]"),
        (ax_p, np.degrees(sol.base[:, 2]), lambda r: np.degrees(r.pitch), "pitch [deg] (+ = nose down)"),
    ):
        ax.plot(sol.t, plan_y, color=INK, linestyle="--", linewidth=2, label="plan")
        for mode, res in results.items():
            ax.plot(res.t, sim_y(res), color=SIM_COLORS.get(mode, MUTED), linewidth=1.8, label=mode)
        ax.axvline(sol.t[-1], color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(label)
    ax_x.text(sol.t[-1], ax_x.get_ylim()[0], " plan ends", color=INK_2, fontsize=8, va="bottom")
    ax_p.legend(fontsize=8, frameon=False, loc="upper right")

    # motor use in the best run: every torque/speed pair against the DC-motor limit
    best = results.get("feedback", next(iter(results.values())))
    w = np.linspace(-C.VELOCITY_LIMIT, C.VELOCITY_LIMIT, 200)
    up = np.minimum(C.EFFORT_LIMIT, C.EFFORT_LIMIT * (1 - w / C.VELOCITY_LIMIT))
    down = np.maximum(-C.EFFORT_LIMIT, C.EFFORT_LIMIT * (-1 - w / C.VELOCITY_LIMIT))
    ax_ts.fill_between(w, down, up, color=PHASE_BANDS[0], zorder=0)
    ax_ts.plot(w, up, color=MUTED, linewidth=1.2)
    ax_ts.plot(w, down, color=MUTED, linewidth=1.2)
    groups = (("hips", slice(0, 4), MUTED), ("thighs", slice(4, 8), FRONT), ("calves", slice(8, 12), REAR))
    for name, cols, color in groups:
        ax_ts.scatter(best.joint_vel[::4, cols].ravel(), best.torque[::4, cols].ravel(), s=5, color=color, alpha=0.5,
                      label=name, zorder=3)  # fmt: skip
    ax_ts.set_xlabel("joint speed [rad/s]")
    ax_ts.set_ylabel("motor torque [N m]")
    ax_ts.set_title(f"MuJoCo motor use ({best.mode}) vs. DC-motor limit", color=INK_2, fontsize=10, loc="left")
    ax_ts.legend(fontsize=8, frameon=False, loc="lower left", markerscale=3)
    fig.suptitle("The planned jump simulated in MuJoCo with the project's motor contract (PD 25 / 0.5, DC-motor limit)",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
