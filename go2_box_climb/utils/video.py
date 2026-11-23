"""Save what the viewport camera sees to an MP4 file (the ``--video`` option).

Works in every viewing mode, also ``--mode headless`` on a cluster: with
``--video``, launch.py renders the camera off screen even though no window is
open. Videos are written to ``videos/`` at the repository root, which git ignores.

Usage in a script that starts Isaac Sim (after ``launch``):

    video = VideoRecorder("play", fps=1 / C.POLICY_DT, enabled=args.video)
    while ...:
        ...  # one policy step
        video.capture()  # one frame per step, so the video plays at real speed
    video.close()  # finishes the file and prints where it is

The MP4 uses the MPEG-4 codec that OpenCV (already installed) can write, so no
extra packages are needed.
"""

import atexit
from datetime import datetime
from pathlib import Path

import numpy as np

VIDEO_DIR = Path(__file__).resolve().parents[2] / "videos"

_open_recorders: list["VideoRecorder"] = []


def _kit_viewport_frame() -> np.ndarray:
    """One RGB frame (height x width x 3, uint8) from the Kit viewport camera."""
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    for viz in sim.visualizers if sim is not None else []:
        if getattr(viz.cfg, "visualizer_type", None) == "kit":
            return viz.render_rgb_array()
    raise RuntimeError("No Kit viewport to record from: start the script with --video (see utils/launch.py).")


class VideoRecorder:
    """Writes one MP4 file, one frame per ``capture()`` call."""

    def __init__(self, name: str, fps: float, enabled: bool = True, out_dir: Path = VIDEO_DIR, frame_source=None):
        """
        Args:
            name: Start of the file name; the date and time are added (``<name>_<YYYY-MM-DD_HH-MM-SS>.mp4``).
            fps: Playback frame rate. Use 1 / (time between two ``capture()`` calls) for real speed.
            enabled: If False, every method does nothing, so scripts can always call them.
            out_dir: Folder for the file (default ``videos/`` at the repository root).
            frame_source: Function returning one RGB frame; default is the Kit viewport camera.
        """
        self.enabled = enabled
        self.fps = float(fps)
        self.path = Path(out_dir) / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
        self.frames = 0
        self._frame_source = frame_source or _kit_viewport_frame
        self._writer = None
        if enabled:
            _open_recorders.append(self)

    def capture(self) -> None:
        """Grab the current camera image and add it to the video."""
        if not self.enabled:
            return
        import cv2

        frame = np.asarray(self._frame_source(), dtype=np.uint8)
        if self.frames == 0 and not frame.any():
            return  # the renderer returns black frames while it warms up; skip them at the start
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open {self.path} for writing")
        self._writer.write(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR))  # OpenCV expects BGR
        self.frames += 1

    def close(self) -> None:
        """Finish the file (an MP4 is only playable after this). Safe to call twice."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[VIDEO] Saved {self.frames} frames ({self.frames / self.fps:.1f} s) to: {self.path}")
        elif self.enabled and self.frames == 0:
            print("[VIDEO] No frames were recorded, so no video was saved.")
        self.enabled = False
        if self in _open_recorders:
            _open_recorders.remove(self)


def close_all_videos() -> None:
    """Finish every open video, e.g. when the script stops early (Ctrl+C). Called by ``close_app``."""
    for recorder in list(_open_recorders):
        recorder.close()


atexit.register(close_all_videos)
