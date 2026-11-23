"""Start Isaac Sim in one of three viewing modes (used by every script).

    --mode headless   no window; fastest (training, batch evaluation)
    --mode gui        open the Isaac Sim window on this computer
    --mode stream     no local window; stream the viewport over WebRTC so you can
                      watch from another computer with the
                      "Isaac Sim WebRTC Streaming Client", connecting to the IP
                      address this script prints

    --video           (with any mode) also save what the camera sees to an MP4
                      in videos/; the script records it with
                      go2_box_climb.utils.video.VideoRecorder

Usage in a script (must happen BEFORE importing anything from isaaclab.* other
than isaaclab.app):

    parser = argparse.ArgumentParser()
    add_launch_args(parser)
    args = parser.parse_args()
    simulation_app = launch(args)
    # ... now import isaaclab modules and build the scene ...
"""

import argparse
import socket
import sys

MODES = ("headless", "gui", "stream")

# The WebRTC client asks for 1080p; matching it avoids a blurry/cropped stream.
_STREAM_WIDTH, _STREAM_HEIGHT = 1920, 1080


def add_launch_args(parser: argparse.ArgumentParser, default_mode: str = "gui") -> None:
    """Add ``--mode`` plus Isaac Lab's standard launcher arguments (``--device``, ...)."""
    from isaaclab.app import AppLauncher

    parser.add_argument(
        "--mode",
        choices=MODES,
        default=default_mode,
        help="headless: no window | gui: Isaac Sim window | stream: WebRTC livestream to another computer",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save what the camera sees to an MP4 in videos/ (scripts that record, e.g. locomotion/play.py; any mode)",
    )
    AppLauncher.add_app_launcher_args(parser)


def local_ip() -> str:
    """Best guess of this computer's IP address on the network (what the stream client connects to)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def launch(args: argparse.Namespace):
    """Launch Isaac Sim according to ``args.mode`` and return the ``SimulationApp``."""
    # Import torch BEFORE Isaac Sim starts: the livestream extension loads an older
    # CUDA runtime library, and if torch is imported after it, torch crashes with
    # "undefined symbol: cudaGetDriverEntryPointByVersion".
    import torch  # noqa: F401
    from isaaclab.app import AppLauncher

    launcher_args = vars(args).copy()
    mode = launcher_args.pop("mode")
    # Our own option: Isaac Lab also reads a "video" key, and would then ask for the
    # moviepy package, which is not in our lock file. We record with OpenCV instead.
    video = launcher_args.pop("video", False)
    if mode == "headless" and video:
        # Still no window, but keep a Kit viewport camera and render it off screen
        # (enable_cameras), so utils/video.py can read its images.
        launcher_args.update(
            livestream=0, headless=True, visualizer=["kit"], visualizer_explicit=True, enable_cameras=True
        )
    elif mode == "headless":
        launcher_args.update(livestream=0, visualizer=["none"], visualizer_explicit=True)
    elif mode == "gui":
        launcher_args.update(livestream=0, visualizer=["kit"], visualizer_explicit=True)
    elif mode == "stream":
        # livestream=2: WebRTC on the local network. Isaac Sim runs headless on this
        # computer and the client connects to this computer's IP address.
        launcher_args.update(livestream=2, visualizer=["kit"], visualizer_explicit=True)
        sys.argv += [
            f"--/app/window/width={_STREAM_WIDTH}",
            f"--/app/window/height={_STREAM_HEIGHT}",
            f"--/app/renderer/resolution/width={_STREAM_WIDTH}",
            f"--/app/renderer/resolution/height={_STREAM_HEIGHT}",
        ]
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    app = AppLauncher(launcher_args).app

    if mode == "stream":
        bar = "=" * 70
        print(f"\n{bar}\n[STREAM] Open the Isaac Sim WebRTC Streaming Client and connect to: {local_ip()}\n"
              f"[STREAM] (wait until the scene has loaded; the first start can take a few minutes)\n{bar}\n")
    return app


def close_app(app, timeout_s: float = 30.0) -> None:
    """Close Isaac Sim, and force the process to exit if closing hangs.

    ``SimulationApp.close()`` sometimes never returns (and then ignores Ctrl+C /
    ``timeout``), leaving the process holding GPU memory. A watchdog ends the
    process after ``timeout_s`` seconds in that case.
    """
    import os
    import threading

    from go2_box_climb.utils.video import close_all_videos

    close_all_videos()  # finish any MP4 first: the watchdog below may end the process abruptly
    sys.stdout.flush()
    watchdog = threading.Timer(timeout_s, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    watchdog.cancel()
