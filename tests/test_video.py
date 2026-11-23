"""Fast checks of the MP4 recorder behind --video (no Isaac Sim needed): run `pytest`."""

import numpy as np

from go2_box_climb.utils.video import VideoRecorder


def _frames(n: int, height: int = 48, width: int = 64):
    """A black warm-up frame followed by n frames that get brighter."""
    yield np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n):
        yield np.full((height, width, 3), 20 + 20 * i, dtype=np.uint8)


def test_video_is_written_and_readable(tmp_path):
    import cv2

    frames = _frames(10)
    video = VideoRecorder("test", fps=50, out_dir=tmp_path, frame_source=lambda: next(frames))
    for _ in range(11):
        video.capture()
    video.close()

    assert video.frames == 10  # the black warm-up frame at the start is skipped
    assert video.path.exists() and video.path.parent == tmp_path
    reader = cv2.VideoCapture(str(video.path))
    read = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        assert frame.shape == (48, 64, 3)
        read += 1
    assert read == 10


def test_disabled_recorder_does_nothing(tmp_path):
    def no_frames():
        raise AssertionError("a disabled recorder must not grab frames")

    video = VideoRecorder("test", fps=50, enabled=False, out_dir=tmp_path, frame_source=no_frames)
    video.capture()
    video.close()
    assert not any(tmp_path.iterdir())
