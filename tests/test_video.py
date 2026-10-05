import cv2
import numpy as np

from pose_analysis.video import iter_rgb_frames, open_video


def test_iter_rgb_frames_reads_written_clip(tmp_path):
    path = tmp_path / "tiny.avi"
    height, width = 16, 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    assert writer.isOpened()
    for value in (10, 80, 160):
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()

    capture, fps = open_video(path)
    frames = list(iter_rgb_frames(capture))

    assert fps == 10
    assert len(frames) == 3
    assert frames[0].shape == (height, width, 3)
    assert frames[0].dtype == np.uint8
