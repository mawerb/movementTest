import cv2
import numpy as np

from pose_analysis.landmarks import LEFT_ELBOW, LEFT_SHOULDER
from pose_analysis.overlay import draw_overlay, overlay_bgr_frames, write_overlay_video
from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS


def _record(index: int, visible: bool = True) -> FrameRecord:
    visibility = 1.0 if visible else 0.0
    points = [Landmark(0.1, 0.1, 0.0, visibility) for _ in range(NUM_LANDMARKS)]
    points[LEFT_SHOULDER] = Landmark(0.2, 0.2, 0.0, visibility)
    points[LEFT_ELBOW] = Landmark(0.8, 0.8, 0.0, visibility)
    return FrameRecord(
        video_id="clip",
        frame_index=index,
        timestamp=index / 10.0,
        landmarks=tuple(points),
    )


def test_draw_overlay_preserves_frame_size():
    bgr = np.zeros((80, 120, 3), dtype=np.uint8)
    overlay = draw_overlay(bgr, _record(0))
    assert overlay.shape == bgr.shape
    assert not np.array_equal(overlay, bgr)


def test_overlay_bgr_frames_pairs_each_record_and_stops_at_the_shorter_side():
    frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(3)]
    records = [_record(0), _record(1)]

    overlays = overlay_bgr_frames(frames, records)

    assert len(overlays) == 2
    assert overlays[0].shape == frames[0].shape
    assert not np.array_equal(overlays[0], frames[0])


def test_write_overlay_video_replays_jsonl_onto_source_clip(tmp_path):
    source = tmp_path / "source.avi"
    height, width = 32, 48
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    out = tmp_path / "overlay.avi"
    written = write_overlay_video(source, [_record(i) for i in range(3)], out)

    assert written == 3
    assert out.exists()
    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert frame.shape == (height, width, 3)
    assert frame.max() > 0
