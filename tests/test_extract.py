import numpy as np

from pose_analysis.extract import extract_frames
from pose_analysis.records import Landmark, NUM_LANDMARKS


class _FakeLandmarker:
    def __init__(self, missing_at=None):
        self.missing_at = set(missing_at or [])
        self.seen = []

    def detect(self, frame_rgb, timestamp_ms: int):
        self.seen.append(timestamp_ms)
        if len(self.seen) - 1 in self.missing_at:
            return None
        return tuple(Landmark(0.1, 0.2, 0.3, 0.9) for _ in range(NUM_LANDMARKS))


def test_extract_frames_sets_video_id_and_timestamps():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    landmarker = _FakeLandmarker()

    records = extract_frames(frames, landmarker, video_id="player_001_session_01", fps=30)

    assert [r.video_id for r in records] == ["player_001_session_01"] * 3
    assert [r.frame_index for r in records] == [0, 1, 2]
    assert records[1].timestamp == 1 / 30
    assert records[0].landmarks[0].x == 0.1
    assert landmarker.seen[1] == int(round(1000 / 30))


def test_extract_frames_keeps_missing_poses_with_zero_visibility():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    records = extract_frames(
        frames, _FakeLandmarker(missing_at={1}), video_id="clip", fps=10
    )

    assert records[1].landmarks[0].visibility == 0.0
    assert len(records) == 2
