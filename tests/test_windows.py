from pose_analysis.windows import sliding_windows, smooth_labels
from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS


def _frame(index: int) -> FrameRecord:
    return FrameRecord(
        video_id="clip",
        frame_index=index,
        timestamp=index / 30.0,
        landmarks=tuple(Landmark(0.0, 0.0, 0.0, 1.0) for _ in range(NUM_LANDMARKS)),
    )


def test_sliding_windows_overlap_by_stride():
    frames = [_frame(i) for i in range(10)]
    windows = sliding_windows(frames, size=4, stride=2)

    assert [tuple(f.frame_index for f in window) for window in windows] == [
        (0, 1, 2, 3),
        (2, 3, 4, 5),
        (4, 5, 6, 7),
        (6, 7, 8, 9),
    ]


def test_smooth_labels_uses_majority_in_neighborhood():
    labels = ["jumpshot", "jumpshot", "unknown", "jumpshot", "jumpshot"]
    assert smooth_labels(labels, radius=1) == [
        "jumpshot",
        "jumpshot",
        "jumpshot",
        "jumpshot",
        "jumpshot",
    ]
