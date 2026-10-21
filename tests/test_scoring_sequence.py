from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS
from pose_analysis.scoring.sequence import SAMPLE_COUNT, frame_is_usable, missing_fraction, trial_tensor
from tests.poses import standing_pose


def _frames(count: int, missing_at: set[int]) -> list[FrameRecord]:
    blank = tuple(Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))
    frames = []
    for index in range(count):
        pose = blank if index in missing_at else standing_pose()
        frames.append(FrameRecord("01 Jump", index, index / 30.0, pose))
    return frames


def test_zero_visibility_pose_is_not_usable():
    blank = tuple(Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))
    assert frame_is_usable(blank) is False
    assert frame_is_usable(standing_pose()) is True


def test_missing_fraction_counts_unusable_frames():
    frames = _frames(4, {0, 1})
    assert missing_fraction(frames) == 0.5
    assert missing_fraction([]) == 1.0


def test_trial_tensor_is_25_by_64_and_keeps_missing_frames():
    frames = _frames(4, {0})
    tensor = trial_tensor(frames)
    assert tensor.shape == (25, SAMPLE_COUNT)
    assert tensor[24, 0] == 1.0
    assert tensor[:24, 0].sum() == 0.0
    assert tensor[24, -1] == 0.0
