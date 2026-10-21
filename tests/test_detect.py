from pose_analysis.detect import classify_window
from tests.poses import stacked_release_pose, standing_pose
from pose_analysis.landmarks import LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE
from pose_analysis.records import Landmark


def _crouch_pose():
    return standing_pose(
        {
            LEFT_HIP: Landmark(-0.2, -0.25, 0.0, 1.0),
            RIGHT_HIP: Landmark(0.2, -0.25, 0.0, 1.0),
            LEFT_KNEE: Landmark(0.05, -0.7, 0.0, 1.0),
            RIGHT_KNEE: Landmark(0.35, -0.7, 0.0, 1.0),
        }
    )


def test_crouch_then_high_release_classifies_as_jumpshot():
    window = [_crouch_pose() for _ in range(10)]
    window.extend(stacked_release_pose() for _ in range(8))
    window.extend(standing_pose() for _ in range(8))

    movement, confidence = classify_window(window)

    assert movement == "jumpshot"
    assert confidence >= 0.65


def test_static_standing_is_unknown_with_low_confidence():
    movement, confidence = classify_window([standing_pose() for _ in range(20)])
    assert movement == "unknown"
    assert confidence < 0.65
