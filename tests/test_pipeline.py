from pose_analysis.feedback import DEFAULT_CONFIDENCE_THRESHOLD
from pose_analysis.pipeline import analyze_posed_window
from tests.poses import stacked_release_pose, standing_pose, unbalanced_landing_pose
from pose_analysis.landmarks import LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE
from pose_analysis.records import Landmark


def test_posed_jumpshot_window_returns_gated_feedback():
    crouch = standing_pose(
        {
            LEFT_HIP: Landmark(-0.2, -0.25, 0.0, 1.0),
            RIGHT_HIP: Landmark(0.2, -0.25, 0.0, 1.0),
            LEFT_KNEE: Landmark(0.05, -0.7, 0.0, 1.0),
            RIGHT_KNEE: Landmark(0.35, -0.7, 0.0, 1.0),
        }
    )
    window = [crouch for _ in range(10)]
    window.extend(stacked_release_pose() for _ in range(8))
    window.append(unbalanced_landing_pose())

    result = analyze_posed_window(window)

    assert result.movement == "jumpshot"
    assert result.confidence >= DEFAULT_CONFIDENCE_THRESHOLD
    assert result.phase is not None
    assert any(issue.type == "landing_balance" for issue in result.issues)


def test_standing_window_is_uncertain_without_form_advice():
    result = analyze_posed_window([standing_pose() for _ in range(20)])
    assert result.movement == "uncertain"
    assert result.issues == ()
