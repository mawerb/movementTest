from pose_analysis.jumpshot import (
    check_elbow_alignment,
    check_landing_balance,
    check_release_stack,
    evaluate_jumpshot,
    jumpshot_phase,
)
from pose_analysis.landmarks import LEFT_HIP, LEFT_KNEE, RIGHT_HIP, RIGHT_KNEE
from pose_analysis.records import Landmark
from tests.poses import (
    dropped_elbow_release_pose,
    flared_elbow_release_pose,
    stacked_release_pose,
    standing_pose,
    unbalanced_landing_pose,
)


def test_stacked_release_has_no_elbow_or_stack_issues():
    pose = stacked_release_pose()
    assert check_elbow_alignment(pose) is None
    assert check_release_stack(pose) is None


def test_flared_elbow_is_flagged():
    issue = check_elbow_alignment(flared_elbow_release_pose())
    assert issue is not None
    assert issue.type == "elbow_alignment"
    assert "elbow" in issue.message.lower()


def test_wrist_below_elbow_flags_release_stack():
    issue = check_release_stack(dropped_elbow_release_pose())
    assert issue is not None
    assert issue.type == "release_stack"


def test_asymmetric_ankles_flag_landing_balance():
    issue = check_landing_balance(unbalanced_landing_pose())
    assert issue is not None
    assert issue.type == "landing_balance"


def test_evaluate_jumpshot_uses_release_and_landing_frames():
    window = [standing_pose() for _ in range(10)]
    window[6] = stacked_release_pose()
    window[-1] = unbalanced_landing_pose()

    result = evaluate_jumpshot(window)

    assert result.phase in {"release", "follow_through", "landing"}
    types = {issue.type for issue in result.issues}
    assert "landing_balance" in types
    assert "elbow_alignment" not in types


def test_phase_is_crouch_when_knees_flex_and_hips_drop():
    crouched = standing_pose(
        {
            LEFT_HIP: Landmark(-0.2, -0.25, 0.0, 1.0),
            RIGHT_HIP: Landmark(0.2, -0.25, 0.0, 1.0),
            LEFT_KNEE: Landmark(0.05, -0.7, 0.0, 1.0),
            RIGHT_KNEE: Landmark(0.35, -0.7, 0.0, 1.0),
        }
    )
    window = [standing_pose() for _ in range(8)] + [crouched]
    assert jumpshot_phase(window) == "crouch"
