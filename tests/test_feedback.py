from pose_analysis.feedback import Feedback, FormIssue, gate_feedback


def test_low_confidence_is_uncertain_and_suppresses_issues():
    issues = [
        FormIssue(
            type="elbow_alignment",
            confidence=0.9,
            message="Keep the shooting elbow more directly under the wrist.",
        )
    ]
    result = gate_feedback(
        movement="jumpshot",
        confidence=0.4,
        phase="release",
        issues=issues,
        threshold=0.65,
    )

    assert result == Feedback(
        movement="uncertain",
        confidence=0.4,
        phase=None,
        issues=(),
    )


def test_high_confidence_keeps_movement_phase_and_issues():
    issue = FormIssue(
        type="elbow_alignment",
        confidence=0.84,
        message="Keep the shooting elbow more directly under the wrist.",
    )
    result = gate_feedback(
        movement="jumpshot",
        confidence=0.91,
        phase="release",
        issues=[issue],
        threshold=0.65,
    )

    assert result.to_dict() == {
        "movement": "jumpshot",
        "confidence": 0.91,
        "phase": "release",
        "issues": [
            {
                "type": "elbow_alignment",
                "confidence": 0.84,
                "message": "Keep the shooting elbow more directly under the wrist.",
            }
        ],
    }
