from pose_analysis.annotations import SegmentAnnotation


def test_segment_annotation_matches_plan_shape():
    segment = SegmentAnnotation(
        video_id="player_001_session_01",
        start_time=3.0,
        end_time=5.0,
        movement="jumpshot",
        quality="correct",
    )
    assert segment.to_dict() == {
        "video_id": "player_001_session_01",
        "start_time": 3.0,
        "end_time": 5.0,
        "movement": "jumpshot",
        "quality": "correct",
    }
    assert SegmentAnnotation.from_dict(segment.to_dict()) == segment
