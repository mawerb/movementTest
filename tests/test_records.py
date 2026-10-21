import json

from pose_analysis.records import FrameRecord, Landmark, dumps_jsonl, loads_jsonl


def test_frame_record_round_trips_plan_json_shape():
    record = FrameRecord(
        video_id="player_001_session_01",
        frame_index=120,
        timestamp=4.0,
        landmarks=[Landmark(x=0.1, y=0.2, z=0.3, visibility=0.9) for _ in range(33)],
    )

    payload = record.to_dict()

    assert payload["video_id"] == "player_001_session_01"
    assert payload["frame_index"] == 120
    assert payload["timestamp"] == 4.0
    assert payload["landmarks"] == [[0.1, 0.2, 0.3, 0.9] for _ in range(33)]

    restored = FrameRecord.from_dict(payload)
    assert restored == record


def test_jsonl_round_trip_keeps_frame_order():
    frames = [
        FrameRecord(
            video_id="clip",
            frame_index=i,
            timestamp=i / 30.0,
            landmarks=[Landmark(x=float(i), y=0.0, z=0.0, visibility=1.0) for _ in range(33)],
        )
        for i in range(3)
    ]

    text = dumps_jsonl(frames)
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 3
    assert json.loads(lines[0])["frame_index"] == 0

    assert loads_jsonl(text) == frames
