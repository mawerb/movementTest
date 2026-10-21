import cv2
import numpy as np

from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS, write_jsonl
from tests.poses import standing_pose


def test_analyze_jsonl_prints_feedback(tmp_path, capsys):
    from main import main

    frames = [
        FrameRecord(
            video_id="clip",
            frame_index=i,
            timestamp=i / 30.0,
            landmarks=standing_pose(),
        )
        for i in range(20)
    ]
    path = tmp_path / "clip.jsonl"
    write_jsonl(path, frames)

    assert main(["analyze", str(path), "--window-size", "10", "--stride", "10"]) == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert output
    assert '"movement"' in output[0]


def test_overlay_writes_annotated_video(tmp_path, capsys):
    from main import main

    video = tmp_path / "clip.avi"
    height, width = 32, 48
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()

    records = [
        FrameRecord(
            video_id="clip",
            frame_index=i,
            timestamp=i / 10.0,
            landmarks=tuple(Landmark(0.3, 0.3, 0.0, 1.0) for _ in range(NUM_LANDMARKS)),
        )
        for i in range(3)
    ]
    jsonl = tmp_path / "clip.jsonl"
    write_jsonl(jsonl, records)
    out = tmp_path / "overlay.avi"

    assert main(["overlay", str(video), str(jsonl), "--out", str(out)]) == 0
    assert out.exists()
    assert "wrote 3 overlay frames" in capsys.readouterr().out
