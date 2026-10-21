import numpy as np
import pytest

from pose_analysis.extract import extract_frames
from pose_analysis.landmarker import MediaPipePoseLandmarker, ensure_model


@pytest.mark.integration
def test_pose_landmarker_runs_on_blank_frames():
    model_path = ensure_model("lite")
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    with MediaPipePoseLandmarker(model_path) as landmarker:
        records = extract_frames(frames, landmarker, video_id="blank", fps=10)
    assert len(records) == 2
    assert records[0].video_id == "blank"
    assert len(records[0].landmarks) == 33
