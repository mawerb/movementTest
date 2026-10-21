from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pose_analysis.features import normalize_landmarks
from pose_analysis.landmarks import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from pose_analysis.records import FrameRecord, Landmark

SAMPLE_COUNT = 64
_VISIBILITY = 0.3
_BODY_JOINTS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)
_REQUIRED = (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER)


def frame_is_usable(landmarks: Sequence[Landmark]) -> bool:
    return all(landmarks[index].visibility > _VISIBILITY for index in _REQUIRED)


def missing_fraction(frames: Sequence[FrameRecord]) -> float:
    if not frames:
        return 1.0
    missing = sum(1 for frame in frames if not frame_is_usable(frame.landmarks))
    return missing / len(frames)


def trial_tensor(frames: Sequence[FrameRecord]) -> np.ndarray:
    if not frames:
        raise ValueError("trial has no frames")
    rows = [_frame_row(frame.landmarks) for frame in frames]
    sampled = [rows[index] for index in _sample_indices(len(rows), SAMPLE_COUNT)]
    return np.stack(sampled, axis=1).astype(np.float32)


def _frame_row(landmarks: Sequence[Landmark]) -> np.ndarray:
    row = np.zeros(25, dtype=np.float32)
    if not frame_is_usable(landmarks):
        row[24] = 1.0
        return row
    posed = normalize_landmarks(landmarks)
    cursor = 0
    for index in _BODY_JOINTS:
        row[cursor] = posed[index].x
        row[cursor + 1] = posed[index].y
        cursor += 2
    return row


def _sample_indices(length: int, count: int) -> list[int]:
    if length == 1:
        return [0] * count
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]
