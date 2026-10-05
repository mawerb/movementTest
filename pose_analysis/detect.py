from __future__ import annotations

from collections.abc import Sequence

from pose_analysis.features import joint_angle, midpoint
from pose_analysis.landmarks import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_WRIST,
)
from pose_analysis.records import Landmark

_HIGH_WRIST = 1.5
_BENT_KNEE = 150.0
_HIP_TRAVEL = 0.08


def classify_window(window: Sequence[Sequence[Landmark]]) -> tuple[str, float]:
    if not window:
        return "unknown", 0.0

    knee_min = min(_mean_knee_angle(frame) for frame in window)
    wrist_max = max(_max_wrist_y(frame) for frame in window)
    hips = [_hip_y(frame) for frame in window]
    hip_travel = max(hips) - min(hips)

    jumpshot_cues = 0
    if knee_min < _BENT_KNEE:
        jumpshot_cues += 1
    if wrist_max >= _HIGH_WRIST:
        jumpshot_cues += 1
    if hip_travel >= _HIP_TRAVEL:
        jumpshot_cues += 1

    if jumpshot_cues == 3:
        return "jumpshot", 0.7 + 0.1 * min(wrist_max - _HIGH_WRIST, 1.0)
    if jumpshot_cues == 2:
        return "transition", 0.55

    ankle_speed = _mean_ankle_speed(window)
    if ankle_speed > 1.5 and wrist_max < 1.2:
        return "running", min(0.8, 0.5 + ankle_speed / 10.0)

    return "unknown", 0.2 + 0.1 * jumpshot_cues


def _max_wrist_y(landmarks: Sequence[Landmark]) -> float:
    return max(landmarks[LEFT_WRIST].y, landmarks[RIGHT_WRIST].y)


def _hip_y(landmarks: Sequence[Landmark]) -> float:
    return midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])[1]


def _mean_knee_angle(landmarks: Sequence[Landmark]) -> float:
    left = joint_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
    right = joint_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
    return (left + right) / 2.0


def _mean_ankle_speed(window: Sequence[Sequence[Landmark]]) -> float:
    if len(window) < 2:
        return 0.0
    total = 0.0
    count = 0
    for prev, curr in zip(window, window[1:]):
        for index in (LEFT_ANKLE, RIGHT_ANKLE):
            dx = curr[index].x - prev[index].x
            dy = curr[index].y - prev[index].y
            total += (dx * dx + dy * dy) ** 0.5
            count += 1
    return total / max(count, 1)
