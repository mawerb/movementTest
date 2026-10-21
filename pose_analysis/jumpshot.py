from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pose_analysis.features import joint_angle, midpoint
from pose_analysis.feedback import FormIssue
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
from pose_analysis.records import Landmark

_EPS = 1e-6
_ELBOW_LINE_RATIO = 0.35
_LANDING_IMBALANCE = 0.35
_HIGH_WRIST = 1.5
_BENT_KNEE = 150.0

_ARM = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}


@dataclass(frozen=True)
class JumpshotResult:
    phase: str
    issues: tuple[FormIssue, ...]


def shooting_side(landmarks: Sequence[Landmark]) -> str:
    if landmarks[LEFT_WRIST].y >= landmarks[RIGHT_WRIST].y:
        return "left"
    return "right"


def check_elbow_alignment(landmarks: Sequence[Landmark]) -> FormIssue | None:
    side = shooting_side(landmarks)
    shoulder, elbow, wrist = (_ARM[side][0], _ARM[side][1], _ARM[side][2])
    dist = _point_line_distance_xy(landmarks[elbow], landmarks[shoulder], landmarks[wrist])
    forearm = _xy_distance(landmarks[elbow], landmarks[wrist])
    ratio = dist / max(forearm, _EPS)
    if ratio <= _ELBOW_LINE_RATIO:
        return None
    return FormIssue(
        type="elbow_alignment",
        confidence=min(0.95, 0.55 + ratio),
        message="Keep the shooting elbow more directly under the wrist.",
    )


def check_release_stack(landmarks: Sequence[Landmark]) -> FormIssue | None:
    side = shooting_side(landmarks)
    shoulder, elbow, wrist = (_ARM[side][0], _ARM[side][1], _ARM[side][2])
    if (
        landmarks[wrist].y > landmarks[elbow].y
        and landmarks[elbow].y > landmarks[shoulder].y
    ):
        return None
    return FormIssue(
        type="release_stack",
        confidence=0.8,
        message="Raise the shooting wrist above the elbow so the arm stays stacked at release.",
    )


def check_landing_balance(landmarks: Sequence[Landmark]) -> FormIssue | None:
    left_offset = abs(landmarks[LEFT_ANKLE].x - landmarks[LEFT_HIP].x)
    right_offset = abs(landmarks[RIGHT_ANKLE].x - landmarks[RIGHT_HIP].x)
    imbalance = abs(left_offset - right_offset)
    foot_height = abs(landmarks[LEFT_ANKLE].y - landmarks[RIGHT_ANKLE].y)
    if imbalance <= _LANDING_IMBALANCE and foot_height <= 0.25:
        return None
    return FormIssue(
        type="landing_balance",
        confidence=min(0.95, 0.55 + imbalance),
        message="Land more evenly with both feet under the hips.",
    )


def evaluate_jumpshot(window: Sequence[Sequence[Landmark]]) -> JumpshotResult:
    if not window:
        raise ValueError("window must not be empty")
    release_index = max(range(len(window)), key=lambda i: _max_wrist_y(window[i]))
    landing_index = len(window) - 1
    issues: list[FormIssue] = []
    elbow = check_elbow_alignment(window[release_index])
    stack = check_release_stack(window[release_index])
    landing = check_landing_balance(window[landing_index])
    if elbow:
        issues.append(elbow)
    if stack:
        issues.append(stack)
    if landing:
        issues.append(landing)
    return JumpshotResult(phase=jumpshot_phase(window), issues=tuple(issues))


def jumpshot_phase(window: Sequence[Sequence[Landmark]]) -> str:
    if not window:
        return "preparation"
    last = window[-1]
    wrist_series = [_max_wrist_y(frame) for frame in window]
    hip_series = [_hip_y(frame) for frame in window]
    knee_series = [_mean_knee_angle(frame) for frame in window]
    peak_index = max(range(len(wrist_series)), key=lambda i: wrist_series[i])
    peak_wrist = wrist_series[peak_index]
    last_index = len(window) - 1
    last_knee = knee_series[-1]
    last_hip = hip_series[-1]
    hip_vy = 0.0
    if len(hip_series) >= 2:
        hip_vy = hip_series[-1] - hip_series[-2]

    if peak_wrist >= _HIGH_WRIST and last_index > peak_index:
        if wrist_series[-1] >= peak_wrist - 0.25:
            return "follow_through"
        return "landing"
    if peak_wrist >= _HIGH_WRIST and last_index == peak_index:
        return "release"
    if last_knee < _BENT_KNEE and last_hip < -0.1:
        return "crouch"
    if hip_vy > 0.05 and last_knee > min(knee_series) + 5:
        return "takeoff"
    return "preparation"


def _max_wrist_y(landmarks: Sequence[Landmark]) -> float:
    return max(landmarks[LEFT_WRIST].y, landmarks[RIGHT_WRIST].y)


def _hip_y(landmarks: Sequence[Landmark]) -> float:
    return midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])[1]


def _mean_knee_angle(landmarks: Sequence[Landmark]) -> float:
    left = joint_angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
    right = joint_angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
    return (left + right) / 2.0


def _xy_distance(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _point_line_distance_xy(point: Landmark, start: Landmark, end: Landmark) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length < _EPS:
        return math.hypot(point.x - start.x, point.y - start.y)
    return abs(dx * (point.y - start.y) - dy * (point.x - start.x)) / length
