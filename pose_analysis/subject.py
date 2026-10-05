from __future__ import annotations

import math
from collections.abc import Sequence

from pose_analysis.landmarks import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    POSE_CONNECTIONS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
)
from pose_analysis.records import Landmark

_MIN_BONES = 6
_VISIBILITY = 0.3
_MAX_LEG_GROWTH = 1.5
_LEGS = (
    (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
)


def skeleton_scale(landmarks: Sequence[Landmark]) -> float | None:
    lengths = []
    for start, end in POSE_CONNECTIONS:
        a = landmarks[start]
        b = landmarks[end]
        if a.visibility <= _VISIBILITY or b.visibility <= _VISIBILITY:
            continue
        lengths.append(math.hypot(a.x - b.x, a.y - b.y))
    if len(lengths) < _MIN_BONES:
        return None
    return sum(lengths) / len(lengths)


def leg_length(landmarks: Sequence[Landmark]) -> float | None:
    """Length of one leg, hip to ankle through the knee. Left leg, else right."""
    for hip, knee, ankle in _LEGS:
        thigh = _bone_length(landmarks, hip, knee)
        shin = _bone_length(landmarks, knee, ankle)
        if thigh is not None and shin is not None:
            return thigh + shin
    return None


def select_subject(
    poses: Sequence[Sequence[Landmark]],
) -> tuple[Landmark, ...] | None:
    scored: list[tuple[float, Sequence[Landmark]]] = []
    for pose in poses:
        if skeleton_scale(pose) is None:
            continue
        length = leg_length(pose)
        if length is not None:
            scored.append((length, pose))
    if not scored:
        return None
    chosen = min(scored, key=lambda item: item[0])[1]
    return tuple(chosen)


class SubjectLock:
    """Keep the shorter leg, and refuse a person whose leg grew past the last switch."""

    def __init__(self, max_growth: float = _MAX_LEG_GROWTH):
        self._leg: float | None = None
        self._max_growth = max_growth

    def choose(
        self, poses: Sequence[Sequence[Landmark]]
    ) -> tuple[Landmark, ...] | None:
        scored: list[tuple[float, tuple[Landmark, ...]]] = []
        for pose in poses:
            if skeleton_scale(pose) is None:
                continue
            length = leg_length(pose)
            if length is None:
                continue
            if self._leg is not None and length > self._max_growth * self._leg:
                continue
            scored.append((length, tuple(pose)))
        if not scored:
            return None
        length, chosen = min(scored, key=lambda item: item[0])
        if self._leg is None or length < self._leg:
            self._leg = length
        return chosen


def _bone_length(landmarks: Sequence[Landmark], start: int, end: int) -> float | None:
    a = landmarks[start]
    b = landmarks[end]
    if a.visibility <= _VISIBILITY or b.visibility <= _VISIBILITY:
        return None
    return math.hypot(a.x - b.x, a.y - b.y)
