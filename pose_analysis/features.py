from __future__ import annotations

import math
from collections.abc import Sequence

from pose_analysis.landmarks import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from pose_analysis.records import Landmark, NUM_LANDMARKS

_EPS = 1e-6


def midpoint(a: Landmark, b: Landmark) -> tuple[float, float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0)


def _distance(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def torso_scale(landmarks: Sequence[Landmark]) -> float:
    hip = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    shoulder = midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    length = _distance(*hip, *shoulder)
    if length > _EPS:
        return length
    width = _distance(
        landmarks[LEFT_SHOULDER].x,
        landmarks[LEFT_SHOULDER].y,
        landmarks[LEFT_SHOULDER].z,
        landmarks[RIGHT_SHOULDER].x,
        landmarks[RIGHT_SHOULDER].y,
        landmarks[RIGHT_SHOULDER].z,
    )
    return max(width, _EPS)


def normalize_landmarks(landmarks: Sequence[Landmark]) -> tuple[Landmark, ...]:
    if len(landmarks) != NUM_LANDMARKS:
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
    origin_x, origin_y, origin_z = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    scale = torso_scale(landmarks)
    posed = []
    for lm in landmarks:
        posed.append(
            Landmark(
                x=(lm.x - origin_x) / scale,
                y=-(lm.y - origin_y) / scale,
                z=(lm.z - origin_z) / scale,
                visibility=lm.visibility,
            )
        )
    return tuple(posed)


def joint_angle(a: Landmark, vertex: Landmark, c: Landmark) -> float:
    v1 = (a.x - vertex.x, a.y - vertex.y, a.z - vertex.z)
    v2 = (c.x - vertex.x, c.y - vertex.y, c.z - vertex.z)
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def velocities(
    previous: Sequence[Landmark], current: Sequence[Landmark], dt: float
) -> tuple[Landmark, ...]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    flow = []
    for prev, curr in zip(previous, current, strict=True):
        flow.append(
            Landmark(
                x=(curr.x - prev.x) / dt,
                y=(curr.y - prev.y) / dt,
                z=(curr.z - prev.z) / dt,
                visibility=min(prev.visibility, curr.visibility),
            )
        )
    return tuple(flow)
