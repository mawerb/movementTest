import pytest

from pose_analysis.features import joint_angle, normalize_landmarks, velocities
from pose_analysis.landmarks import (
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
)
from pose_analysis.records import Landmark, NUM_LANDMARKS


def _blank() -> list[Landmark]:
    return [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]


def _standing() -> list[Landmark]:
    points = _blank()
    points[LEFT_HIP] = Landmark(0.45, 0.60, 0.0, 1.0)
    points[RIGHT_HIP] = Landmark(0.55, 0.60, 0.0, 1.0)
    points[LEFT_SHOULDER] = Landmark(0.42, 0.35, 0.0, 1.0)
    points[RIGHT_SHOULDER] = Landmark(0.58, 0.35, 0.0, 1.0)
    return points


def test_normalize_centers_hips_and_scales_torso_with_y_up():
    posed = normalize_landmarks(_standing())

    hip_x = (posed[LEFT_HIP].x + posed[RIGHT_HIP].x) / 2
    hip_y = (posed[LEFT_HIP].y + posed[RIGHT_HIP].y) / 2
    shoulder_y = (posed[LEFT_SHOULDER].y + posed[RIGHT_SHOULDER].y) / 2

    assert hip_x == pytest.approx(0.0, abs=1e-6)
    assert hip_y == pytest.approx(0.0, abs=1e-6)
    assert shoulder_y == pytest.approx(1.0, abs=1e-6)
    assert posed[LEFT_SHOULDER].visibility == 1.0


def test_knee_angle_is_90_for_right_angle():
    hip = Landmark(0.0, 1.0, 0.0, 1.0)
    knee = Landmark(0.0, 0.0, 0.0, 1.0)
    ankle = Landmark(1.0, 0.0, 0.0, 1.0)

    assert abs(joint_angle(hip, knee, ankle) - 90.0) < 1e-6


def test_velocity_is_displacement_over_dt():
    prev = _blank()
    curr = _blank()
    curr[LEFT_KNEE] = Landmark(0.5, 0.7, 0.0, 1.0)

    flow = velocities(prev, curr, dt=0.1)

    assert abs(flow[LEFT_KNEE].y - 2.0) < 1e-6
    assert abs(flow[LEFT_KNEE].x) < 1e-6
