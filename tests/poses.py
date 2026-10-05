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
from pose_analysis.records import Landmark, NUM_LANDMARKS


def standing_pose(overrides: dict[int, Landmark] | None = None) -> tuple[Landmark, ...]:
    points = [Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]
    defaults = {
        LEFT_HIP: Landmark(-0.2, 0.0, 0.0, 1.0),
        RIGHT_HIP: Landmark(0.2, 0.0, 0.0, 1.0),
        LEFT_SHOULDER: Landmark(-0.35, 1.0, 0.0, 1.0),
        RIGHT_SHOULDER: Landmark(0.35, 1.0, 0.0, 1.0),
        LEFT_ELBOW: Landmark(-0.4, 0.45, 0.0, 1.0),
        RIGHT_ELBOW: Landmark(0.4, 0.45, 0.0, 1.0),
        LEFT_WRIST: Landmark(-0.42, 0.05, 0.0, 1.0),
        RIGHT_WRIST: Landmark(0.42, 0.05, 0.0, 1.0),
        LEFT_KNEE: Landmark(-0.2, -0.85, 0.0, 1.0),
        RIGHT_KNEE: Landmark(0.2, -0.85, 0.0, 1.0),
        LEFT_ANKLE: Landmark(-0.2, -1.7, 0.0, 1.0),
        RIGHT_ANKLE: Landmark(0.2, -1.7, 0.0, 1.0),
    }
    if overrides:
        defaults.update(overrides)
    for index, landmark in defaults.items():
        points[index] = landmark
    return tuple(points)


def stacked_release_pose() -> tuple[Landmark, ...]:
    return standing_pose(
        {
            RIGHT_SHOULDER: Landmark(0.35, 1.0, 0.0, 1.0),
            RIGHT_ELBOW: Landmark(0.36, 1.55, 0.0, 1.0),
            RIGHT_WRIST: Landmark(0.37, 2.15, 0.0, 1.0),
            LEFT_WRIST: Landmark(-0.3, 0.2, 0.0, 1.0),
        }
    )


def flared_elbow_release_pose() -> tuple[Landmark, ...]:
    return standing_pose(
        {
            RIGHT_SHOULDER: Landmark(0.35, 1.0, 0.0, 1.0),
            RIGHT_ELBOW: Landmark(0.95, 1.45, 0.0, 1.0),
            RIGHT_WRIST: Landmark(0.37, 2.15, 0.0, 1.0),
            LEFT_WRIST: Landmark(-0.3, 0.2, 0.0, 1.0),
        }
    )


def dropped_elbow_release_pose() -> tuple[Landmark, ...]:
    return standing_pose(
        {
            RIGHT_SHOULDER: Landmark(0.35, 1.0, 0.0, 1.0),
            RIGHT_ELBOW: Landmark(0.36, 1.7, 0.0, 1.0),
            RIGHT_WRIST: Landmark(0.37, 1.2, 0.0, 1.0),
            LEFT_WRIST: Landmark(-0.3, 0.2, 0.0, 1.0),
        }
    )


def unbalanced_landing_pose() -> tuple[Landmark, ...]:
    return standing_pose(
        {
            LEFT_ANKLE: Landmark(-0.9, -1.7, 0.0, 1.0),
            RIGHT_ANKLE: Landmark(0.2, -1.7, 0.0, 1.0),
        }
    )
