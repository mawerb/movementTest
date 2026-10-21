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
from pose_analysis.subject import SubjectLock, select_subject, skeleton_scale


def _body(center_x: float, scale: float) -> tuple[Landmark, ...]:
    points = [Landmark(center_x, 0.5, 0.0, 0.0) for _ in range(NUM_LANDMARKS)]
    joints = {
        LEFT_SHOULDER: (-0.35, 1.0),
        RIGHT_SHOULDER: (0.35, 1.0),
        LEFT_ELBOW: (-0.45, 0.5),
        RIGHT_ELBOW: (0.45, 0.5),
        LEFT_WRIST: (-0.5, 0.05),
        RIGHT_WRIST: (0.5, 0.05),
        LEFT_HIP: (-0.2, 0.0),
        RIGHT_HIP: (0.2, 0.0),
        LEFT_KNEE: (-0.2, -0.7),
        RIGHT_KNEE: (0.2, -0.7),
        LEFT_ANKLE: (-0.2, -1.4),
        RIGHT_ANKLE: (0.2, -1.4),
    }
    for index, (x, y) in joints.items():
        points[index] = Landmark(center_x + x * scale, 0.45 - y * scale, 0.0, 1.0)
    return tuple(points)


def test_child_skeleton_has_smaller_joint_distances_than_adult():
    child = _body(0.3, scale=0.12)
    adult = _body(0.7, scale=0.28)
    assert skeleton_scale(child) < skeleton_scale(adult)


def test_select_subject_keeps_the_smaller_of_two_people():
    child = _body(0.3, scale=0.12)
    adult = _body(0.7, scale=0.28)
    chosen = select_subject([adult, child])
    assert chosen is child


def test_select_subject_uses_leg_length_not_arm_span():
    child = list(_body(0.3, scale=0.12))
    adult = _body(0.7, scale=0.28)
    child[LEFT_WRIST] = Landmark(0.3 - 2.0, 0.45, 0.0, 1.0)
    child[RIGHT_WRIST] = Landmark(0.3 + 2.0, 0.45, 0.0, 1.0)
    assert select_subject([adult, tuple(child)]) == tuple(child)


def test_select_subject_returns_none_when_empty():
    assert select_subject([]) is None


def test_select_subject_ignores_tiny_incomplete_detections():
    adult = _body(0.7, scale=0.28)
    blob = [Landmark(0.1, 0.1, 0.0, 0.0) for _ in range(NUM_LANDMARKS)]
    blob[LEFT_SHOULDER] = Landmark(0.10, 0.10, 0.0, 1.0)
    blob[RIGHT_SHOULDER] = Landmark(0.11, 0.10, 0.0, 1.0)
    chosen = select_subject([tuple(blob), adult])
    assert chosen is adult


def test_lock_picks_shorter_leg_after_the_child_moves():
    lock = SubjectLock()
    child = _body(0.3, scale=0.12)
    adult = _body(0.7, scale=0.28)
    assert lock.choose([adult, child]) == child

    child_moved = _body(0.34, scale=0.12)
    assert lock.choose([adult, child_moved]) == child_moved


def test_lock_rejects_a_leg_longer_than_one_and_a_half_times():
    lock = SubjectLock()
    child = _body(0.50, scale=0.12)
    grown = _body(0.50, scale=0.12 * 1.4)
    instructor = _body(0.56, scale=0.12 * 1.6)
    assert lock.choose([child]) == child
    assert lock.choose([grown]) == grown
    assert lock.choose([instructor]) is None
    assert lock.choose([instructor, child]) == child


def test_lock_retargets_when_a_smaller_person_appears():
    lock = SubjectLock()
    adult = _body(0.7, scale=0.28)
    child = _body(0.3, scale=0.12)
    assert lock.choose([adult]) == adult
    assert lock.choose([adult, child]) == child
    assert lock.choose([adult]) is None


def test_lock_keeps_shorter_leg_when_instructor_hip_is_closer():
    lock = SubjectLock()
    child = _body(0.50, scale=0.12)
    adult = _body(0.56, scale=0.28)
    assert lock.choose([adult, child]) == child

    child_after_jump = _body(0.62, scale=0.12)
    assert lock.choose([adult, child_after_jump]) == child_after_jump
