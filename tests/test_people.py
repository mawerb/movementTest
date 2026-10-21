from pose_analysis.people import PersonBox, expand_box, remap_landmarks
from pose_analysis.records import Landmark, NUM_LANDMARKS


def test_expand_box_pads_and_clamps_to_frame():
    box = PersonBox(x=10, y=10, width=20, height=40, score=0.9)
    x0, y0, x1, y1 = expand_box(box, frame_width=100, frame_height=80, pad=0.5)
    assert (x0, y0, x1, y1) == (0, 0, 40, 70)


def test_remap_landmarks_converts_crop_coords_into_full_image():
    crop_landmarks = tuple(
        Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)
    )
    mapped = remap_landmarks(
        crop_landmarks,
        x0=100,
        y0=50,
        x1=300,
        y1=150,
        frame_width=400,
        frame_height=200,
    )
    assert mapped[0].x == 0.25
    assert mapped[0].y == 0.25

    center = tuple(Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(NUM_LANDMARKS))
    mapped_center = remap_landmarks(center, 100, 50, 300, 150, 400, 200)
    assert mapped_center[0].x == 0.5
    assert mapped_center[0].y == 0.5
