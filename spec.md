# Pose analysis spec

## Goal

Build a pose pipeline that can extract body landmarks from gym videos of young children, keep the child when an instructor is also in frame, and later score movements against gold-standard labels.

The first working slice is video and webcam in, 33 landmarks out, skeleton overlay, and a jumpshot heuristic. The longer research target is TGMD-3 style criterion scoring (0/1 per performance criterion) using RA gold labels.

## Current system

```
video / webcam
    -> person boxes (EfficientDet-lite0)
    -> pose on each crop (MediaPipe Pose Landmarker, 33 joints)
    -> keep the person with the shorter leg on every frame
    -> FrameRecord JSONL
    -> optional overlay / jumpshot heuristic
```

Two entry points share `pose_analysis/`:

| Entry | Role |
|---|---|
| `main.py extract` | Offline video → JSONL |
| `main.py analyze` | Video or JSONL → movement/form JSON |
| `main.py overlay` | Video + JSONL → skeleton replay (`--out` and/or `--preview`) |
| `camera.py` | Live webcam coach |

## Landmark record

One JSONL line per frame, matching the original plan:

```json
{
  "video_id": "player_001_session_01",
  "frame_index": 120,
  "timestamp": 4.0,
  "landmarks": [[0.45, 0.32, -0.1, 0.9]]
}
```

- 33 landmarks, each `[x, y, z, visibility]`
- `x, y` are normalized image coordinates (origin top-left, y down)
- Missing pose → 33 zeros with visibility `0`
- Types: `pose_analysis/records.py` (`FrameRecord`, `Landmark`)

## Subject selection

Wide gym shots (for example `videos/jump/01_Jump.mp4`) show a toddler and an adult. Full-frame Pose Landmarker only sees the adult.

Rules:

1. Detect person boxes with score ≥ 0.3.
2. Expand each box, crop, upscale so the short side is at least 256 px.
3. Run pose on each crop and map joints back to the full frame.
4. On every frame, keep the skeleton with the shorter leg (hip to ankle through the knee). Use the left leg, or the right leg when the left is not visible.
5. Remember the leg length from the last time the chosen person changed. Reject a pose whose leg is more than 1.5 times that length, and store a missing pose. A shorter leg updates the remembered length.
6. Ignore incomplete detections (fewer than 6 visible bones) so a speaker or fragment cannot win.

Implemented in `pose_analysis/people.py` and `pose_analysis/subject.py`. A new extract creates a new lock.

## Features and windows

Before geometry, landmarks are converted to pose space (`pose_analysis/features.py`):

- origin at hip midpoint
- y flipped up
- scale by torso length (shoulder mid to hip mid)

Offline analysis uses overlapping windows of 45 frames, stride 15, then majority-vote label smoothing.

## Movement detection (baseline only)

`pose_analysis/detect.py` is a kinematic placeholder, not a trained model.

- bent knees + high wrist + hip bounce → `jumpshot`
- two of those three → `transition`
- fast ankles, wrists low → `running`
- else `unknown`

Named but unimplemented: `dribbling`, `passing`, `layup`.

## Jumpshot form (rule-based)

If the window is a jumpshot (`pose_analysis/jumpshot.py`):

| Check | Meaning |
|---|---|
| `elbow_alignment` | Shooting elbow should sit on the shoulder–wrist line |
| `release_stack` | Wrist above elbow above shoulder at release |
| `landing_balance` | Feet under hips, similar stance |

Phase is inferred from the window: preparation, crouch, takeoff, release, follow-through, landing.

Feedback (`pose_analysis/feedback.py`) is gated at confidence 0.65. Below that: `movement=uncertain`, no coaching text.

## Overlay

`draw_overlay` maps JSONL joints onto the source video frame and draws bones/joints. Optional `--analyze` stamps heuristic text. Frame `i` in the video pairs with `frame_index` `i` in the JSONL from that same clip.

## Labeling (agreed, not fully built)

RA gold standard for TGMD-3 stays 0/1 per performance criterion. A whole-video 0/1 is too vague unless the file is already one trimmed trial.

Required join to landmarks:

| Field | Who | Why |
|---|---|---|
| `video_id` / filename | spreadsheet | joins to JSONL |
| participant, skill, trial | RAs | TGMD-3 protocol (usually 2 trials) |
| `start_time`, `end_time` | RAs or us | the model must see the attempt, not the walk-up |
| criterion columns 0/1 | RAs after IRR | gold labels |
| `subject` | only if lock fails | leftover override |

Do not ask RAs to label jumpshot phases or basketball cues. Do not replace RA scores with a local LLM.

`SegmentAnnotation` in `pose_analysis/annotations.py` is the first schema (`video_id`, start/end, movement, quality). It is not yet wired to Excel or TGMD-3 criteria.

## What is not built

- Trained movement or criterion classifiers
- Excel / TGMD-3 import
- Trial start/end annotation UI
- Held-out-person evaluation
- Multi-movement form checkers beyond jumpshot
- MediaPipe 1.x (crashes on this Mac during Pose Landmarker init)

## Constraints

- Python 3.12, `mediapipe==0.10.14`, CPU delegate
- Models download into `models/` on first use (`pose_landmarker_lite.task`, `efficientdet_lite0.tflite`)
- Videos used in this work are wide-court TCU gym clips with a child and an instructor

## Success criteria (from the original plan, still open)

- Movement labels work on unseen people
- Segments have reliable start/end
- Form or criterion feedback is specific and traceable to a rule or a labeled example
- Low-confidence output is marked uncertain
