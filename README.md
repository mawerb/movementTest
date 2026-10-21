# movementTest

Pose Landmarker pipeline for basketball movement detection and jumpshot form feedback.

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The lite Pose Landmarker model downloads automatically into `models/` on first run.

When two people are in frame, a person detector finds each body, pose runs on each crop, and the first frame keeps the smaller skeleton (usually the child). Later frames follow that same person by position so the overlay does not jump to the instructor.

This project pins `mediapipe==0.10.14`. Newer MediaPipe 1.x builds currently crash on this Mac during Pose Landmarker graph init.

## Live camera

```bash
python camera.py
```

Press `q` to quit. The overlay shows the current movement guess, jumpshot phase, and rule-based form issues when confidence is high enough.

## Offline video

Extract 33 landmarks per frame to JSONL:

```bash
python main.py extract path/to/clip.mp4 --video-id player_001_session_01
```

Analyze a video or a saved landmark file:

```bash
python main.py analyze path/to/clip.mp4
python main.py analyze data/landmarks/player_001_session_01.jsonl
```

Draw the skeleton back onto the original video (from extracted JSONL):

```bash
python main.py overlay path/to/clip.mp4 data/landmarks/player_001_session_01.jsonl --out clip_overlay.mp4
python main.py overlay path/to/clip.mp4 data/landmarks/player_001_session_01.jsonl --preview
```

## Tests

```bash
python -m pytest
```

Movement classification is a kinematic baseline until labeled videos exist. Jumpshot form checks are geometric: elbow under the wrist, stacked release, and landing balance.
