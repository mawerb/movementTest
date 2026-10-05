# Agent notes

Read `spec.md` before changing behavior. This repo is a research pose pipeline, not a product app.

## Stack

- Python 3.12 venv at `.venv/`
- `mediapipe==0.10.14` (do not upgrade to 1.x; Pose Landmarker init crashes on this Mac)
- OpenCV, numpy, pytest
- Run tests: `.venv/bin/python -m pytest tests -q`

## Layout

| Path | Job |
|---|---|
| `main.py` | CLI: `extract`, `analyze`, `overlay` |
| `camera.py` | Live webcam |
| `pose_analysis/` | Shared core |
| `tests/` | Unit tests; `tests/poses.py` is fake skeletons |
| `models/` | Downloaded `.task` / `.tflite` (gitignored) |
| `data/landmarks/` | Extracted JSONL (gitignored) |
| `videos/` | Local clips (mp4 gitignored) |

## How work should happen

- Tests first for behavior changes. Watch the test fail, then implement.
- Keep modules small: records, people, subject, features, detect, jumpshot, overlay.
- Do not train a model until there are trial-level labels with start/end times.
- Do not add an LLM/VLM as the scorer. Gold labels come from RA TGMD-3 0/1s.
- After extract/overlay/landmarker changes, re-extract before overlay. Old JSONL will not pick up subject-lock fixes.

## Subject lock (do not regress)

Wide gym videos have a child and an instructor. Full-frame pose only sees the adult.

Required behavior in `MediaPipePoseLandmarker.detect`:

1. Detect people, pose each crop, remap to full image.
2. **Every frame:** `select_subject` keeps the shorter leg (left hip–knee–ankle, or the right leg when the left is not visible).
3. `SubjectLock` remembers the leg length from the last switch. A pose whose leg is more than 1.5 times that length is a missing pose. A shorter leg updates the remembered length.
4. Ignore poses with fewer than 6 visible bones.
5. `SubjectLock` lives on the landmarker instance. A new `extract` run starts a new lock.

If you change this, add or update tests in `tests/test_subject.py` and `tests/test_people.py`.

## Data rules

- One `FrameRecord` per frame, 33 landmarks `[x, y, z, visibility]`.
- Missing pose stays in the file as visibility 0, so indexes stay aligned with the video.
- Overlay pairs video frame `i` with JSONL `frame_index` `i` from the **same** clip.
- Whole-video 0/1 criterion labels are not enough unless the file is already one trimmed trial.

## Commands agents should use

```bash
source .venv/bin/activate
python main.py extract videos/jump/01_Jump.mp4 --video-id clip
python main.py overlay videos/jump/01_Jump.mp4 data/landmarks/clip.jsonl --preview
python main.py overlay videos/jump/01_Jump.mp4 data/landmarks/clip.jsonl --out clip_overlay.mp4
python -m pytest tests -q
```

## Known pitfalls

- Person-crop extract is slower than full-frame pose. That is expected.
- EfficientDet can miss the child on some frames. If the only detected leg is more than 1.5 times the locked leg, that frame stays blank.
- A leftover object (speaker, bag) can get a tiny fake skeleton. Person score threshold is 0.3 and incomplete poses need ≥ 6 bones.
- `camera.py` and `main.py` must keep sharing `pose_analysis`. Do not fork landmark or overlay logic.

## Next work (when asked)

1. Import RA Excel: video, skill, trial, start/end, criterion 0/1s.
2. Train or evaluate only on those trial windows, on the locked child track.
3. Add movements one at a time. Do not start a single model that scores every TGMD-3 error.
