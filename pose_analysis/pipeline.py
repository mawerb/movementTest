from __future__ import annotations

from collections.abc import Sequence

from pose_analysis.detect import classify_window
from pose_analysis.feedback import Feedback, gate_feedback
from pose_analysis.features import normalize_landmarks
from pose_analysis.jumpshot import evaluate_jumpshot
from pose_analysis.records import FrameRecord, Landmark
from pose_analysis.windows import sliding_windows, smooth_labels


def posed_frames(frames: Sequence[FrameRecord]) -> list[tuple[Landmark, ...]]:
    return [normalize_landmarks(frame.landmarks) for frame in frames]


def analyze_posed_window(window: Sequence[Sequence[Landmark]]) -> Feedback:
    movement, confidence = classify_window(window)
    issues = ()
    phase = None
    if movement == "jumpshot":
        assessment = evaluate_jumpshot(window)
        issues = assessment.issues
        phase = assessment.phase
    return gate_feedback(movement, confidence, phase, issues)


def analyze_frames(
    frames: Sequence[FrameRecord],
    window_size: int = 45,
    stride: int = 15,
) -> list[Feedback]:
    posed = posed_frames(frames)
    windows = sliding_windows(
        [frame for frame in frames], size=window_size, stride=stride
    )
    if not windows:
        if posed:
            return [analyze_posed_window(posed)]
        return []
    raw = []
    for window in windows:
        start = window[0].frame_index
        posed_window = posed[start : start + len(window)]
        raw.append(analyze_posed_window(posed_window))
    labels = smooth_labels([item.movement for item in raw])
    smoothed = []
    for feedback, label in zip(raw, labels, strict=True):
        if label == feedback.movement:
            smoothed.append(feedback)
        else:
            smoothed.append(
                gate_feedback(label, feedback.confidence, feedback.phase, feedback.issues)
            )
    return smoothed


class RollingAnalyzer:
    def __init__(self, window_size: int = 45):
        self.window_size = window_size
        self._frames: list[FrameRecord] = []

    def push(self, frame: FrameRecord) -> Feedback:
        self._frames.append(frame)
        self._frames = self._frames[-self.window_size :]
        return analyze_posed_window(posed_frames(self._frames))
