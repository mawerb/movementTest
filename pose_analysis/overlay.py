from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from pose_analysis.feedback import Feedback
from pose_analysis.landmarks import POSE_CONNECTIONS
from pose_analysis.records import FrameRecord


def draw_overlay(
    bgr: np.ndarray,
    frame: FrameRecord,
    feedback: Feedback | None = None,
) -> np.ndarray:
    canvas = bgr.copy()
    height, width = canvas.shape[:2]
    visible = [lm.visibility > 0.3 for lm in frame.landmarks]
    for start, end in POSE_CONNECTIONS:
        if visible[start] and visible[end]:
            p1 = _pixel(frame.landmarks[start], width, height)
            p2 = _pixel(frame.landmarks[end], width, height)
            cv2.line(canvas, p1, p2, (80, 200, 120), 2)
    for landmark, is_visible in zip(frame.landmarks, visible, strict=True):
        if is_visible:
            cv2.circle(canvas, _pixel(landmark, width, height), 3, (40, 220, 255), -1)

    if feedback is None:
        return canvas

    lines = [
        f"{feedback.movement}  {feedback.confidence:.2f}",
    ]
    if feedback.phase:
        lines.append(f"phase: {feedback.phase}")
    for issue in feedback.issues:
        lines.append(f"- {issue.type}: {issue.message}")
    y = 28
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def overlay_bgr_frames(
    frames: Sequence[np.ndarray],
    records: Sequence[FrameRecord],
    feedbacks: Sequence[Feedback | None] | None = None,
) -> list[np.ndarray]:
    count = min(len(frames), len(records))
    overlays = []
    for index in range(count):
        feedback = None
        if feedbacks is not None and index < len(feedbacks):
            feedback = feedbacks[index]
        overlays.append(draw_overlay(frames[index], records[index], feedback))
    return overlays


def write_overlay_video(
    video_path: str | Path,
    records: Sequence[FrameRecord],
    out_path: str | Path,
    feedbacks: Sequence[Feedback | None] | None = None,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*_fourcc_for(out_path))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot write video: {out_path}")

    written = 0
    try:
        for index, record in enumerate(records):
            ok, bgr = capture.read()
            if not ok:
                break
            feedback = None
            if feedbacks is not None and index < len(feedbacks):
                feedback = feedbacks[index]
            writer.write(draw_overlay(bgr, record, feedback))
            written += 1
    finally:
        capture.release()
        writer.release()
    return written


def preview_overlay_video(
    video_path: str | Path,
    records: Sequence[FrameRecord],
    feedbacks: Sequence[Feedback | None] | None = None,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    shown = 0
    try:
        for index, record in enumerate(records):
            ok, bgr = capture.read()
            if not ok:
                break
            feedback = None
            if feedbacks is not None and index < len(feedbacks):
                feedback = feedbacks[index]
            cv2.imshow("Pose Overlay", draw_overlay(bgr, record, feedback))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            shown += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return shown


def _fourcc_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".avi":
        return "MJPG"
    return "mp4v"


def _pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
    y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
    return x, y
