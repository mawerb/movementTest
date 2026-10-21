from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    return capture, fps


def iter_rgb_frames(capture: cv2.VideoCapture) -> Iterator:
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
