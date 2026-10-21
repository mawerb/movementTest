from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
import urllib.request

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions, RunningMode
import numpy as np

from pose_analysis.records import Landmark

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DETECTOR_FILENAME = "efficientdet_lite0.tflite"
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
)

PERSON_SCORE_THRESHOLD = 0.3
CROP_PAD = 0.35
MIN_CROP_SIDE = 256


@dataclass(frozen=True)
class PersonBox:
    x: int
    y: int
    width: int
    height: int
    score: float


def ensure_detector_model() -> Path:
    path = MODEL_DIR / DETECTOR_FILENAME
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DETECTOR_URL, path)
    return path


class PersonDetector:
    def __init__(self, model_path: Path | None = None, running_mode: RunningMode = RunningMode.VIDEO):
        path = model_path or ensure_detector_model()
        options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=running_mode,
            max_results=4,
            score_threshold=PERSON_SCORE_THRESHOLD,
            category_allowlist=["person"],
        )
        self._detector = ObjectDetector.create_from_options(options)
        self._mode = running_mode

    def detect(self, frame_rgb, timestamp_ms: int = 0) -> list[PersonBox]:
        rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        image = Image(image_format=ImageFormat.SRGB, data=rgb)
        if self._mode == RunningMode.VIDEO:
            result = self._detector.detect_for_video(image, timestamp_ms)
        elif self._mode == RunningMode.IMAGE:
            result = self._detector.detect(image)
        else:
            raise RuntimeError("LIVE_STREAM mode is not supported by detect()")
        boxes = []
        for detection in result.detections:
            box = detection.bounding_box
            score = 0.0
            if detection.categories:
                score = float(detection.categories[0].score or 0.0)
            boxes.append(
                PersonBox(
                    x=int(box.origin_x),
                    y=int(box.origin_y),
                    width=int(box.width),
                    height=int(box.height),
                    score=score,
                )
            )
        boxes.sort(key=lambda item: item.score, reverse=True)
        return boxes

    def close(self) -> None:
        self._detector.close()


def expand_box(
    box: PersonBox,
    frame_width: int,
    frame_height: int,
    pad: float = CROP_PAD,
) -> tuple[int, int, int, int]:
    pad_x = int(box.width * pad)
    pad_y = int(box.height * pad)
    x0 = max(0, box.x - pad_x)
    y0 = max(0, box.y - pad_y)
    x1 = min(frame_width, box.x + box.width + pad_x)
    y1 = min(frame_height, box.y + box.height + pad_y)
    return x0, y0, x1, y1


def crop_person(frame_bgr_or_rgb, box: PersonBox, pad: float = CROP_PAD):
    height, width = frame_bgr_or_rgb.shape[:2]
    x0, y0, x1, y1 = expand_box(box, width, height, pad=pad)
    crop = frame_bgr_or_rgb[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)


def upscale_crop(crop, min_side: int = MIN_CROP_SIDE):
    height, width = crop.shape[:2]
    if min(height, width) >= min_side:
        return crop
    scale = min_side / max(1, min(height, width))
    return cv2.resize(
        crop,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def remap_landmarks(
    landmarks: Sequence[Landmark],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_width: int,
    frame_height: int,
) -> tuple[Landmark, ...]:
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    mapped = []
    for lm in landmarks:
        mapped.append(
            Landmark(
                x=(x0 + lm.x * crop_w) / frame_width,
                y=(y0 + lm.y * crop_h) / frame_height,
                z=lm.z,
                visibility=lm.visibility,
            )
        )
    return tuple(mapped)
