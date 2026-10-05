from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from pose_analysis.people import (
    PersonDetector,
    crop_person,
    remap_landmarks,
    upscale_crop,
)
from pose_analysis.records import Landmark, NUM_LANDMARKS
from pose_analysis.subject import SubjectLock

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "lite": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    "full": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
}


def ensure_model(variant: str = "lite") -> Path:
    if variant not in MODEL_FILES:
        raise ValueError(f"unknown model variant: {variant}")
    filename, url = MODEL_FILES[variant]
    path = MODEL_DIR / filename
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


class MediaPipePoseLandmarker:
    def __init__(
        self,
        model_path: Path,
        running_mode: RunningMode = RunningMode.VIDEO,
        num_poses: int = 2,
    ):
        self._mode = running_mode
        self._landmarker = PoseLandmarker.create_from_options(
            _pose_options(model_path, running_mode, num_poses)
        )
        self._crop_pose = PoseLandmarker.create_from_options(
            _pose_options(model_path, RunningMode.IMAGE, 1)
        )
        people_mode = running_mode if running_mode != RunningMode.LIVE_STREAM else RunningMode.VIDEO
        self._people = PersonDetector(running_mode=people_mode)
        self._lock = SubjectLock()

    def detect(self, frame_rgb, timestamp_ms: int) -> tuple[Landmark, ...] | None:
        rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        poses = self._poses_from_people(rgb, timestamp_ms)
        if not poses:
            poses = self._poses_from_full_frame(rgb, timestamp_ms)
        return self._lock.choose(poses)

    def _poses_from_people(self, rgb, timestamp_ms: int) -> list[tuple[Landmark, ...]]:
        height, width = rgb.shape[:2]
        poses = []
        for box in self._people.detect(rgb, timestamp_ms):
            crop, bounds = crop_person(rgb, box)
            if crop.size == 0:
                continue
            crop = upscale_crop(crop)
            pose = self._detect_image(crop)
            if pose is None:
                continue
            poses.append(remap_landmarks(pose, *bounds, width, height))
        return poses

    def _poses_from_full_frame(self, rgb, timestamp_ms: int) -> list[tuple[Landmark, ...]]:
        image = Image(image_format=ImageFormat.SRGB, data=rgb)
        if self._mode == RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        elif self._mode == RunningMode.IMAGE:
            result = self._landmarker.detect(image)
        else:
            raise RuntimeError("LIVE_STREAM mode is not supported by detect()")
        return _result_poses(result)

    def _detect_image(self, rgb) -> tuple[Landmark, ...] | None:
        image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
        result = self._crop_pose.detect(image)
        poses = _result_poses(result)
        return poses[0] if poses else None

    def close(self) -> None:
        self._landmarker.close()
        self._crop_pose.close()
        self._people.close()

    def __enter__(self) -> "MediaPipePoseLandmarker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _pose_options(model_path: Path, running_mode: RunningMode, num_poses: int) -> PoseLandmarkerOptions:
    return PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def _result_poses(result) -> list[tuple[Landmark, ...]]:
    if not result.pose_landmarks:
        return []
    poses = []
    for pose in result.pose_landmarks:
        if len(pose) != NUM_LANDMARKS:
            continue
        poses.append(tuple(_to_landmark(item) for item in pose))
    return poses


def _to_landmark(item) -> Landmark:
    visibility = item.visibility
    if visibility is None:
        visibility = item.presence if item.presence is not None else 0.0
    return Landmark(
        x=float(item.x or 0.0),
        y=float(item.y or 0.0),
        z=float(item.z or 0.0),
        visibility=float(visibility),
    )
