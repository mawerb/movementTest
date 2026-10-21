from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS

_MISSING = tuple(Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(NUM_LANDMARKS))


class PoseDetector(Protocol):
    def detect(self, frame_rgb, timestamp_ms: int) -> Sequence[Landmark] | None:
        ...


def extract_frames(
    frames: Iterable,
    landmarker: PoseDetector,
    video_id: str,
    fps: float,
) -> list[FrameRecord]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    records: list[FrameRecord] = []
    for index, frame in enumerate(frames):
        timestamp = index / fps
        timestamp_ms = int(round(timestamp * 1000.0))
        detected = landmarker.detect(frame, timestamp_ms)
        landmarks = tuple(detected) if detected is not None else _MISSING
        records.append(
            FrameRecord(
                video_id=video_id,
                frame_index=index,
                timestamp=timestamp,
                landmarks=landmarks,
            )
        )
    return records
