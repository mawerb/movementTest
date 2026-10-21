from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentAnnotation:
    video_id: str
    start_time: float
    end_time: float
    movement: str
    quality: str
    phases: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        payload = {
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "movement": self.movement,
            "quality": self.quality,
        }
        if self.phases is not None:
            payload["phases"] = list(self.phases)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SegmentAnnotation":
        phases = payload.get("phases")
        return cls(
            video_id=payload["video_id"],
            start_time=payload["start_time"],
            end_time=payload["end_time"],
            movement=payload["movement"],
            quality=payload["quality"],
            phases=tuple(phases) if phases is not None else None,
        )
