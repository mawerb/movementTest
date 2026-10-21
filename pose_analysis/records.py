from dataclasses import dataclass
import json


NUM_LANDMARKS = 33


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_index: int
    timestamp: float
    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS:
            raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(self.landmarks)}")
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        if self.world_landmarks is not None:
            if len(self.world_landmarks) != NUM_LANDMARKS:
                raise ValueError(
                    f"expected {NUM_LANDMARKS} world landmarks, got {len(self.world_landmarks)}"
                )
            object.__setattr__(self, "world_landmarks", tuple(self.world_landmarks))

    def to_dict(self) -> dict:
        payload = {
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "landmarks": [
                [lm.x, lm.y, lm.z, lm.visibility] for lm in self.landmarks
            ],
        }
        if self.world_landmarks is not None:
            payload["world_landmarks"] = [
                [lm.x, lm.y, lm.z, lm.visibility] for lm in self.world_landmarks
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FrameRecord":
        world = payload.get("world_landmarks")
        return cls(
            video_id=payload["video_id"],
            frame_index=payload["frame_index"],
            timestamp=payload["timestamp"],
            landmarks=_landmarks_from_rows(payload["landmarks"]),
            world_landmarks=_landmarks_from_rows(world) if world is not None else None,
        )


def _landmarks_from_rows(rows: list) -> tuple[Landmark, ...]:
    return tuple(Landmark(x=row[0], y=row[1], z=row[2], visibility=row[3]) for row in rows)


def dumps_jsonl(frames: list[FrameRecord]) -> str:
    return "".join(json.dumps(frame.to_dict()) + "\n" for frame in frames)


def loads_jsonl(text: str) -> list[FrameRecord]:
    frames = []
    for line in text.splitlines():
        if not line.strip():
            continue
        frames.append(FrameRecord.from_dict(json.loads(line)))
    return frames


def write_jsonl(path, frames: list[FrameRecord]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dumps_jsonl(frames))


def read_jsonl(path) -> list[FrameRecord]:
    with open(path, encoding="utf-8") as handle:
        return loads_jsonl(handle.read())
