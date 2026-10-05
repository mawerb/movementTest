from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class FormIssue:
    type: str
    confidence: float
    message: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "message": self.message,
        }


@dataclass(frozen=True)
class Feedback:
    movement: str
    confidence: float
    phase: str | None
    issues: tuple[FormIssue, ...]

    def to_dict(self) -> dict:
        return {
            "movement": self.movement,
            "confidence": round(self.confidence, 3),
            "phase": self.phase,
            "issues": [issue.to_dict() for issue in self.issues],
        }


DEFAULT_CONFIDENCE_THRESHOLD = 0.65


def gate_feedback(
    movement: str,
    confidence: float,
    phase: str | None,
    issues: Sequence[FormIssue],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Feedback:
    if confidence < threshold:
        return Feedback(
            movement="uncertain",
            confidence=confidence,
            phase=None,
            issues=(),
        )
    return Feedback(
        movement=movement,
        confidence=confidence,
        phase=phase,
        issues=tuple(issues),
    )
