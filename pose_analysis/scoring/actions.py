from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    HORIZONTAL_JUMP = "horizontal_jump"


class TrialStatus(Enum):
    SCORED = "scored"
    UNCERTAIN = "uncertain"
    NOT_TRAINED = "not_trained"


@dataclass(frozen=True)
class TrialScore:
    status: TrialStatus
    scores: tuple[int, int, int, int] | None
    probabilities: tuple[float, float, float, float] | None
    trial_score: int | None
