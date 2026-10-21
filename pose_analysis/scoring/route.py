from pathlib import Path

from pose_analysis.scoring.actions import Action, TrialScore, TrialStatus

WEIGHT_PATH = Path("models/horizontal_jump_cnn.pt")


def score_trial(action: Action, frames, weight_path: Path | None = None) -> TrialScore:
    if action is not Action.HORIZONTAL_JUMP:
        raise ValueError(f"no scorer for {action}")
    path = WEIGHT_PATH if weight_path is None else weight_path
    if not path.is_file():
        return TrialScore(
            status=TrialStatus.NOT_TRAINED,
            scores=None,
            probabilities=None,
            trial_score=None,
        )
    raise RuntimeError("weights exist but the jump network is not wired yet")
