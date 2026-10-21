from pathlib import Path

import torch

from pose_analysis.scoring.actions import Action, TrialScore, TrialStatus
from pose_analysis.scoring.jump_net import HorizontalJumpNet, decide
from pose_analysis.scoring.sequence import missing_fraction, trial_tensor

WEIGHT_PATH = Path("models/horizontal_jump_cnn.pt")


def score_trial(action: Action, frames, weight_path: Path | None = None) -> TrialScore:
    if action is not Action.HORIZONTAL_JUMP:
        raise ValueError(f"no scorer for {action}")
    path = WEIGHT_PATH if weight_path is None else weight_path
    if not path.is_file():
        return _empty(TrialStatus.NOT_TRAINED)
    if missing_fraction(frames) > 0.5:
        return _empty(TrialStatus.UNCERTAIN)
    net = HorizontalJumpNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    features = torch.from_numpy(trial_tensor(frames)).unsqueeze(0)
    with torch.no_grad():
        probabilities = net(features).squeeze(0).tolist()
    scores = decide(probabilities)
    return TrialScore(
        status=TrialStatus.SCORED,
        scores=scores,
        probabilities=tuple(float(value) for value in probabilities),
        trial_score=sum(scores),
    )


def _empty(status: TrialStatus) -> TrialScore:
    return TrialScore(status=status, scores=None, probabilities=None, trial_score=None)
