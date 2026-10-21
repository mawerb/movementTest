from pathlib import Path

import pytest

from pose_analysis.scoring.actions import Action, TrialStatus
from pose_analysis.scoring.route import score_trial


def test_horizontal_jump_without_weights_is_not_trained(tmp_path):
    result = score_trial(Action.HORIZONTAL_JUMP, [], weight_path=tmp_path / "missing.pt")
    assert result.status is TrialStatus.NOT_TRAINED
    assert result.scores is None
    assert result.probabilities is None
    assert result.trial_score is None


def test_other_actions_are_refused():
    with pytest.raises(ValueError, match="no scorer"):
        score_trial("skip", [])


import torch

from pose_analysis.records import FrameRecord, Landmark, NUM_LANDMARKS
from pose_analysis.scoring.jump_net import HorizontalJumpNet
from tests.poses import standing_pose


def _clip(count: int, missing_at: set[int]) -> list[FrameRecord]:
    blank = tuple(Landmark(0.0, 0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS))
    frames = []
    for index in range(count):
        pose = blank if index in missing_at else standing_pose()
        frames.append(FrameRecord("01 Jump", index, index / 30.0, pose))
    return frames


def _save_known_net(path):
    net = HorizontalJumpNet()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.head.bias.copy_(torch.tensor([0.0, 0.0, 2.0, -2.0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def test_too_many_missing_frames_are_uncertain(tmp_path):
    weights = tmp_path / "horizontal_jump_cnn.pt"
    _save_known_net(weights)
    result = score_trial(Action.HORIZONTAL_JUMP, _clip(4, {0, 1, 2}), weight_path=weights)
    assert result.status is TrialStatus.UNCERTAIN
    assert result.scores is None
    assert result.trial_score is None


def test_scored_trial_sums_the_four_marks(tmp_path):
    weights = tmp_path / "horizontal_jump_cnn.pt"
    _save_known_net(weights)
    result = score_trial(Action.HORIZONTAL_JUMP, _clip(4, set()), weight_path=weights)
    assert result.status is TrialStatus.SCORED
    assert result.scores == (0, 0, 1, 0)
    assert result.trial_score == 1
    assert result.probabilities is not None
    assert len(result.probabilities) == 4
