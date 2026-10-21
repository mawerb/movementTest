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
