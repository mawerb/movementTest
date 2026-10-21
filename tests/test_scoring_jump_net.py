import torch

from pose_analysis.scoring.jump_net import HorizontalJumpNet, decide, masked_bce


def test_decide_uses_a_strict_half_cutoff():
    assert decide([0.2, 0.5, 0.9, 0.51]) == (0, 0, 1, 1)


def test_network_returns_four_probabilities():
    net = HorizontalJumpNet()
    output = net(torch.zeros(2, 25, 64))
    assert output.shape == (2, 4)
    assert torch.all(output > 0.0)
    assert torch.all(output < 1.0)


def test_masked_bce_ignores_criteria_without_gold():
    probabilities = torch.tensor([0.9, 0.1, 0.8, 0.2])
    loss = masked_bce(probabilities, [1, None, 0, None])
    expected = torch.nn.functional.binary_cross_entropy(
        probabilities[[0, 2]], torch.tensor([1.0, 0.0])
    )
    assert torch.isclose(loss, expected)


def test_masked_bce_is_zero_when_no_criterion_has_gold():
    loss = masked_bce(torch.tensor([0.2, 0.3, 0.4, 0.6]), [None, None, None, None])
    assert float(loss) == 0.0
