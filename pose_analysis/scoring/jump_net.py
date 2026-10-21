from collections.abc import Sequence

import torch
from torch import nn


class HorizontalJumpNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(25, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.conv(features).squeeze(-1)
        return torch.sigmoid(self.head(pooled))


def decide(probabilities: Sequence[float]) -> tuple[int, int, int, int]:
    if len(probabilities) != 4:
        raise ValueError(f"expected 4 probabilities, got {len(probabilities)}")
    scores = tuple(1 if probability > 0.5 else 0 for probability in probabilities)
    return scores  # type: ignore[return-value]


def masked_bce(probabilities: torch.Tensor, gold: Sequence[int | None]) -> torch.Tensor:
    if probabilities.shape[-1] != 4 or len(gold) != 4:
        raise ValueError("expected 4 criteria")
    indexes = [index for index, mark in enumerate(gold) if mark is not None]
    if not indexes:
        return probabilities.new_zeros(())
    chosen = probabilities[..., indexes]
    target = probabilities.new_tensor([float(gold[index]) for index in indexes])
    return torch.nn.functional.binary_cross_entropy(chosen, target)
