from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pose_analysis.records import FrameRecord


def sliding_windows(
    frames: Sequence[FrameRecord], size: int = 45, stride: int = 15
) -> list[list[FrameRecord]]:
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    if len(frames) < size:
        return []
    windows = []
    start = 0
    while start + size <= len(frames):
        windows.append(list(frames[start : start + size]))
        start += stride
    return windows


def smooth_labels(labels: Sequence[str], radius: int = 2) -> list[str]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    smoothed = []
    n = len(labels)
    for i, current in enumerate(labels):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        counts = Counter(labels[lo:hi])
        top = counts.most_common()
        winners = [label for label, count in top if count == top[0][1]]
        smoothed.append(current if current in winners else winners[0])
    return smoothed
