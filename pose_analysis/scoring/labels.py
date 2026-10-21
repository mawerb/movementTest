from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO

JUMP_COLUMNS = (
    "Jump_1_1",
    "Jump_1_2",
    "Jump_2_1",
    "Jump_2_2",
    "Jump_3_1",
    "Jump_3_2",
    "Jump_4_1",
    "Jump_4_2",
)


def gold_mark(scores: Sequence[int]) -> int | None:
    if not scores:
        return None
    average = sum(scores) / len(scores)
    return 1 if average > 0.5 else 0


@dataclass(frozen=True)
class JumpLabelRow:
    video_name: str
    assessor_marks: tuple[tuple[int | None, ...], ...]
    gold: tuple[int | None, ...]
    trial_files: tuple[str | None, str | None]
    jump_total: int | None


def label_row(
    video_name: str,
    assessor_marks: Sequence[Sequence[int | None]],
    trial_files: tuple[str | None, str | None] = (None, None),
) -> JumpLabelRow:
    stored = tuple(tuple(marks) for marks in assessor_marks)
    for marks in stored:
        if len(marks) != 8:
            raise ValueError(f"expected 8 jump marks, got {len(marks)}")
    gold = []
    for index in range(8):
        values = [marks[index] for marks in stored]
        if any(value is None for value in values):
            gold.append(None)
            continue
        gold.append(gold_mark(values))
    gold_tuple = tuple(gold)
    total = sum(gold_tuple) if all(mark is not None for mark in gold_tuple) else None
    return JumpLabelRow(
        video_name=video_name,
        assessor_marks=stored,
        gold=gold_tuple,
        trial_files=trial_files,
        jump_total=total,
    )


def trial_gold(
    row: JumpLabelRow, trial: int
) -> tuple[int | None, int | None, int | None, int | None]:
    if trial not in (1, 2):
        raise ValueError(f"trial must be 1 or 2, got {trial}")
    offset = trial - 1
    return (
        row.gold[offset],
        row.gold[2 + offset],
        row.gold[4 + offset],
        row.gold[6 + offset],
    )


def import_label_csv(text: str) -> tuple[JumpLabelRow, ...]:
    reader = csv.DictReader(StringIO(text))
    grouped: dict[str, list[tuple[int | None, ...]]] = defaultdict(list)
    files: dict[str, tuple[str | None, str | None]] = {}
    order: list[str] = []
    for raw in reader:
        name = raw["video_name"].strip()
        if name not in grouped:
            order.append(name)
        grouped[name].append(tuple(_cell(raw.get(column)) for column in JUMP_COLUMNS))
        trial_files = (
            _filename(raw.get("trial_1_file")),
            _filename(raw.get("trial_2_file")),
        )
        if any(trial_files):
            files[name] = trial_files
    return tuple(
        label_row(name, grouped[name], files.get(name, (None, None))) for name in order
    )


def _cell(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    number = int(stripped)
    if number not in (0, 1):
        raise ValueError(f"jump mark must be 0 or 1, got {number}")
    return number


def _filename(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
