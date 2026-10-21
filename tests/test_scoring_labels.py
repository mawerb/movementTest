from pose_analysis.scoring.labels import gold_mark, import_label_csv, label_row, trial_gold


def test_gold_mark_thresholds_the_assessor_average():
    assert gold_mark([1, 0]) == 0
    assert gold_mark([1, 1, 0]) == 1
    assert gold_mark([1]) == 1
    assert gold_mark([]) is None


def test_label_row_ignores_blank_cells_and_sums_eight_gold_marks():
    row = label_row(
        "01 Jump",
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, None, None, None, None, None, None],
        ],
    )
    assert row.gold[0] == 0
    assert row.gold[1] == 0
    assert row.gold[2] is None
    assert row.jump_total is None
    assert row.trial_files == (None, None)
    assert trial_gold(row, 1) == (0, None, None, None)
    assert trial_gold(row, 2)[0] == 0


def test_jump_total_is_the_sum_of_eight_gold_marks():
    row = label_row("02 Jump", [[1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0]])
    assert row.gold == (1, 1, 0, 0, 1, 1, 0, 0)
    assert row.jump_total == 4


def test_import_label_csv_groups_assessors_and_leaves_files_empty():
    text = """video_name,assessor,Jump_1_1,Jump_1_2,Jump_2_1,Jump_2_2,Jump_3_1,Jump_3_2,Jump_4_1,Jump_4_2,Jump_Total
01 Jump,Jacqlyn,0,0,0,0,0,0,0,0,0
01 Jump,Victor,0,1,0,0,0,0,0,0,1
"""
    rows = import_label_csv(text)
    assert len(rows) == 1
    assert rows[0].video_name == "01 Jump"
    assert rows[0].gold[1] == 0
    assert rows[0].trial_files == (None, None)
    assert rows[0].jump_total == 0


def test_import_label_csv_keeps_trial_filenames_when_present():
    text = """video_name,assessor,Jump_1_1,Jump_1_2,Jump_2_1,Jump_2_2,Jump_3_1,Jump_3_2,Jump_4_1,Jump_4_2,trial_1_file,trial_2_file
01 Jump,Jacqlyn,1,1,1,1,1,1,1,1,01_Jump-1.mp4,01_Jump-2.mp4
"""
    rows = import_label_csv(text)
    assert rows[0].trial_files == ("01_Jump-1.mp4", "01_Jump-2.mp4")
    assert rows[0].jump_total == 8
