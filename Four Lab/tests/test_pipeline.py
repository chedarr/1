# -*- coding: utf-8 -*-
import numpy as np
import pytest

from src import config
from src.features import make_windows
from src.loader import clean, nan_runs
from src.split import assign_windows, chronological_ranges


def test_nan_runs_detects_series():
    mask = np.array([False, True, True, False, True], dtype=bool)
    runs = nan_runs(mask)
    assert runs == [{"start": 1, "end": 2, "length": 2}, {"start": 4, "end": 4, "length": 1}]


def test_clean_drops_rows_and_reports_long_runs(frame):
    cleaned, info = clean(frame)
    assert len(cleaned) == 38
    assert info["n_rows_dropped"] == 2
    assert info["nan_runs_long"] == 0
    assert info["nan_runs_total"] == 1


def test_clean_marks_long_run():
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=30, freq="10min")
    values = {name: np.arange(30, dtype=float) for name in config.FEATURE_COLUMNS}
    values[config.TARGET_FEATURE][5:15] = np.nan
    frame = pd.DataFrame(values, index=index)
    _, info = clean(frame)
    assert info["nan_runs_long"] == 1
    assert info["longest_run_steps"] == 10


def test_windows_shapes(frame):
    cleaned, _ = clean(frame)
    result = make_windows(cleaned, lookback=4, horizon=3, stride=1)
    assert result["X"].shape[1:] == (4, 5)
    assert result["y"].shape[1] == 3
    assert result["X"].shape[0] == result["y"].shape[0]


def test_windows_skip_gap():
    import pandas as pd

    index = pd.date_range("2020-01-01", periods=20, freq="10min")
    values = {name: np.arange(20, dtype=float) for name in config.FEATURE_COLUMNS}
    frame = pd.DataFrame(values, index=index)
    # вырезаем 3 шага => скачок времени, окна через него недопустимы
    frame = frame.drop(frame.index[8:11])
    result = make_windows(frame, lookback=4, horizon=2, stride=1)
    assert result["skipped_windows"] > 0
    assert result["starts"].size + result["skipped_windows"] == len(range(0, len(frame) - 5))


def test_chronological_ranges_cover_all_rows():
    ranges = chronological_ranges(1000)
    assert ranges["train"][0] == 0
    assert ranges["test"][1] == 1000
    assert ranges["val"][0] == ranges["train"][1]


def test_assign_windows_never_crosses_boundaries():
    starts = np.arange(0, 100, dtype=np.int64)
    ranges = {"train": (0, 50), "val": (50, 70), "test": (70, 100)}
    masks = assign_windows(starts, lookback=5, horizon=5, ranges=ranges)
    span = 10
    for name, (begin, end) in ranges.items():
        selected = starts[masks[name]]
        assert np.all(selected >= begin)
        assert np.all(selected + span <= end)
    total = sum(int(m.sum()) for m in masks.values())
    assert total < len(starts)


def test_config_target_is_feature():
    assert config.TARGET_FEATURE in config.FEATURE_COLUMNS
    assert len(config.CHANNELS) == 5
