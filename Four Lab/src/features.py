# -*- coding: utf-8 -*-
"""Окна (lookback, 5 каналов) -> (horizon,) для целевой переменной."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_windows(
    df: pd.DataFrame,
    lookback: int = config.LOOKBACK,
    horizon: int = config.HORIZON,
    stride: int = config.STRIDE,
) -> dict:
    """Собирает окна, пропуская те, что пересекают разрыв во времени.

    Ряд очищен от NaN, поэтому разрыв проявляется как скачок timestamp.
    Окно валидно, только если все его шаги идут подряд с шагом 10 минут.
    """
    values = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    stamps = df.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    n_rows = len(df)
    span = lookback + horizon
    expected = (span - 1) * config.STEP_MINUTES * 60 * 1_000_000_000

    candidates = np.arange(0, n_rows - span + 1, stride, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("Ряд короче одного окна")
    last = candidates + span - 1
    valid = (stamps[last] - stamps[candidates]) == expected
    starts = candidates[valid]

    x = np.stack([values[i : i + lookback] for i in starts]).astype(np.float32)
    target_idx = config.FEATURE_COLUMNS.index(config.TARGET_FEATURE)
    y = np.stack([values[i + lookback : i + span, target_idx] for i in starts]).astype(np.float32)
    origins = df.index[starts + lookback - 1]
    skipped = int(candidates.size - starts.size)
    return {
        "X": x,
        "y": y,
        "starts": starts,
        "origins": origins,
        "skipped_windows": skipped,
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "target_index": target_idx,
    }
