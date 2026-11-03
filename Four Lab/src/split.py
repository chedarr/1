# -*- coding: utf-8 -*-
"""Хронологический split 70/15/15 и привязка окон к сплитам."""
from __future__ import annotations

import numpy as np

from . import config


def chronological_ranges(n_rows: int, fractions: tuple[float, ...] = config.SPLIT_FRACTIONS) -> dict:
    """Границы сплитов по строкам ряда (без перемешивания)."""
    n_train = int(round(n_rows * fractions[0]))
    n_val = int(round(n_rows * fractions[1]))
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_rows),
    }


def assign_windows(starts: np.ndarray, lookback: int, horizon: int, ranges: dict) -> dict:
    """Окно целиком лежит внутри одного сплита — иначе выкидывается."""
    span = lookback + horizon
    masks = {}
    for name, (begin, end) in ranges.items():
        masks[name] = (starts >= begin) & (starts + span <= end)
    return masks


def split_info(n_rows: int, ranges: dict, starts: np.ndarray, masks: dict,
               lookback: int, horizon: int, stride: int) -> dict:
    """Сводка по сплитам для JSON."""
    info = {
        "split": "chronological",
        "fractions": list(config.SPLIT_FRACTIONS),
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "n_rows": int(n_rows),
        "ranges": {k: [int(v[0]), int(v[1])] for k, v in ranges.items()},
        "windows": {k: int(m.sum()) for k, m in masks.items()},
        "windows_total": int(len(starts)),
        "windows_dropped_by_boundaries": int(len(starts) - sum(int(m.sum()) for m in masks.values())),
    }
    return info
