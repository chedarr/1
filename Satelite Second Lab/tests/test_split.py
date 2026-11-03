# -*- coding: utf-8 -*-
"""Разбиение по экспериментам: отсутствие утечек и воспроизводимость."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.loader_adapt import load_split
from src.split_adapt import (
    GROUP_COL,
    assert_disjoint,
    group_kfold_indices,
    groups,
    holdout_by_experiment,
)


def test_no_overlap_between_train_and_test():
    train, test = load_split()
    assert set(train.experiments).isdisjoint(set(test.experiments))
    assert_disjoint(train.meta, test.meta)


def test_overlap_is_detected():
    meta = pd.DataFrame({GROUP_COL: ["1", "2", "3"]})
    with pytest.raises(ValueError):
        assert_disjoint(meta, pd.DataFrame({GROUP_COL: ["3", "4"]}))


def test_holdout_is_deterministic_and_disjoint():
    train, _ = load_split()
    idx_fit, idx_val, val_groups = holdout_by_experiment(train.meta, 2, seed=42)
    again = holdout_by_experiment(train.meta, 2, seed=42)
    assert np.array_equal(idx_fit, again[0])
    assert np.array_equal(idx_val, again[1])
    assert len(val_groups) == 2
    assert set(groups(train.meta)[idx_fit]).isdisjoint(set(groups(train.meta)[idx_val]))
    assert len(idx_fit) + len(idx_val) == train.n_samples


def test_holdout_covers_all_rows():
    train, _ = load_split()
    idx_fit, idx_val, _ = holdout_by_experiment(train.meta, 3, seed=7)
    assert sorted(np.concatenate([idx_fit, idx_val]).tolist()) == list(range(train.n_samples))


def test_kfold_does_not_split_experiments():
    train, _ = load_split()
    g = groups(train.meta)
    folds = group_kfold_indices(train.meta, n_splits=4)
    assert len(folds) == 4
    seen: list[str] = []
    for tr, va in folds:
        assert set(g[tr]).isdisjoint(set(g[va]))
        seen.extend(g[va].tolist())
    assert len(seen) == train.n_samples


def test_kfold_rejects_too_many_splits():
    train, _ = load_split()
    with pytest.raises(ValueError):
        group_kfold_indices(train.meta, n_splits=99)
