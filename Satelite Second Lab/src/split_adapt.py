# -*- coding: utf-8 -*-
"""Разбиение по экспериментам и кросс-валидация без утечки.

Окна одного эксперимента сильно перекрываются, поэтому случайная
кросс-валидация внутри Train даёт завышенную оценку. Все разбиения
выполняются по группам-экспериментам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config

GROUP_COL = config.META_PREFIX + "experiment"


def groups(meta: pd.DataFrame) -> np.ndarray:
    """Идентификаторы экспериментов, соответствующие строкам выборки."""
    return meta[GROUP_COL].to_numpy()


def assert_disjoint(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> None:
    """Проверяет, что ни один эксперимент не попал в обе выборки."""
    overlap = set(groups(train_meta)) & set(groups(test_meta))
    if overlap:
        raise ValueError(f"эксперименты встречаются и в Train, и в Test: {sorted(overlap)}")


def group_kfold_indices(meta: pd.DataFrame, n_splits: int = 4):
    """K-fold по экспериментам: возвращает список пар (индексы train, индексы val)."""
    g = groups(meta)
    n_groups = len(np.unique(g))
    if n_splits > n_groups:
        raise ValueError(f"групп {n_groups}, а фолдов запрошено {n_splits}")
    splitter = GroupKFold(n_splits=n_splits)
    empty = np.zeros(len(meta), dtype=int)
    return list(splitter.split(empty, empty, groups=g))


def holdout_by_experiment(
    meta: pd.DataFrame, n_val_experiments: int = 2, seed: int = config.SEED
):
    """Выделяет часть экспериментов под валидацию (для ранней остановки).

    Группы перемешиваются фиксированным генератором, поэтому результат
    воспроизводим и не зависит от порядка строк.
    """
    g = groups(meta)
    unique = np.array(sorted(np.unique(g)))
    if n_val_experiments >= len(unique):
        raise ValueError("под валидацию запрошено слишком много экспериментов")
    rng = np.random.default_rng(seed)
    val_groups = rng.choice(unique, size=n_val_experiments, replace=False)
    val_mask = np.isin(g, val_groups)
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), sorted(val_groups.tolist())
