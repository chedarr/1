# -*- coding: utf-8 -*-
"""Загрузка оконных данных ADAPT и отбор признаков.

Признаками считаются только колонки сенсоров; служебные колонки
(с префиксом из config.META_PREFIX) в матрицу признаков не попадают.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


@dataclass
class Dataset:
    """Оконная выборка: матрица признаков, метки и служебные колонки."""

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    features: list[str]
    source: Path

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def experiments(self) -> list[str]:
        """Список экспериментов, представленных в выборке."""
        return sorted(self.meta[config.META_PREFIX + "experiment"].unique())

    def summary(self) -> dict:
        """Краткая сводка по выборке (для отчёта и тестов)."""
        labels, counts = np.unique(self.y, return_counts=True)
        return {
            "источник": self.source.name,
            "окон": self.n_samples,
            "признаков": self.n_features,
            "экспериментов": len(self.experiments),
            "норма": int(counts[labels == 0].sum()) if (labels == 0).any() else 0,
            "отказ": int(counts[labels == 1].sum()) if (labels == 1).any() else 0,
        }


def feature_columns(columns) -> list[str]:
    """Оставляет только колонки-признаки: без цели и без служебных колонок."""
    return [
        c
        for c in columns
        if c != config.TARGET and not c.startswith(config.META_PREFIX)
    ]


def meta_columns(columns) -> list[str]:
    """Служебные колонки (не признаки)."""
    return [c for c in columns if c.startswith(config.META_PREFIX)]


def load_dataset(path: Path | str) -> Dataset:
    """Читает parquet и разделяет его на признаки, метку и служебные колонки."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла данных: {path}")
    df = pd.read_parquet(path)
    if config.TARGET not in df.columns:
        raise ValueError(f"в {path.name} нет колонки-цели {config.TARGET!r}")

    feats = feature_columns(df.columns)
    if len(feats) != config.N_FEATURES:
        raise ValueError(
            f"{path.name}: признаков {len(feats)}, ожидалось {config.N_FEATURES}"
        )
    X = df[feats].to_numpy(dtype=np.float32)
    y = df[config.TARGET].to_numpy(dtype=np.int32)
    meta = df[meta_columns(df.columns)].reset_index(drop=True)
    return Dataset(X=X, y=y, meta=meta, features=feats, source=path)


def load_split() -> tuple[Dataset, Dataset]:
    """Возвращает (train, test) в согласованном порядке признаков."""
    train = load_dataset(config.TRAIN_PATH)
    test = load_dataset(config.TEST_PATH)
    if train.features != test.features:
        raise ValueError("порядок признаков в train и test различается")
    return train, test
