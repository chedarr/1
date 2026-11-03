# -*- coding: utf-8 -*-
"""Baseline: persistence, constant, linear."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def persistence(x: np.ndarray, target_index: int, horizon: int,
                target_mean: float, target_std: float) -> np.ndarray:
    """Последнее известное значение целевой, повторённое horizon раз.

    x — нормализованные признаки, поэтому значение возвращается в исходных единицах.
    """
    last = x[:, -1, target_index] * target_std + target_mean
    return np.repeat(last[:, None], horizon, axis=1)


def constant(x: np.ndarray, train_mean: float, horizon: int) -> np.ndarray:
    """Среднее целевой по train, повторённое horizon раз."""
    return np.full((x.shape[0], horizon), train_mean, dtype=np.float32)


def linear(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
           alpha: float = 1.0) -> np.ndarray:
    """Ridge на всех лагах всех каналов; multioutput — сразу horizon шагов."""
    model = Ridge(alpha=alpha, solver="cholesky")
    model.fit(x_train.reshape(len(x_train), -1), y_train)
    return model.predict(x_eval.reshape(len(x_eval), -1)).astype(np.float32)
