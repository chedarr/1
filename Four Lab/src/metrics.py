# -*- coding: utf-8 -*-
"""Метрики регрессии: MAE, RMSE, R², skill score, MAE по шагам горизонта."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, baseline_mae: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    by_step = np.abs(y_true - y_pred).mean(axis=0)
    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_by_step": [float(v) for v in by_step],
        "bias": float((y_pred - y_true).mean()),
        "n_samples": int(y_true.shape[0]),
        "horizon": int(y_true.shape[1]),
    }
    if baseline_mae is not None:
        result["skill_score"] = float(1.0 - mae / baseline_mae) if baseline_mae else None
        result["baseline_mae"] = float(baseline_mae)
    return result
