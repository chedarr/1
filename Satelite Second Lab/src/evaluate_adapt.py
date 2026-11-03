# -*- coding: utf-8 -*-
"""Обучение, кросс-валидация по экспериментам и baseline-модели.

Масштабирование признаков выполняется ВНУТРИ каждого разбиения:
среднее и дисперсия считаются только по обучающей части, поэтому
оценка не завышается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import config
from src.metrics_adapt import compute_metrics
from src.model_adapt import build_mlp, callbacks
from src.split_adapt import group_kfold_indices

EPOCHS = 150
BATCH_SIZE = 32


def fit_mlp(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = config.SEED,
    verbose: int = 0,
):
    """Обучает сеть с ранней остановкой по валидационной части."""
    model = build_mlp(n_features=X_fit.shape[1], seed=seed)
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks(),
        verbose=verbose,
    )
    return model, history


def cross_validate_mlp(
    X_raw: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int = 4,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
) -> list[dict]:
    """Кросс-валидация по экспериментам; масштабирование — внутри фолда."""
    folds: list[dict] = []
    for i, (tr, va) in enumerate(group_kfold_indices(meta, n_splits), 1):
        scaler = StandardScaler().fit(X_raw[tr])
        model, _ = fit_mlp(
            scaler.transform(X_raw[tr]).astype(np.float32),
            y[tr],
            scaler.transform(X_raw[va]).astype(np.float32),
            y[va],
            epochs=epochs,
            batch_size=batch_size,
            seed=config.SEED + i,
            verbose=verbose,
        )
        prob = model.predict(scaler.transform(X_raw[va]).astype(np.float32), verbose=0)
        m = compute_metrics(y[va], prob)
        m["фолд"] = i
        m["эксперименты"] = sorted(
            meta.iloc[va][config.META_PREFIX + "experiment"].unique()
        )
        folds.append(m)
    return folds


def baseline_majority(y_eval: np.ndarray) -> dict:
    """Постоянный ответ «норма»: вероятности отказа равны нулю."""
    return compute_metrics(y_eval, np.zeros(len(y_eval), dtype=float))


def baseline_logreg(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int = config.SEED,
) -> dict:
    """Логистическая регрессия со стандартизацией по обучающей части."""
    scaler = StandardScaler().fit(X_fit)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(X_fit), y_fit)
    prob = clf.predict_proba(scaler.transform(X_eval))[:, 1]
    m = compute_metrics(y_eval, prob)
    m["число_признаков"] = int(X_fit.shape[1])
    return m


def aggregate_folds(folds: list[dict]) -> dict:
    """Среднее и стандартное отклонение метрик по фолдам."""
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    out: dict = {"фолдов": len(folds)}
    for k in keys:
        vals = np.array([f[k] for f in folds if k in f], dtype=float)
        if vals.size:
            out[f"{k}_среднее"] = float(vals.mean())
            out[f"{k}_отклонение"] = float(vals.std(ddof=0))
    return out
