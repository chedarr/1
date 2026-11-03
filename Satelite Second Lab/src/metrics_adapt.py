# -*- coding: utf-8 -*-
"""Метрики бинарной классификации: accuracy, precision, recall, F1."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Считает метрики по вероятностям и порогу решения.

    Метрики соответствуют состояниям: 1 — отказ (положительный класс),
    0 — норма. `precision` и `recall` считаются именно по отказу.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "порог": float(threshold),
        "объектов": int(y_true.size),
        "верно_норма": int(tn),
        "ложная_тревога": int(fp),
        "пропуск_отказа": int(fn),
        "верно_отказ": int(tp),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def metrics_table(metrics: dict) -> str:
    """Текстовое представление метрик для консоли и отчёта."""
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    lines = [f"{k:<10} {metrics[k]:.4f}" for k in keys if k in metrics]
    lines.append(
        f"матрица ошибок: верно норма {metrics['верно_норма']}, "
        f"ложная тревога {metrics['ложная_тревога']}, "
        f"пропуск отказа {metrics['пропуск_отказа']}, "
        f"верно отказ {metrics['верно_отказ']}"
    )
    return "\n".join(lines)
