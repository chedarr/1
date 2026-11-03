# -*- coding: utf-8 -*-
"""Метрики классификации: accuracy, macro-F1, per-class, confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Считает метрики по предсказаниям (int-метки)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(class_names))
        ).tolist(),
        "n_samples": int(y_true.size),
    }
