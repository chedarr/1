# -*- coding: utf-8 -*-
"""Тесты метрик: accuracy, macro-F1, per-class, confusion matrix."""
from __future__ import annotations

import numpy as np

from src.metrics import evaluate

CLASSES = ["a", "b", "c"]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    result = evaluate(y, y, CLASSES)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["n_samples"] == 6
    assert np.trace(np.asarray(result["confusion_matrix"])) == 6
    for name in CLASSES:
        assert result["per_class"][name]["f1"] == 1.0


def test_known_mistakes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    result = evaluate(y_true, y_pred, CLASSES)
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == 0.5
    assert result["per_class"]["a"]["support"] == 2
    assert result["per_class"]["b"]["support"] == 2
    assert result["per_class"]["c"]["support"] == 0
    assert result["per_class"]["c"]["f1"] == 0.0
    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1 and matrix[1, 0] == 1


def test_confusion_matrix_sums_to_n():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=50)
    y_pred = rng.integers(0, 3, size=50)
    result = evaluate(y_true, y_pred, CLASSES)
    matrix = np.asarray(result["confusion_matrix"])
    assert matrix.sum() == 50
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
