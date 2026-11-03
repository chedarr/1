# -*- coding: utf-8 -*-
"""Метрики: значения на эталонных примерах."""
from __future__ import annotations

import numpy as np

from src.metrics_adapt import compute_metrics, metrics_table


def test_perfect_prediction():
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y, prob)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert m[key] == 1.0
    assert m["верно_норма"] == 2 and m["верно_отказ"] == 2
    assert m["ложная_тревога"] == 0 and m["пропуск_отказа"] == 0


def test_all_norm_prediction_has_zero_f1():
    y = np.array([0, 1, 1, 0])
    m = compute_metrics(y, np.zeros(4))
    assert m["f1"] == 0.0
    assert m["recall"] == 0.0
    assert m["accuracy"] == 0.5


def test_confusion_counts():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.9, 0.4, 0.6])
    m = compute_metrics(y, prob)
    assert m["верно_норма"] == 1
    assert m["ложная_тревога"] == 1
    assert m["пропуск_отказа"] == 1
    assert m["верно_отказ"] == 1


def test_threshold_changes_result():
    y = np.array([0, 1])
    prob = np.array([0.4, 0.6])
    assert compute_metrics(y, prob, threshold=0.5)["accuracy"] == 1.0
    assert compute_metrics(y, prob, threshold=0.7)["accuracy"] == 0.5


def test_metrics_table_contains_all_lines():
    text = metrics_table(compute_metrics(np.array([0, 1]), np.array([0.2, 0.8])))
    assert "accuracy" in text and "f1" in text
    assert "матрица ошибок" in text
