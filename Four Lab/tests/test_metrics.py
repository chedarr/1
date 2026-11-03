# -*- coding: utf-8 -*-
import numpy as np

from src.baselines import constant, linear, persistence
from src.metrics import evaluate


def test_metrics_perfect_prediction():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = evaluate(y, y, baseline_mae=1.0)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["skill_score"] == 1.0
    assert result["mae_by_step"] == [0.0, 0.0]


def test_metrics_skill_negative_when_worse():
    y = np.zeros((4, 2))
    pred = np.ones((4, 2))
    result = evaluate(y, pred, baseline_mae=0.5)
    assert result["skill_score"] == -1.0


def test_persistence_repeats_last_value():
    x = np.zeros((3, 5, 5), dtype=np.float32)
    x[:, -1, 2] = 7.0
    result = persistence(x, target_index=2, horizon=4, target_mean=0.0, target_std=1.0)
    assert result.shape == (3, 4)
    assert np.all(result == 7.0)


def test_persistence_inverse_scaling():
    x = np.zeros((1, 3, 5), dtype=np.float32)
    x[:, -1, 2] = 1.0
    result = persistence(x, target_index=2, horizon=2, target_mean=2.0, target_std=0.5)
    assert np.allclose(result, 2.5)


def test_constant_baseline():
    x = np.zeros((2, 4, 5), dtype=np.float32)
    result = constant(x, train_mean=3.0, horizon=6)
    assert result.shape == (2, 6)
    assert np.all(result == 3.0)


def test_linear_baseline_shape():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(60, 4, 5)).astype(np.float32)
    y_train = rng.normal(size=(60, 3)).astype(np.float32)
    x_eval = rng.normal(size=(10, 4, 5)).astype(np.float32)
    result = linear(x_train, y_train, x_eval)
    assert result.shape == (10, 3)
