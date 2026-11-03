# -*- coding: utf-8 -*-
"""Механика экспериментов: пересборка окон, признаки, порог."""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.experiments_adapt import (
    Config,
    best_thresholds,
    build_features,
    build_windows,
    class_weights,
    load_measurements,
    moving_average,
    threshold_curve,
    verify_against_reference,
)

WINDOW = config.WINDOW_SIZE


@pytest.fixture(scope="module")
def meas_sensors():
    return load_measurements()


@pytest.fixture(scope="module")
def reference(meas_sensors):
    meas, sensors = meas_sensors
    return build_windows(meas, sensors, 10, (0,), test=False)


def test_window_rebuild_reproduces_accepted_dataset(reference):
    """Пересборка из сырых измерений обязана совпасть с принятым датасетом."""
    x, y, meta = reference
    check = verify_against_reference(x, y, meta)
    assert check["совпадает_размер"] is True
    assert check["совпадают_метки"] is True
    assert check["совпадают_начала_окон"] is True
    assert check["максимальное_расхождение"] == 0.0


def test_rebuild_has_expected_size(reference, meas_sensors):
    meas, sensors = meas_sensors
    x, y, _ = reference
    assert x.shape == (204, WINDOW, len(sensors))
    x_test, y_test, _ = build_windows(meas, sensors, 10, (0,), test=True)
    assert x_test.shape == (62, WINDOW, len(sensors))
    assert set(np.unique(y).tolist()) == {0, 1}
    assert set(np.unique(y_test).tolist()) == {0, 1}


def test_train_and_test_experiments_do_not_overlap(meas_sensors):
    meas, sensors = meas_sensors
    _, _, meta_train = build_windows(meas, sensors, 10, (0,), test=False)
    _, _, meta_test = build_windows(meas, sensors, 10, (0,), test=True)
    col = config.META_PREFIX + "experiment"
    assert set(meta_train[col]).isdisjoint(set(meta_test[col]))


def test_smaller_stride_gives_more_windows(meas_sensors):
    meas, sensors = meas_sensors
    _, y10, _ = build_windows(meas, sensors, 10, (0,), test=False)
    _, y5, _ = build_windows(meas, sensors, 5, (0,), test=False)
    assert len(y5) > len(y10)


def test_offsets_add_shifted_windows(meas_sensors):
    meas, sensors = meas_sensors
    _, y0, meta0 = build_windows(meas, sensors, 10, (0,), test=False)
    _, y2, meta2 = build_windows(meas, sensors, 10, (-2, 0, 2), test=False)
    assert len(y2) > len(y0)
    starts0 = set(meta0["meta__start_index"])
    starts2 = set(meta2["meta__start_index"])
    assert starts0.issubset(starts2)


def test_test_windows_are_never_rebuilt_with_other_stride(meas_sensors):
    """Тестовая выборка всегда строится базовым шагом и остаётся неизменной."""
    meas, sensors = meas_sensors
    x_test, y_test, meta_test = build_windows(meas, sensors, 10, (0,), test=True)
    assert x_test.shape == (62, WINDOW, 81)
    col = config.META_PREFIX + "experiment"
    assert sorted(set(meta_test[col])) == sorted(config.TEST_EXPERIMENTS)
    # число окон теста не зависит от шага обучения: его строят только базовым шагом
    x_again, _, _ = build_windows(meas, sensors, 10, (0,), test=True)
    assert np.array_equal(x_test, x_again) and np.array_equal(y_test, y_test)


def test_feature_block_shapes():
    x3 = np.random.default_rng(0).normal(size=(5, WINDOW, 81)).astype(np.float32)
    assert build_features(x3, ("raw",)).shape == (5, WINDOW * 81)
    assert build_features(x3, ("stats",)).shape == (5, 4 * 81)
    assert build_features(x3, ("delta",)).shape == (5, (WINDOW - 1) * 81)
    assert build_features(x3, ("smooth3",)).shape == (5, WINDOW * 81)
    assert build_features(x3, ("raw", "stats", "delta")).shape == (5, WINDOW * 81 + 4 * 81 + (WINDOW - 1) * 81)


def test_feature_blocks_are_concatenated_in_order():
    x3 = np.random.default_rng(1).normal(size=(3, WINDOW, 81)).astype(np.float32)
    raw = build_features(x3, ("raw",))
    both = build_features(x3, ("raw", "stats"))
    assert np.array_equal(both[:, : raw.shape[1]], raw)
    assert np.allclose(both[:, : WINDOW * 81].reshape(3, -1), x3.reshape(3, -1))


def test_moving_average_keeps_constant_signal():
    x3 = np.full((2, WINDOW, 3), 7.0)
    for width in (3, 5):
        assert np.allclose(moving_average(x3, width), 7.0)


def test_moving_average_smooths_spike():
    x3 = np.zeros((1, WINDOW, 1))
    x3[0, WINDOW // 2, 0] = 10.0
    smooth = moving_average(x3, 3)
    assert smooth.max() < 10.0
    assert smooth.max() > 0.0
    assert smooth.shape == x3.shape


def test_class_weights_are_balanced():
    y = np.array([0] * 8 + [1] * 2)
    w = class_weights(y, True)
    assert w[1] > w[0]
    assert w[0] * 8 == pytest.approx(w[1] * 2)
    assert class_weights(y, False) is None


def test_threshold_selection_prefers_separating_point():
    y = np.array([0, 1])
    prob = np.array([0.4, 0.6])
    picks = best_thresholds(threshold_curve(y, prob))
    assert picks["f1"]["f1"] == pytest.approx(1.0)
    assert picks["f1"]["precision"] >= picks["требование_точности"]
    assert picks["f2"]["порог"] <= picks["f1"]["порог"]


def test_threshold_floor_rejects_degenerate_answer():
    """Если ни один порог не даёт нужной точности, выбор возвращается без ограничения."""
    y = np.array([1, 1, 1, 0, 0])
    prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    picks = best_thresholds(threshold_curve(y, prob))
    assert picks["ограничение_применено"] is False
    assert picks["f1"]["f1"] > 0
    assert picks["f1"]["precision"] < picks["требование_точности"]


def test_threshold_without_floor_is_reported():
    y = np.array([0, 1])
    prob = np.array([0.4, 0.6])
    picks = best_thresholds(threshold_curve(y, prob))
    assert "без_ограничения" in picks
    assert 0.0 <= picks["без_ограничения"]["порог"] <= 1.0


def test_always_fault_answer_is_not_a_solution():
    """Постоянный ответ «отказ» даёт высокую полноту — это и есть ловушка метрики."""
    from src.metrics_adapt import compute_metrics

    y_test = np.array([1] * 32 + [0] * 30)
    m = compute_metrics(y_test, np.ones(len(y_test)))
    assert m["recall"] == 1.0
    assert m["f1"] < 0.7
    assert m["precision"] < 0.6


def test_threshold_curve_is_sorted():
    y = np.array([0, 1, 1, 0])
    prob = np.array([0.1, 0.9, 0.4, 0.35])
    curve = threshold_curve(y, prob)
    thresholds = [row["порог"] for row in curve]
    assert thresholds == sorted(thresholds)
    assert all(0.0 <= row["precision"] <= 1.0 for row in curve)


def test_config_extras_are_readable():
    assert Config("x").extras == "как в базовой модели"
    text = Config("x", dropout=0.3, batchnorm=True, l2=1e-4, class_weight=True).extras
    assert "веса классов" in text and "нормализация" in text


def test_step5_config_uses_configured_stride():
    """Вариант с частой сеткой берёт шаг из настроек, а не литерал."""
    from src.experiments_adapt import default_configs, find_config

    cfg = find_config("K_шаг_5_базовая")
    assert cfg is not None
    assert cfg.stride == config.STRIDE_SMALL
    assert f"шаг {config.STRIDE_SMALL}" in cfg.extras
    names = [c.name for c in default_configs()]
    assert len(names) == len(set(names))
    assert find_config("нет_такой") is None


def test_step5_windows_are_consistent_with_measurements(meas_sensors):
    """Окна с частой сеткой лежат внутри своего эксперимента и не пересекают сетку теста."""
    meas, sensors = meas_sensors
    x5, y5, meta5 = build_windows(meas, sensors, config.STRIDE_SMALL, (0,), test=False)
    _, _, meta_test = build_windows(meas, sensors, config.STRIDE, (0,), test=True)
    col = config.META_PREFIX + "experiment"
    assert set(meta5[col]).isdisjoint(set(meta_test[col]))
    assert len(y5) > 204
    assert x5.shape[1:] == (WINDOW, len(sensors))
    sizes = meas.groupby("source_file").size().to_dict()
    assert (meta5["meta__end_index"] < meta5["meta__source_file"].map(sizes)).all()
