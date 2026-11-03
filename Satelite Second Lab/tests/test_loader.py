# -*- coding: utf-8 -*-
"""Загрузка данных: состав выборок и отбор признаков."""
from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.loader_adapt import feature_columns, load_dataset, load_split


def test_shapes():
    train, test = load_split()
    assert train.n_samples == 204
    assert test.n_samples == 62
    assert train.n_features == config.N_FEATURES == 1620
    assert test.n_features == 1620


def test_feature_columns_excludes_target_and_meta():
    cols = [config.META_PREFIX + "experiment", "00_E135", config.TARGET]
    assert feature_columns(cols) == ["00_E135"]


def test_experiment_counts():
    train, test = load_split()
    assert len(train.experiments) == config.N_TRAIN_EXPERIMENTS
    assert len(test.experiments) == config.N_TEST_EXPERIMENTS


def test_labels_are_binary():
    train, test = load_split()
    assert set(np.unique(train.y).tolist()) == {0, 1}
    assert set(np.unique(test.y).tolist()) == {0, 1}


def test_no_nan():
    train, test = load_split()
    assert not np.isnan(train.X).any()
    assert not np.isnan(test.X).any()


def test_feature_order_matches():
    train, test = load_split()
    assert train.features == test.features


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "нет_такого.parquet")


def test_summary_counts_match_labels():
    train, _ = load_split()
    summary = train.summary()
    assert summary["норма"] == int((train.y == 0).sum())
    assert summary["отказ"] == int((train.y == 1).sum())
