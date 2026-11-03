# -*- coding: utf-8 -*-
"""Честная оценка по одному эксперименту (без обучения) и датасет с шагом 5.

Обучение здесь не запускается: проверяются разбиение на фолды, подсчёт сводки,
воспроизводимость зерна и отсутствие утечек между обучением и тестом.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.loeo_adapt import fold_seed, loeo_splits, summarize

EXP_COL = config.META_PREFIX + "experiment"


def _meta(experiments: dict[str, int]) -> pd.DataFrame:
    """Служебная таблица окон: эксперимент -> число окон."""
    rows = []
    for exp, count in experiments.items():
        rows.extend({"meta__experiment": exp, "meta__start_index": i} for i in range(count))
    return pd.DataFrame(rows)


def test_loeo_gives_one_fold_per_experiment():
    meta = _meta({"a": 3, "b": 2, "c": 4})
    splits = loeo_splits(meta)
    assert len(splits) == 3
    assert [s["эксперимент"] for s in splits] == ["a", "b", "c"]
    for item in splits:
        held = set(meta.iloc[item["индексы_контроля"]][EXP_COL])
        assert held == {item["эксперимент"]}
        assert item["эксперимент"] not in set(meta.iloc[item["индексы_обучения"]][EXP_COL])


def test_loeo_covers_every_window_exactly_once():
    meta = _meta({"a": 3, "b": 2, "c": 4})
    splits = loeo_splits(meta)
    held = np.concatenate([s["индексы_контроля"] for s in splits])
    assert sorted(held.tolist()) == list(range(len(meta)))
    for item in splits:
        assert len(item["индексы_обучения"]) == len(meta) - len(item["индексы_контроля"])
        assert not set(item["индексы_обучения"]) & set(item["индексы_контроля"])


def test_loeo_seed_is_fixed_per_fold():
    meta = _meta({"a": 1, "b": 1, "c": 1})
    splits = loeo_splits(meta)
    seeds = [s["зерно"] for s in splits]
    assert seeds == [fold_seed(1), fold_seed(2), fold_seed(3)]
    assert seeds == [config.SEED + 1, config.SEED + 2, config.SEED + 3]
    assert len(set(seeds)) == len(seeds)  # фолды не делят зерно


def test_loeo_summary_reports_spread():
    folds = [
        {"f1": 1.0, "f1_0.5": 0.9, "precision": 1.0, "recall": 1.0, "accuracy": 1.0,
         "roc_auc": 1.0, "верно_отказ": 8, "пропуск_отказа": 0},
        {"f1": 0.4, "f1_0.5": 0.5, "precision": 0.5, "recall": 0.33, "accuracy": 0.6,
         "roc_auc": 0.7, "верно_отказ": 3, "пропуск_отказа": 6},
        {"f1": 0.7, "f1_0.5": 0.7, "precision": 0.8, "recall": 0.6, "accuracy": 0.8,
         "roc_auc": 0.9, "верно_отказ": 5, "пропуск_отказа": 3},
    ]
    summary = summarize(folds)
    assert summary["фолдов"] == 3
    assert summary["f1_среднее"] == pytest.approx(0.7, abs=1e-9)
    assert summary["f1_минимум"] == pytest.approx(0.4)
    assert summary["f1_максимум"] == pytest.approx(1.0)
    assert summary["f1_отклонение"] > 0
    assert summary["провалов_ниже_0.5"] == 1
    assert summary["полных_попаданий"] == 1
    assert summary["отказов_всего"] == 25


def test_loeo_summary_with_suffix_uses_fixed_threshold():
    folds = [
        {"f1": 1.0, "f1_0.5": 0.2, "верно_отказ": 1, "пропуск_отказа": 0},
        {"f1": 0.8, "f1_0.5": 0.6, "верно_отказ": 1, "пропуск_отказа": 0},
    ]
    summary = summarize(folds, "_0.5")
    assert summary["f1_0.5_среднее"] == pytest.approx(0.4)
    assert summary["f1_0.5_минимум"] == pytest.approx(0.2)


# ------------------------------------------------------------- датасет с шагом 5
def test_step5_dataset_has_no_test_experiments():
    if not config.STEP5_PATH.exists():
        pytest.skip("датасет с частой сеткой ещё не собран")
    frame = pd.read_parquet(config.STEP5_PATH)
    experiments = set(frame["meta__experiment"])
    assert not experiments & set(config.TEST_EXPERIMENTS)
    assert len(experiments) == config.N_TRAIN_EXPERIMENTS


def test_step5_dataset_does_not_share_windows_with_test():
    """Утечка окон: ни одно окно обучения не совпадает с тестовым."""
    if not config.STEP5_PATH.exists() or not config.TEST_PATH.exists():
        pytest.skip("датасеты ещё не собраны")
    step5 = pd.read_parquet(config.STEP5_PATH)
    test = pd.read_parquet(config.TEST_PATH)
    pairs5 = set(zip(step5["meta__source_file"], step5["meta__start_index"]))
    pairs_test = set(zip(test["meta__source_file"], test["meta__start_index"]))
    assert not pairs5 & pairs_test
    assert not set(step5["meta__source_file"]) & set(test["meta__source_file"])


def test_step5_dataset_has_more_windows_and_same_shape():
    if not config.STEP5_PATH.exists():
        pytest.skip("датасет с частой сеткой ещё не собран")
    frame = pd.read_parquet(config.STEP5_PATH)
    feats = [c for c in frame.columns
             if not c.startswith(config.META_PREFIX) and c != config.TARGET]
    assert len(feats) == config.N_FEATURES
    assert (frame["meta__start_index"] % config.STRIDE_SMALL == 0).all()
    assert frame["meta__end_index"].equals(
        frame["meta__start_index"] + config.WINDOW_SIZE - 1
    )
    reference = pd.read_parquet(config.TRAIN_PATH)
    assert len(frame) > len(reference)


def test_step5_dataset_is_not_read_as_reference():
    """Основной датасет обучения остался на эталонном шаге."""
    reference = pd.read_parquet(config.TRAIN_PATH)
    assert (reference["meta__start_index"] % config.STRIDE == 0).all()
    assert len(reference) == 204
