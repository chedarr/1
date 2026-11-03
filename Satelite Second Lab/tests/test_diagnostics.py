# -*- coding: utf-8 -*-
"""Дешёвые проверки диагностики разброса LOEO (без обучения и без сети).

Проверяется только то, что уже посчитано и записано диагностическим скриптом:
наличие и состав отчёта, отсутствие NaN, а также каноническое определение
расстояния — в него входят ровно признаки, варьирующиеся на обучении, и оно
конечно и сопоставимо между записями. Плюс то, что статистики z-пространства
обучаются исключительно на обучающих окнах и что набор «обучение_12» содержит
ровно 12 наблюдений.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src import config
from scripts.diagnose_loeo import (
    CHARS,
    Z_CLIP,
    build_characteristics,
    build_correlations,
    combined_frame,
    feature_columns,
    informative_mask,
    load_frames,
    load_loeo,
    rms_distance,
    train_stats,
    unbounded_distance,
)

REPORT_PATH = config.REPORTS_DIR / "loeo_diagnostics.json"
EXP = config.META_PREFIX + "experiment"


def _report() -> dict:
    if not REPORT_PATH.exists():
        pytest.skip("отчёт диагностики ещё не посчитан")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _has_nan(value) -> bool:
    """Есть ли где-нибудь NaN или бесконечность (рекурсивно по структуре)."""
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nan(v) for v in value)
    return False


def test_report_has_16_experiments_and_matching_windows():
    """Отчёт есть, в нём 16 характеристик, а объёмы совпадают с отчётом LOEO."""
    report = _report()
    rows = report["характеристики"]
    assert len(rows) == config.N_EXPERIMENTS == 16

    folds = {str(f["эксперимент"]): f for f in load_loeo()["фолды"]}
    assert {r["эксперимент"] for r in rows} == set(folds)
    for row in rows:
        assert row["окон"] == int(folds[row["эксперимент"]]["объектов"])

    assert len(report["корреляции"]["все_16"]["f1"]) == len(CHARS)
    assert len(report["корреляции"]["обучение_12"]["f1"]) == len(CHARS)
    assert not _has_nan(report), "в отчёте есть NaN или бесконечность"

    health = report["здоровье_данных"]
    assert health["варьирующихся_на_обучении"] + health["постоянных_на_обучении"] \
        == health["всего_признаков"] == config.N_FEATURES


def test_distance_uses_only_varying_features_and_is_bounded():
    """В расстояние входят только варьирующиеся признаки, и оно ограничено сверху."""
    report = _report()
    rows = report["характеристики"]
    train, test, features = load_frames()
    frame = combined_frame(train, test)

    _, sigma = train_stats(frame, features)
    mask = informative_mask(sigma)
    n_informative = int(mask.sum())
    assert 0 < n_informative < len(features)  # часть признаков на обучении постоянна

    # число признаков в расстоянии одинаково для всех записей и равно числу
    # варьирующихся на обучении
    for row in rows:
        assert row["признаков_в_расстоянии"] == n_informative

    # ограничение z числом 10 задаёт верхнюю границу самого расстояния
    for row in rows:
        assert row["dist_to_train"] is not None
        assert math.isfinite(row["dist_to_train"])
        assert 0.0 <= row["dist_to_train"] <= Z_CLIP
        assert row["dist_to_train_looe"] <= Z_CLIP

    # порча тестовых окон в признаке, постоянном на обучении, НЕ меняет расстояние
    const_idx = int(np.flatnonzero(~mask)[0])
    mu, sigma = train_stats(frame, features)
    sub_test = frame[frame[EXP] == config.TEST_EXPERIMENTS[0]]
    before = rms_distance(sub_test, features, mu, sigma, mask)

    spoiled = frame.copy()
    held = spoiled["источник"] == "тест"
    spoiled.loc[held, features[const_idx]] += 1000.0
    mu2, sigma2 = train_stats(spoiled, features)
    mask2 = informative_mask(sigma2)
    after = rms_distance(spoiled[spoiled[EXP] == config.TEST_EXPERIMENTS[0]],
                         features, mu2, sigma2, mask2)
    assert mask2.sum() == n_informative
    assert after == before

    # для контраста: тот же сдвиг в признаке, варьирующемся на обучении, меняет
    # расстояние, то есть информативные признаки действительно учитываются
    var_idx = int(np.flatnonzero(mask)[0])
    spoiled_var = frame.copy()
    spoiled_var.loc[held, features[var_idx]] += 1000.0
    changed = rms_distance(spoiled_var[spoiled_var[EXP] == config.TEST_EXPERIMENTS[0]],
                           features, mu, sigma, mask)
    assert changed != before

    # ограничение работает: вариант без ограничения на том же входе даёт не меньше,
    # а на патологической записи (почти нулевой разброс) — намного больше
    unbounded = unbounded_distance(
        spoiled_var[spoiled_var[EXP] == config.TEST_EXPERIMENTS[0]],
        features, mu, sigma, mask)
    assert unbounded >= changed


def test_train_stats_ignore_test_windows_and_are_idempotent():
    """Среднее и разброс z-пространства берутся только из обучающих окон."""
    train, test, features = load_frames()
    frame = combined_frame(train, test)

    mu_first, sigma_first = train_stats(frame, features)
    mu_again, sigma_again = train_stats(frame, features)
    # повторный вызов на том же входе даёт те же числа (идемпотентность)
    assert np.array_equal(mu_first, mu_again)
    assert np.array_equal(sigma_first, sigma_again)

    # эталон: посчитано напрямую по обучающей части, без замены нулей
    mu_ref = train[features].mean(axis=0).to_numpy(dtype=float)
    sigma_ref = train[features].std(axis=0, ddof=0).to_numpy(dtype=float)
    assert np.allclose(mu_first, mu_ref)
    assert np.allclose(sigma_first, sigma_ref)

    # порча тестовых окон не меняет обучающую статистику ни на йоту
    spoiled = frame.copy()
    held = spoiled["источник"] == "тест"
    spoiled.loc[held, features] = 100.0 * spoiled.loc[held, features].to_numpy() + 7.0
    mu_spoiled, sigma_spoiled = train_stats(spoiled, features)
    assert np.array_equal(mu_first, mu_spoiled)
    assert np.array_equal(sigma_first, sigma_spoiled)

    # исключение обучающего эксперимента действительно сужает выборку
    first_experiment = sorted(set(train[EXP]))[0]
    mu_looe, _ = train_stats(frame, features, exclude=first_experiment)
    assert not np.allclose(mu_first, mu_looe)


def test_train_correlation_set_has_exactly_12_rows():
    """В набор «обучение_12» попадают ровно 12 записей, во «все_16» — 16."""
    train, test, _ = load_frames()
    report = load_loeo()
    rows, _, _ = build_characteristics(train, test, report)

    assert sum(r["источник"] == "обучение" for r in rows) == config.N_TRAIN_EXPERIMENTS == 12

    correlations, flat = build_correlations(rows)
    for target in ("f1", "f1_в_пороге_0.5"):
        for char in CHARS:
            assert correlations["обучение_12"][target][char]["n"] == 12
            assert correlations["все_16"][target][char]["n"] == 16
    assert {item["набор"] for item in flat} == {"все_16", "обучение_12"}
