# -*- coding: utf-8 -*-
"""Проверки данных и отсутствия утечек перед обучением и после него."""
from __future__ import annotations

import numpy as np

from src import config
from src.loader_adapt import Dataset
from src.split_adapt import assert_disjoint, groups


def _check(
    result: list, name: str, ok: bool, detail: str = "", level: str = "критично"
) -> None:
    """Добавляет пункт проверки.

    level='критично' — при невыполнении обучение останавливается;
    level='информация' — невыполнение лишь отмечается в отчёте.
    """
    if ok:
        mark = "OK"
    else:
        mark = "ПРОБЛЕМА" if level == "критично" else "ВНИМАНИЕ"
    result.append(
        {
            "проверка": name,
            "результат": mark,
            "детали": detail,
            "уровень": level,
        }
    )


def full_report(train: Dataset, test: Dataset) -> list[dict]:
    """Набор проверок корректности и отсутствия утечек."""
    res: list[dict] = []

    _check(res, "Число признаков", train.n_features == config.N_FEATURES
           and test.n_features == config.N_FEATURES,
           f"train {train.n_features}, test {test.n_features}, ожидалось {config.N_FEATURES}")

    meta_in_features = [f for f in train.features if f.startswith(config.META_PREFIX)]
    _check(res, "Служебные колонки не попали в признаки", not meta_in_features,
           f"найдено: {meta_in_features}" if meta_in_features else "нет")

    _check(res, "Порядок признаков одинаков", train.features == test.features,
           "совпадает" if train.features == test.features else "различается")

    nan_tr = int(np.isnan(train.X).sum())
    nan_te = int(np.isnan(test.X).sum())
    _check(res, "Пропуски в значениях сенсоров", nan_tr == 0 and nan_te == 0,
           f"train {nan_tr}, test {nan_te}")

    labels_tr = set(np.unique(train.y).tolist())
    labels_te = set(np.unique(test.y).tolist())
    _check(res, "Метка принимает только 0 и 1",
           labels_tr <= {0, 1} and labels_te <= {0, 1},
           f"train {sorted(labels_tr)}, test {sorted(labels_te)}")

    try:
        assert_disjoint(train.meta, test.meta)
        _check(res, "Эксперименты Train и Test не пересекаются", True,
               f"train {len(train.experiments)}, test {len(test.experiments)}")
    except ValueError as exc:
        _check(res, "Эксперименты Train и Test не пересекаются", False, str(exc))

    same_exp = set(np.unique(groups(train.meta)).tolist()) <= set(train.experiments)
    _check(res, "Группировка по экспериментам читается", same_exp,
           f"{len(train.experiments)} групп в Train")

    const_train = int((train.X.std(axis=0) == 0).sum())
    const_test = int((test.X.std(axis=0) == 0).sum())
    _check(
        res,
        "Постоянные признаки",
        const_train == 0,
        f"не меняются на всей выборке: в обучении {const_train} из {train.n_features} "
        f"колонок, в тесте {const_test}; такие колонки не несут информации "
        f"и не мешают обучению",
        level="информация",
    )

    n_tr, n_te = train.n_samples, test.n_samples
    share = float(train.y.mean())
    _check(res, "Размеры выборок", n_tr > 0 and n_te > 0,
           f"train {n_tr}, test {n_te}")
    _check(res, "Баланс классов в Train", 0.3 <= share <= 0.7,
           f"доля отказа {share:.2f}")

    return res


def problems(report: list[dict]) -> list[dict]:
    """Только критичные невыполненные проверки (останавливают обучение)."""
    return [r for r in report if r["результат"] == "ПРОБЛЕМА"]


def notes(report: list[dict]) -> list[dict]:
    """Информационные замечания, не мешающие обучению."""
    return [r for r in report if r["результат"] == "ВНИМАНИЕ"]
