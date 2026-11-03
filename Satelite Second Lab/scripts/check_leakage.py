# -*- coding: utf-8 -*-
"""Проверка отсутствия утечек: пересечение экспериментов и состав признаков.

Запуск:  .venv\\Scripts\\python.exe scripts\\check_leakage.py
Код возврата 1 — утечка обнаружена.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.loader_adapt import load_split, meta_columns  # noqa: E402
from src.split_adapt import assert_disjoint  # noqa: E402


def check_step5(test) -> bool:
    """Датасет обучения с частой сеткой окон: та же проверка утечки.

    Смысл: уменьшение шага окон не должно привести к тому, что обучающее окно
    совпадёт с тестовым или что тестовая сетка окон пересоберётся.
    """
    if not config.STEP5_PATH.exists():
        print("[--] датасета с частой сеткой окон нет: проверка пропущена")
        return True
    step5 = pd.read_parquet(config.STEP5_PATH)
    ok = True

    overlap = sorted(set(step5["meta__experiment"]) & set(test.experiments))
    if overlap:
        ok = False
        print(f"[УТЕЧКА] в датасете с шагом {config.STRIDE_SMALL} тестовые эксперименты: {overlap}")
    else:
        print(f"[OK] датасет с шагом {config.STRIDE_SMALL}: эксперименты теста не входят")

    files = set(step5["meta__source_file"])
    test_files = set(test.meta["meta__source_file"])
    if files & test_files:
        ok = False
        print(f"[УТЕЧКА] общие файлы с тестом: {sorted(files & test_files)}")

    pairs = set(zip(step5["meta__source_file"], step5["meta__start_index"]))
    test_pairs = set(zip(test.meta["meta__source_file"], test.meta["meta__start_index"]))
    if pairs & test_pairs:
        ok = False
        print(f"[УТЕЧКА] совпадающих окон с тестом: {len(pairs & test_pairs)}")
    else:
        print(f"[OK] датасет с шагом {config.STRIDE_SMALL}: окон {len(step5)}, "
              f"совпадений с тестом нет")

    if not (test.meta["meta__start_index"] % config.STRIDE == 0).all():
        ok = False
        print("[УТЕЧКА] тестовая сетка окон изменилась: шаг не равен эталонному")

    return ok


def main() -> int:
    train, test = load_split()
    ok = True

    overlap = sorted(set(train.experiments) & set(test.experiments))
    if overlap:
        ok = False
        print(f"[УТЕЧКА] эксперименты в обеих выборках: {overlap}")
    else:
        print(f"[OK] эксперименты не пересекаются: "
              f"обучение {len(train.experiments)}, тест {len(test.experiments)}")

    try:
        assert_disjoint(train.meta, test.meta)
    except ValueError as exc:  # страховка на случай изменения проверки выше
        ok = False
        print(f"[УТЕЧКА] {exc}")

    bad = [f for f in train.features if f.startswith(config.META_PREFIX)]
    if bad:
        ok = False
        print(f"[УТЕЧКА] служебные колонки среди признаков: {bad[:5]}")
    else:
        print(f"[OK] признаков {len(train.features)}, служебных среди них нет")

    extra = [c for c in meta_columns(list(train.meta.columns)) if c == config.TARGET]
    if extra:
        ok = False
        print(f"[УТЕЧКА] цель попала в служебные колонки: {extra}")

    if not check_step5(test):
        ok = False

    print("[OK] утечек не найдено" if ok else "[ИТОГ] обнаружены проблемы")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
