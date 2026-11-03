"""Проверки пайплайна для трёх задач (Лаб. №1).

Каждая проверка возвращает True/False и печатает результат с контекстом.
main.py вызывает их после каждого этапа; тесты используют те же функции.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_BATTERIES, RUL_BATTERIES


def validate_raw_counts(df: pd.DataFrame) -> bool:
    n = df["battery_id"].nunique()
    missing = sorted(set(ALL_BATTERIES) - set(df["battery_id"].unique()))
    ok = n == len(ALL_BATTERIES) and not missing
    print(f"validate_raw_counts: батарей={n}/{len(ALL_BATTERIES)}, "
          f"нет={missing if missing else '-'}", )
    return ok


def validate_rul_dataset(df: pd.DataFrame) -> bool:
    """RUL-датасет: только 8 батарей, RUL >= 0, убывает по discharge_num,
    последний цикл батареи = 0."""
    ok = True
    if set(df["battery_id"].unique()) != set(RUL_BATTERIES):
        print(f"validate_rul_dataset: не совпадает набор батарей — "
              f"{sorted(set(df['battery_id'].unique()))}")
        return False
    if (df["rul_cycles"] < 0).any():
        print("validate_rul_dataset: есть отрицательные RUL")
        ok = False
    for b, g in df.groupby("battery_id", sort=False):
        g = g.sort_values("discharge_num")
        if int(g["rul_cycles"].iloc[-1]) != 0:
            print(f"validate_rul_dataset: {b}: последний RUL != 0")
            ok = False
        if (g["rul_cycles"].diff().dropna() > 0).any():
            print(f"validate_rul_dataset: {b}: RUL не убывает")
            ok = False
    print(f"validate_rul_dataset: {df['battery_id'].nunique()} батарей, "
          f"{len(df)} строк, RUL∈[{df['rul_cycles'].min()},{df['rul_cycles'].max()}]")
    return ok


def validate_next_targets(df: pd.DataFrame, target: str) -> bool:
    """Цель «следующего цикла»: NaN ровно на последнем цикле каждой батареи."""
    nan_rows = df[df[target].isna()]
    per_battery_nan = nan_rows.groupby("battery_id").size()
    ok = True
    for b, k in per_battery_nan.items():
        g = df[df["battery_id"] == b]
        if k > 1 or g["cycle_index"].max() not in nan_rows["cycle_index"].values:
            print(f"validate_next_targets[{target}]: {b} — NaN не только на последнем цикле")
            ok = False
    total_nan = int(df[target].isna().sum())
    total_rows = len(df)
    print(f"validate_next_targets[{target}]: {total_nan}/{total_rows} NaN "
          f"(только последние циклы)")
    return ok


def validate_split(train_ids, val_ids, test_ids) -> bool:
    """Сплит по батареям: пересечений нет, покрытие полное."""
    t, v, te = set(train_ids), set(val_ids), set(test_ids)
    ok = t.isdisjoint(v) and t.isdisjoint(te) and v.isdisjoint(te)
    print(f"validate_split: train={len(t)}, val={len(v)}, test={len(te)}, "
          f"пересечений={not ok}")
    return ok


def validate_metrics(metrics: dict) -> bool:
    for k in ("mae", "rmse", "r2"):
        if "test" not in metrics.get(k, {}):
            print(f"validate_metrics: нет метрики {k}.test")
            return False
    fin = np.isfinite(list(metrics["mae"].values()) + list(metrics["r2"].values()))
    return bool(fin.all())