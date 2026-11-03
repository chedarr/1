"""Признаки и цели (§ Лаб. №1).

Из очищенной таблицы discharge-циклов строится единая таблица признаков
для всех 34 батарей:
  - база: измерения цикла (параметр-статистики напряжений, токов, температур,
    ёмкость, длительности), capacity_ratio = capacity / cap0, discharge_num;
  - лаги (внутри батареи): voltage_mean_lag1/lag2, capacity_lag1/lag2/lag3;
  - производные: voltage_diff, capacity_diff, capacity_slope, discharge_rate;
  - цели «следующего цикла»: voltage_mean_next, capacity_next (последний цикл
    батареи — NaN, отбрасывается dropna в датасетах задач B/C).

capacity_ratio использует cap0 = median(валидных ёмкостей [3:8]) — прошлое/текущее
значение батареи, без информации из будущего. RUL_cycles НЕ строится здесь — см. rul.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CLEAN_CYCLES_PATH,
    FEATURES_PATH,
    LOW_CAP_THRESHOLD,
    N_INIT,
    SKIP_FORMATION,
)


def compute_cap0(capacity: pd.Series) -> float:
    """cap0 = median(valid[SKIP_FORMATION : SKIP_FORMATION + N_INIT]),
    valid = capacity >= LOW_CAP_THRESHOLD."""
    valid = capacity[capacity >= LOW_CAP_THRESHOLD]
    win = valid.iloc[SKIP_FORMATION: SKIP_FORMATION + N_INIT]
    return float(win.median()) if len(win) > 0 else float("nan")


def add_cap0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cap0"] = out.groupby("battery_id", sort=False)["capacity"].transform(compute_cap0)
    return out


def build_feature_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Вся таблица признаков (34 батареи): база + лаги + производные + цели."""
    if df is None:
        df = pd.read_parquet(CLEAN_CYCLES_PATH)
    out = add_cap0(df)
    g = out.groupby("battery_id", sort=False)

    out["capacity_ratio"] = out["capacity"] / out["cap0"]
    out["discharge_num"] = out["cycle_index"]
    out["discharge_rate"] = out["capacity"] / out["time_total"].where(out["time_total"] > 0)

    # длина серии батареи (для нормированной цели RUL_frac; НЕ признак)
    out["total_cycles"] = g["discharge_num"].transform("max")

    # лаги (внутри батареи)
    out["voltage_mean_lag1"] = g["voltage_mean"].shift(1)
    out["voltage_mean_lag2"] = g["voltage_mean"].shift(2)
    out["capacity_lag1"] = g["capacity"].shift(1)
    out["capacity_lag2"] = g["capacity"].shift(2)
    out["capacity_lag3"] = g["capacity"].shift(3)

    # производные
    out["voltage_diff"] = out["voltage_mean"] - out["voltage_mean_lag1"]
    out["capacity_diff"] = out["capacity"] - out["capacity_lag1"]
    out["capacity_slope"] = (out["capacity"] - out["capacity_lag3"]) / 3.0

    # цели «следующего цикла»
    out["voltage_mean_next"] = g["voltage_mean"].shift(-1)
    out["capacity_next"] = g["capacity"].shift(-1)

    return out


def save_feature_table(df: pd.DataFrame | None = None, out_path=None) -> pd.DataFrame:
    out = build_feature_table(df)
    (out_path or FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path or FEATURES_PATH, index=False)
    return out


def make_voltage_dataset(feat: pd.DataFrame, feature_cols, target_col) -> pd.DataFrame:
    """Датасет задачи B: все 34 батареи, dropna по признакам и цели."""
    cols = [c for c in ["battery_id", "cycle_index", "date"] if c in feat.columns]
    df = feat[cols + list(feature_cols) + [target_col]].dropna()
    return df.reset_index(drop=True)


def make_capacity_dataset(feat: pd.DataFrame, feature_cols, target_col) -> pd.DataFrame:
    """Датасет задачи C: все 34 батареи, dropna по признакам и цели."""
    cols = [c for c in ["battery_id", "cycle_index", "date"] if c in feat.columns]
    df = feat[cols + list(feature_cols) + [target_col]].dropna()
    return df.reset_index(drop=True)


if __name__ == "__main__":
    feat = save_feature_table()
    print("Строк:", len(feat), "Батарей:", feat["battery_id"].nunique())
    print("Цели:", feat["voltage_mean_next"].notna().sum(),
          feat["capacity_next"].notna().sum())