"""RUL — задача A (Лаб. №1).

RUL только для 8 батарей с полной кривой деградации
(B0005, B0006, B0018, B0045, B0046, B0047, B0048, B0053).

    RUL_cycles = max(discharge_num) - discharge_num  (внутри батареи)

Для остальных 26 батарей (censored / аномальные / короткие) RUL не определяется
и они не используются в задаче A — для них работают задачи B и C (все 34).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURES_PATH,
    RUL_BATTERIES,
    RUL_DATASET_PATH,
    RUL_FEATURES,
    RUL_TARGET,
    RUL_TARGET_CYCLES,
)


def compute_rul(feat: pd.DataFrame, batteries: list[str] | None = None) -> pd.DataFrame:
    """Добавить колонки RUL_cycles = max(discharge_num) - discharge_num
    и RUL_frac = RUL_cycles / total_cycles (нормированная цель обучения)
    для заданных батарей (по умолчанию — все, входящие в таблицу)."""
    batteries = batteries or RUL_BATTERIES
    out = feat.copy()
    out[RUL_TARGET_CYCLES] = np.nan
    out[RUL_TARGET] = np.nan
    for b in batteries:
        m = out["battery_id"] == b
        s = out.loc[m, "discharge_num"]
        if len(s) == 0:
            continue
        cyl = (s.max() - s).values
        if "total_cycles" in out.columns:
            tot = out.loc[m, "total_cycles"].iloc[0]
        else:
            tot = float(s.max())
        out.loc[m, RUL_TARGET_CYCLES] = cyl
        out.loc[m, RUL_TARGET] = np.where(tot > 0, cyl / tot, np.nan)
    return out


def make_rul_dataset(feat: pd.DataFrame,
                     feature_cols=None, target_col=RUL_TARGET) -> pd.DataFrame:
    """Датасет задачи A: ровно 8 батарей, их признаки + RUL_frac (цель)
    + RUL_cycles и total_cycles (для интерпретации)."""
    feature_cols = feature_cols or RUL_FEATURES
    cols = [c for c in ["battery_id", "cycle_index", "date"] if c in feat.columns]
    keep = cols + list(feature_cols) + [RUL_TARGET_CYCLES, "total_cycles", target_col]
    df = feat[feat["battery_id"].isin(RUL_BATTERIES)].copy()
    df = compute_rul(df, batteries=RUL_BATTERIES)
    df = df[[c for c in keep if c in df.columns]].dropna()
    return df.reset_index(drop=True)


def save_rul_dataset(feat: pd.DataFrame | None = None, out_path=None) -> pd.DataFrame:
    if feat is None:
        feat = pd.read_parquet(FEATURES_PATH)
    df = make_rul_dataset(feat)
    (out_path or RUL_DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path or RUL_DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    feat = pd.read_parquet(FEATURES_PATH)
    df = save_rul_dataset(feat)
    print("RUL-датасет:", len(df), "строк,", df["battery_id"].nunique(), "батарей")
    print("RUL от", df[RUL_TARGET].min(), "до", df[RUL_TARGET].max())