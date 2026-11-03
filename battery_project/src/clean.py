"""Минимальная чистка (§3).

Удаляются только физически невозможные значения:
  - capacity < 0 или > 2.6;
  - любые voltage_* < 0;
  - re/rct < 0 или > 1.

НЕ удаляется capacity < 0.5 — только флаг is_low_capacity (для отчёта).

Импеданс (re, rct): per-battery MAD-фильтр |x - median| > 5*MAD → NaN
(строка НЕ удаляется), затем ffill внутри батареи. Обратное (заполнение вперёд) запрещено.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAPACITY_MAX, CLEAN_CYCLES_PATH, IMPEDANCE_MAD_K, LOW_CAP_THRESHOLD, RAW_CYCLES_PATH


def _mad_filter(series: pd.Series, k: float = IMPEDANCE_MAD_K) -> pd.Series:
    """|x - median| > k*MAD → NaN. При MAD == 0 фильтр не применяется (иначе
    порог 0 превратил бы все ряды в NaN)."""
    x = series.astype(float)
    med = x.median()
    mad = np.median((x - med).abs())
    if not np.isfinite(mad) or mad == 0:
        return x
    out = x.where((x - med).abs() <= k * mad)
    return out


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Чистка таблицы discharge-циклов."""
    if df is None:
        df = pd.read_parquet(RAW_CYCLES_PATH)
    out = df.copy()

    # --- жёсткие физические ограничения: строку удаляем
    mask = out["capacity"].notna()
    mask &= ~((out["capacity"] < 0) | (out["capacity"] > CAPACITY_MAX))
    voltage_cols = [c for c in out.columns if c.startswith("voltage_")]
    mask &= ~(out[voltage_cols] < 0).any(axis=1)
    for col in ("re", "rct"):
        v = out[col]
        mask &= ~(v.notna() & ((v < 0) | (v > 1)))
    out = out[mask].reset_index(drop=True)

    # --- флаг низкой ёмкости (НЕ удаляем)
    out["is_low_capacity"] = out["capacity"] < LOW_CAP_THRESHOLD

    # --- импеданс: MAD → NaN, затем ffill внутри батареи
    for col in ("re", "rct"):
        out[col] = out.groupby("battery_id", sort=False)[col].transform(
            lambda s: _mad_filter(s),
        )
        out[col] = out.groupby("battery_id", sort=False)[col].transform(
            lambda s: s.ffill(),
        )
    return out


def save_clean(df: pd.DataFrame | None = None, out_path=None) -> pd.DataFrame:
    out = clean(df)
    (out_path or CLEAN_CYCLES_PATH).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path or CLEAN_CYCLES_PATH, index=False)
    return out


if __name__ == "__main__":
    df = save_clean()
    print(f"Строк после чистки: {len(df)}")
    print(f"is_low_capacity: {int(df['is_low_capacity'].sum())}")
    print(f"re NaN: {int(df['re'].isna().sum())}, rct NaN: {int(df['rct'].isna().sum())}")