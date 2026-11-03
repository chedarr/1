# -*- coding: utf-8 -*-
"""Загрузка и очистка 10-минутного ряда GOCE (5 каналов)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Читает 5 parquet, объединяет по timestamp (inner join)."""
    frames = []
    for channel in config.CHANNELS:
        path = config.DATA_RAW / f"{channel}_stats_10min.parquet"
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frames.append(frame[["value_mean"]].rename(columns={"value_mean": f"{channel}_mean"}))
    df = frames[0]
    for frame in frames[1:]:
        df = df.join(frame, how="inner")
    df = df.sort_index()
    return df[config.FEATURE_COLUMNS]


def nan_runs(mask: np.ndarray) -> list[dict]:
    """Возвращает серии подряд идущих True: start, end (включительно), length."""
    runs: list[dict] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append({"start": int(start), "end": int(i - 1), "length": int(i - start)})
            start = None
    if start is not None:
        runs.append({"start": int(start), "end": int(len(mask) - 1), "length": int(len(mask) - start)})
    return runs


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Удаляет строки с NaN и фиксирует длинные NaN-серии (разрывы)."""
    target = df[config.TARGET_FEATURE]
    mask = target.isna().to_numpy()
    runs = nan_runs(mask)
    long_runs = [r for r in runs if r["length"] > config.MAX_GAP_STEPS]

    cleaned = df.dropna(subset=config.FEATURE_COLUMNS).copy()
    info = {
        "n_rows_raw": int(len(df)),
        "n_rows_clean": int(len(cleaned)),
        "n_rows_dropped": int(len(df) - len(cleaned)),
        "nan_fraction_raw": float(mask.mean()),
        "nan_runs_total": len(runs),
        "nan_runs_long": len(long_runs),
        "nan_rows_in_long_runs": int(sum(r["length"] for r in long_runs)),
        "longest_run_steps": int(max((r["length"] for r in runs), default=0)),
        "longest_run_days": round(max((r["length"] for r in runs), default=0) * config.STEP_MINUTES / 1440, 2),
        "max_gap_steps": config.MAX_GAP_STEPS,
        "long_runs": [
            {
                "start": str(df.index[r["start"]]),
                "end": str(df.index[r["end"]]),
                "length": r["length"],
                "days": round(r["length"] * config.STEP_MINUTES / 1440, 2),
            }
            for r in long_runs
        ],
        "time_start": str(df.index.min()),
        "time_end": str(df.index.max()),
        "step_minutes": config.STEP_MINUTES,
    }
    return cleaned, info


def summary(df: pd.DataFrame, cleaned: pd.DataFrame, info: dict) -> dict:
    """Сводка: корреляции каналов и автокорреляция целевой."""
    corr = cleaned.corr(method="pearson").round(6)
    target = cleaned[config.TARGET_FEATURE]
    acf = {}
    for lag in (1, 6, 144, 288):
        acf[str(lag)] = round(float(target.autocorr(lag=lag)), 6)
    info = dict(info)
    info["channels"] = config.CHANNELS
    info["target_channel"] = config.TARGET_CHANNEL
    info["feature_columns"] = config.FEATURE_COLUMNS
    info["channel_correlation"] = {
        row: {col: float(corr.loc[row, col]) for col in corr.columns} for row in corr.index
    }
    info["target_autocorrelation"] = acf
    info["target_stats_clean"] = {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
        "median": float(target.median()),
    }
    return info


def build_data_summary() -> dict:
    """Считает сводку и сохраняет data/interim/data_summary.json."""
    raw = load_raw()
    cleaned, info = clean(raw)
    result = summary(raw, cleaned, info)
    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    (config.DATA_INTERIM / "data_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result
