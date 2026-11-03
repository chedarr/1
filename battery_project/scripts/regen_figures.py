# -*- coding: utf-8 -*-
"""Пересборка отдельных графиков отчёта из сохранённых артефактов (без обучения).

Перерисовывает:
  - `capacity_degradation.png` и `07_anomalous_batteries.png` — из
    `battery_features.parquet`;
  - `09_rul_vs_rul_frac.png` — предсказания теста RUL восстанавливаются из
    сохранённой `model_rul.keras` (тот же imputer/scaler, обученные на train),
    поэтому числа совпадают с `metrics_rul.json`.

Запуск:

    python scripts/regen_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    FEATURES_PATH,
    METRICS_RUL_PATH,
    MODEL_RUL_PATH,
    RUL_DATASET_PATH,
    RUL_FEATURES,
    RUL_TARGET,
)
from src.infographics import (  # noqa: E402
    plot_anomalous_batteries,
    plot_rul_vs_rul_frac,
)
from src.plots import plot_capacity_degradation  # noqa: E402


def rul_test_frame() -> pd.DataFrame:
    """Тестовые строки RUL с предсказаниями сохранённой model_rul.keras."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from src.model import _split_by_battery

    df = pd.read_parquet(RUL_DATASET_PATH)
    sort_by = (["battery_id", "discharge_num"] if "discharge_num" in df.columns
               else ["battery_id", "cycle_index"])
    df = df.sort_values(sort_by).reset_index(drop=True)
    X = df[RUL_FEATURES].astype(float)
    train_idx, _val_idx, test_idx = _split_by_battery(df)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X.iloc[train_idx])
    scaler.fit(imputer.transform(X.iloc[train_idx]))
    Xte = scaler.transform(imputer.transform(X.iloc[test_idx])).astype(np.float32)

    model = keras_load(MODEL_RUL_PATH)
    test_df = df.iloc[test_idx].copy()
    test_df["y_true"] = df[RUL_TARGET].astype(float).to_numpy()[test_idx]
    test_df["y_pred"] = model.predict(Xte, verbose=0).ravel()

    with METRICS_RUL_PATH.open(encoding="utf-8") as f:
        saved = json.load(f)
    mae = float(np.mean(np.abs(test_df["y_pred"] - test_df["y_true"])))
    print(f"[09] MAE восстановленная = {mae:.6f}, в metrics_rul.json = "
          f"{saved['test']['mae']:.6f}")
    if abs(mae - saved["test"]["mae"]) > 1e-6:
        raise SystemExit("MAE не совпала с metrics_rul.json — цифры на графике были бы другими")
    return test_df[["battery_id", "discharge_num", "y_true", "y_pred"]]


def keras_load(path: Path):
    import keras
    return keras.models.load_model(path)


def main() -> None:
    feat = pd.read_parquet(FEATURES_PATH)
    print("перерисовка из features.parquet:")
    print(" ", plot_capacity_degradation(feat))
    print(" ", plot_anomalous_batteries(feat))

    rul_df = pd.read_parquet(RUL_DATASET_PATH)
    test_df = rul_test_frame()
    print("перерисовка из сохранённой модели:")
    print(" ", plot_rul_vs_rul_frac({"rul": test_df}, rul_df))


if __name__ == "__main__":
    main()
