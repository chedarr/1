"""Универсальная модель для трёх задач (Лаб. №1).

Архитектура (одна на все задачи):
    Input(n) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1)
Loss: MSE, метрика: MAE, EarlyStopping(patience=10, restore_best_weights=True).

Сплит строго по батареям (GroupShuffleSplit, test_size=0.25, random_state=42):
  первый сплит → test, внутри оставшихся → val.
NaN в признаках закрываются медианой (fit только на train), затем StandardScaler
(fit только на train).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers

from .config import (
    EARLY_STOP_PATIENCE,
    MLP_BATCH_SIZE,
    MLP_DENSE_UNITS,
    MLP_DROPOUT,
    MLP_EPOCHS,
    RANDOM_STATE,
    TEST_SIZE,
)
from .validate import validate_split


def _build_mlp(n_features: int) -> keras.Model:
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(MLP_DENSE_UNITS[0], activation="relu")(inp)
    x = layers.Dropout(MLP_DROPOUT)(x)
    x = layers.Dense(MLP_DENSE_UNITS[1], activation="relu")(x)
    x = layers.Dense(1)(x)
    model = keras.Model(inp, x)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def _split_by_battery(df: pd.DataFrame, rs: int = RANDOM_STATE):
    """(idx_train, idx_val, idx_test), split строго по батареям."""
    groups = df["battery_id"].to_numpy()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=rs)
    tr_idx, te_idx = next(gss1.split(df, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=rs)
    tr_idx2, va_idx2 = next(gss2.split(df.iloc[tr_idx], groups=groups[tr_idx]))
    train_ids = df["battery_id"].iloc[tr_idx[tr_idx2]].unique()
    val_ids = df["battery_id"].iloc[tr_idx[va_idx2]].unique()
    test_ids = df["battery_id"].iloc[te_idx].unique()
    assert validate_split(train_ids, val_ids, test_ids)
    return tr_idx[tr_idx2], tr_idx[va_idx2], te_idx


def train_task(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    run_name: str,
    model_path,
    metrics_path,
    seed: int = RANDOM_STATE,
) -> dict:
    """Обучить MLP для одной задачи; сохранить модель и метрики.

    df — датасет задачи (уже dropna по целям), со столбцом battery_id.
    Возвращает: metrics dict, history dict, словарь test-предсказаний.
    """
    keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # Сортировка по battery_id и индексу цикла. discharge_num эквивалентен
    # cycle_index (features.py) и в новых датасетах может отсутствовать.
    sort_by = (["battery_id", "discharge_num"]
               if "discharge_num" in df.columns
               else ["battery_id", "cycle_index"])
    df = df.sort_values(sort_by).reset_index(drop=True)
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float).to_numpy()

    train_idx, val_idx, test_idx = _split_by_battery(df, rs=seed)

    # препроцессинг: медиана (NaN) + стандартизация — fit только на train
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(X.iloc[train_idx])
    scaler.fit(imputer.transform(X.iloc[train_idx]))

    Xtr = scaler.transform(imputer.transform(X.iloc[train_idx])).astype(np.float32)
    Xva = scaler.transform(imputer.transform(X.iloc[val_idx])).astype(np.float32)
    Xte = scaler.transform(imputer.transform(X.iloc[test_idx])).astype(np.float32)
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]

    model = _build_mlp(X.shape[1])
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True,
    )
    print(f"[{run_name}] обучение: train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"строк, батарей train/val/test = "
          f"{df['battery_id'].iloc[train_idx].nunique()}/"
          f"{df['battery_id'].iloc[val_idx].nunique()}/"
          f"{df['battery_id'].iloc[test_idx].nunique()}")
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early],
        verbose=0,
    )

    def _metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"mae": mae, "rmse": rmse, "r2": float(r2_score(y_true, y_pred))}

    pred_tr = model.predict(Xtr, verbose=0).ravel()
    pred_va = model.predict(Xva, verbose=0).ravel()
    pred_te = model.predict(Xte, verbose=0).ravel()

    result = {
        "task": run_name,
        "target": target_col,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_rows": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "n_batteries": {
            "train": int(df["battery_id"].iloc[train_idx].nunique()),
            "val": int(df["battery_id"].iloc[val_idx].nunique()),
            "test": int(df["battery_id"].iloc[test_idx].nunique()),
        },
        "train": _metrics(ytr, pred_tr),
        "val": _metrics(yva, pred_va),
        "test": _metrics(yte, pred_te),
        "epochs_stopped": int(len(history.history["loss"])),
        "split": {
            "train": sorted(df["battery_id"].iloc[train_idx].unique().astype(str).tolist()),
            "val": sorted(df["battery_id"].iloc[val_idx].unique().astype(str).tolist()),
            "test": sorted(df["battery_id"].iloc[test_idx].unique().astype(str).tolist()),
        },
    }

    # per-battery MAE на тесте
    test_df = df.iloc[test_idx].copy()
    test_df["y_true"] = yte
    test_df["y_pred"] = pred_te
    per_battery = {}
    for b, g in test_df.groupby("battery_id"):
        per_battery[str(b)] = float(np.mean(np.abs(g["y_true"] - g["y_pred"])))
    result["per_battery_mae_test"] = per_battery

    history_dict = {"loss": history.history.get("loss"),
                    "val_loss": history.history.get("val_loss"),
                    "mae": history.history.get("mae"),
                    "val_mae": history.history.get("val_mae")}
    result["history"] = history_dict

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "history"},
                  f, ensure_ascii=False, indent=2, default=str)

    print(f"[{run_name}] MAE train/val/test = "
          f"{result['train']['mae']:.4f} / {result['val']['mae']:.4f} / "
          f"{result['test']['mae']:.4f}   "
          f"RMSE test = {result['test']['rmse']:.4f}   "
          f"R² test = {result['test']['r2']:.4f}")
    keep = ["battery_id"]
    for c in ("discharge_num", "cycle_index"):
        if c in test_df.columns:
            keep.append(c)
    return result, test_df[keep + ["y_true", "y_pred"]], history_dict


def load_history(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("src/model.py: используй через main.py / scripts/run_all.py")