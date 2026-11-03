"""Baseline-модели (Лаб. №1): константа, линейная регрессия, Random Forest.

Baseline нужны, чтобы измерить, насколько MLP лучше простых методов на том же
сплите по батареям. Принципы те же, что у MLP:
  - обучаются только на train-строках (без утечки test);
  - для линейной регрессии препроцессинг = SimpleImputer(median) +
    StandardScaler, fit только на train;
  - для Random Forest используется median-imputer (fit на train), scaler не
    нужен (древовидная модель инвариантна к монотонным преобразованиям);
  - метрики на тесте — MAE / RMSE / R², формулы совпадают с model.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

_RF_SEED = 42

MODELS = ("constant", "linear", "rf")
MODEL_NAMES = {"constant": "const", "linear": "LR", "rf": "RF"}
MODEL_RU = {
    "constant": "константа",
    "linear": "линейная регрессия",
    "rf": "Random Forest",
}


def baseline_constant(X_train, y_train, X_test) -> np.ndarray:
    """Константный предсказатель: среднее y_train на все строки теста."""
    return np.full(len(X_test), float(np.mean(y_train)))


def baseline_linear(X_train, y_train, X_test) -> np.ndarray:
    """Линейная регрессия: median-imputer + StandardScaler, fit только на train."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    model = LinearRegression()
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def baseline_rf(X_train, y_train, X_test) -> np.ndarray:
    """Random Forest: 100 деревьев; median-imputer (fit на train), без scaler."""
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=_RF_SEED)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


def _metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / R² — те же формулы, что в model.py."""
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_baselines(df, target_col: str, feature_cols: list[str], split: dict) -> dict:
    """Обучить три baseline на train-батареях и оценить на test-батареях.

    df — датасет задачи (battery_rul / battery_voltage / battery_capacity),
    target_col — колонка цели, feature_cols — те же признаки, что у MLP.
    split — dict {"train": [battery_id, ...], "test": [battery_id, ...]}:
    ровно те батареи, что использовал MLP (GroupShuffleSplit, rs=42).

    Возвращает {"constant": ..., "linear": ..., "rf": ...}, где каждое
    значение — {"mae": ..., "rmse": ..., "r2": ...} на тесте.
    """
    train_ids = set(split["train"])
    test_ids = set(split["test"])
    assert train_ids.isdisjoint(test_ids), "пересечение train/test в baseline-сплите"

    m_tr = df["battery_id"].isin(train_ids).to_numpy()
    m_te = df["battery_id"].isin(test_ids).to_numpy()
    assert m_tr.any(), "в train-батареях нет ни одной строки"
    assert m_te.any(), "в test-батареях нет ни одной строки"

    X_train = df.loc[m_tr, feature_cols].astype(float)
    y_train = df.loc[m_tr, target_col].astype(float).to_numpy()
    X_test = df.loc[m_te, feature_cols].astype(float)
    y_test = df.loc[m_te, target_col].astype(float).to_numpy()

    predictors = {
        "constant": baseline_constant,
        "linear": baseline_linear,
        "rf": baseline_rf,
    }
    result = {}
    for name, fn in predictors.items():
        y_pred = fn(X_train, y_train, X_test)
        result[name] = _metrics(y_test, y_pred)
    return result