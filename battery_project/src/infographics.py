"""Инфографики (Лаб. №1, часть 2): 10 дополнительных графиков в reports/figures/.
Файлы пронумерованы 01_..10_ — это набор «новых» инфографик (Часть 2):

- 01_comparison_metrics    — сравнение метрик трёх задач (MAE/RMSE/R²)
- 02_per_battery_mae       — MAE по батареям теста (3 subplot)
- 03_training_curves_all   — val MAE по эпохам для трёх задач (нормирован)
- 04_pipeline_schema       — схема пайплайна
- 05_model_architecture    — схема архитектуры сети
- 06_correlation_heatmap   — корреляции признаков RUL_frac (топ-10)
- 07_anomalous_batteries   — аномальные батареи (резкие падения ёмкости)
- 08_feature_importance    — permutation importance для модели RUL_frac
- 09_rul_vs_rul_frac       — RUL_frac в долях и тот же прогноз в пересчёте на циклы
- 10_summary_dashboard     — итоговая сводка (4 subplot)

Единый стиль: seaborn whitegrid, фиксированная палитра, DPI 150,
подписи осей с единицами, заголовки на русском.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .config import (
    ANOMALOUS_BATTERIES,
    FIGURES_DIR,
    METRICS_RUL_PATH,
    RUL_FEATURES,
    RUL_TARGET,
    RUL_TARGET_CYCLES,
)

# ---- единый стиль ------------------------------------------------
C_RUL, C_VOL, C_CAP = "#d62728", "#1f77b4", "#2ca02c"
TASK_LABELS = {"rul": "RUL_frac", "voltage": "Напряжение", "capacity": "Ёмкость"}
TASK_UNITS = {"rul": "доли", "voltage": "В", "capacity": "Ahr"}
TASK_COLORS = {"rul": C_RUL, "voltage": C_VOL, "capacity": C_CAP}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "axes.linewidth": 1.5,
    "axes.titleweight": "bold",
})


def _save(fig, filename: str):
    path = FIGURES_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  инфографика: {path.name}")
    return path


def _mae_per_battery(test_df: pd.DataFrame) -> pd.Series:
    sub = test_df[["battery_id", "y_true", "y_pred"]]
    return sub.groupby("battery_id", sort=True).apply(
        lambda g: float(np.mean(np.abs(g["y_pred"] - g["y_true"]))),
        include_groups=False,
    )


# ---- MLP-хелперы для инфографик ----------------------------------
def _fit_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42, epochs: int | None = None,
             validation_data=None):
    """Обучить ту же архитектуру, что и train_task. Если передать
    validation_data — EarlyStopping по val_loss (как в основном пайплайне),
    иначе — по train loss (не влияет на permutation importance)."""
    import keras
    from keras import layers

    from .config import EARLY_STOP_PATIENCE, MLP_BATCH_SIZE, MLP_EPOCHS

    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    monitor = "val_loss" if validation_data is not None else "loss"
    model.fit(X, y, epochs=epochs or MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, verbose=0,
              validation_data=validation_data,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor=monitor, patience=EARLY_STOP_PATIENCE,
                  restore_best_weights=True)])
    return model


class _ModelWrap(RegressorMixin, BaseEstimator):
    """Обёртка для permutation_importance: модель уже обучена, fit — no-op."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()

    def score(self, X, y):
        return float(r2_score(y, self.predict(X)))


# ---- 1. Сравнение метрик трёх задач ------------------------------
def plot_comparison_metrics(results: dict):
    tasks = ["rul", "voltage", "capacity"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        vals = [results[t]["test"][metric] for t in tasks]
        bars = ax.bar([TASK_LABELS[t] for t in tasks], vals,
                      color=[TASK_COLORS[t] for t in tasks], width=0.55)
        ax.set_title(f"{metric.upper()} (test)")
        for b, v, t in zip(bars, vals, tasks):
            ax.annotate(f"{v:.4f}\n({TASK_UNITS[t]})", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=14)
        if metric == "r2":
            ax.set_ylim(0, 1.05)
        ax.margins(y=0.18)
    fig.suptitle("Сравнение метрик трёх задач регрессии", fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "01_comparison_metrics.png")


# ---- 2. Per-battery MAE ------------------------------------------
def plot_per_battery_mae(test_frames: dict):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, name in zip(axes, ["rul", "voltage", "capacity"]):
        pb = _mae_per_battery(test_frames[name]).sort_values()
        ax.barh(pb.index.astype(str), pb.values, color=TASK_COLORS[name], height=0.6)
        for bar, v in zip(ax.patches, pb.values):
            ax.annotate(f"{v:.3f} {TASK_UNITS[name]}",
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=14)
        ax.set_title(f"{TASK_LABELS[name]}: MAE по батареям теста")
        ax.set_xlabel(f"MAE, {TASK_UNITS[name]}")
        ax.set_xlim(0, max(pb.max() * 1.4, 0.05))
    fig.suptitle("Per-battery MAE (test)", fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "02_per_battery_mae.png")


# ---- 3. Кривые обучения трёх задач --------------------------------
def plot_training_curves_all(histories: dict):
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in ["rul", "voltage", "capacity"]:
        vm = np.asarray(histories[name]["val_mae"], dtype=float)
        vn = vm / max(vm.max(), 1e-12)
        ax.plot(np.arange(1, len(vn) + 1), vn, color=TASK_COLORS[name],
                label=f"{TASK_LABELS[name]} ({len(vn)} эпох)")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("val MAE (нормированный к 0–1)")
    ax.set_title("Кривые обучения: val MAE по эпохам")
    ax.legend(fontsize=16, framealpha=0.9, loc="best")
    fig.tight_layout()
    return _save(fig, "03_training_curves_all.png")


# ---- 4. Схема пайплайна (вертикальная) -------------------------------
def plot_pipeline_schema():
    """Вертикальная схема пайплайна: 8 блоков сверху вниз, стрелки между ними."""
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis("off")

    # блоки сверху вниз: (основной текст, доп. текст, цвет этапа)
    # цвета этапов: данные — голубой, обработка — жёлтый, модели — зелёный,
    # результат — фиолетовый
    blocks = [
        ("Данные: .mat (34 файла)", "", "#d1e8ff"),
        ("Парсер (parser.py)", "парсинг циклов", "#ffe9c7"),
        ("Чистка (clean)", "удаление невозможных значений", "#ffe9c7"),
        ("Признаки (features)", "лаги, производные, цели", "#ffe9c7"),
        ("Три датасета", "RUL (8) · Напряжение (34) · Ёмкость (33)", "#d1e8ff"),
        ("Модели MLP ×3", "Dense(64)→Dropout(0.2)→Dense(32)→Dense(1)", "#d8f0d8"),
        ("Метрики", "MAE · RMSE · R²", "#e8d8f5"),
        ("Отчёт", "lab1_report.html", "#e8d8f5"),
    ]
    bx, bw = 2.25, 9.5          # центр блока по x и ширина
    bh, gap = 1.7, 0.5          # высота блока и зазор под стрелку
    step = bh + gap
    centers = [1.75 + i * step for i in range(len(blocks))]  # снизу вверх

    for (main, sub, fc), yc in zip(blocks, centers):
        box = FancyBboxPatch((bx - bw / 2, yc - bh / 2), bw, bh,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             fc=fc, ec="#444444", lw=1.5, zorder=2)
        ax.add_patch(box)
        if sub:
            ax.text(bx, yc + 0.26, main, ha="center", va="center",
                    fontsize=18, fontweight="bold", zorder=3)
            ax.text(bx, yc - 0.34, sub, ha="center", va="center",
                    fontsize=13, color="#333333", zorder=3)
        else:
            ax.text(bx, yc, main, ha="center", va="center",
                    fontsize=18, fontweight="bold", zorder=3)

    # вертикальные стрелки между блоками
    for i in range(len(blocks) - 1):
        y_bottom = centers[i] - bh / 2   # низ верхнего блока
        y_top = centers[i + 1] + bh / 2  # верх нижнего блока
        ax.annotate("", xy=(bx, y_top), xytext=(bx, y_bottom), zorder=1,
                    arrowprops=dict(arrowstyle="->", lw=3, color="black"))

    ax.text(7, 17.55, "Пайплайн лабораторной работы №1",
            ha="center", va="top", fontsize=20, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_pipeline_schema.png")


# ---- 5. Архитектура сети (крупнее, подписи на русском) ---------------
def plot_model_architecture():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15)
    ax.axis("off")

    layers = [
        ("Вход", "Input(14) — 14 признаков цикла", "0 пар.", "#d1e8ff"),
        ("Полносвязный слой", "Dense(64, ReLU) — 64 нейрона", "960 пар. (14·64 + 64)", "#d8f0d8"),
        ("Dropout (0.2)", "выключение нейронов при обучении", "0 пар.", "#ffe9c7"),
        ("Полносвязный слой", "Dense(32, ReLU) — 32 нейрона", "2080 пар. (64·32 + 32)", "#d8f0d8"),
        ("Выход", "Dense(1) — 1 число · Loss = MSE", "33 пар. (32 + 1)", "#e8d8f5"),
    ]
    bx, bw, bh = 7.0, 9.5, 1.8
    gap = 0.7
    step = bh + gap
    centers = [12.0 - i * step for i in range(len(layers))]  # 12.0, 9.5, 7.0, 4.5, 2.0

    for (name, desc, params, fc), yc in zip(layers, centers):
        box = FancyBboxPatch((bx - bw / 2, yc - bh / 2), bw, bh,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             fc=fc, ec="#444444", lw=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(bx, yc + 0.28, name, ha="center", va="center",
                fontsize=18, fontweight="bold", zorder=3)
        ax.text(bx, yc - 0.36, desc, ha="center", va="center",
                fontsize=13, color="#333333", zorder=3)
        ax.text(12.25, yc + 0.32, params, ha="left", va="center",
                fontsize=12, color="#444444", zorder=3)

    for i in range(len(layers) - 1):
        y_bottom = centers[i] - bh / 2   # низ текущего слоя
        y_top = centers[i + 1] + bh / 2  # верх следующего
        ax.annotate("", xy=(bx, y_top), xytext=(bx, y_bottom), zorder=1,
                    arrowprops=dict(arrowstyle="->", lw=3, color="black"))

    ax.text(bx, 0.55,
            "Всего параметров: 3073 · Optimizer: Adam(1e-3) · EarlyStopping(patience=10) · seed=42",
            ha="center", va="center", fontsize=13, color="#333333")
    ax.text(bx, 14.2, "Архитектура сети MLP (одна для трёх задач)",
            ha="center", va="top", fontsize=20, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_model_architecture.png")


# ---- 6. Тепловая карта корреляций ----------------------------------
def plot_correlation_heatmap(rul_df: pd.DataFrame):
    feats = [f for f in RUL_FEATURES if f in rul_df.columns]
    data = rul_df[feats + [RUL_TARGET]].dropna()
    corr = data.corr()
    top = corr[RUL_TARGET].abs().drop(RUL_TARGET).sort_values(ascending=False).head(10).index
    sub = data[list(top) + [RUL_TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.4, linecolor="white", ax=ax,
                annot_kws={"fontsize": 10}, cbar_kws={"label": "r"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_title(f"Корреляции признаков RUL (топ-10 по |r| с целью {RUL_TARGET})")
    fig.tight_layout()
    return _save(fig, "06_correlation_heatmap.png")


# ---- 7. Аномальные батареи -----------------------------------------
def plot_anomalous_batteries(feat: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    shown = False
    for b in ANOMALOUS_BATTERIES:
        g = feat[feat["battery_id"] == b].sort_values("discharge_num")
        ax.plot(g["discharge_num"], g["capacity"], alpha=0.85, label=b)
        anom = g[g["capacity_ratio"] < 0.5]
        if not anom.empty:
            ax.scatter(anom["discharge_num"], anom["capacity"], s=60, color=C_RUL,
                       zorder=3, label="capacity_ratio < 0.5" if not shown else None)
            shown = True
    ax.set_xlabel("номер цикла (цикл)")
    ax.set_ylabel("Ёмкость (Ahr)")
    ax.set_title("Аномальные батареи: резкие падения ёмкости")
    ax.legend(fontsize=16, framealpha=0.9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return _save(fig, "07_anomalous_batteries.png")


# ---- 8. Feature importance (permutation, без утечки) -----------------
def plot_feature_importance(rul_df: pd.DataFrame, metrics_path=METRICS_RUL_PATH):
    """Permutation importance для RUL_frac на батареях, которых модель не видела.

    Процедура:
      1) train/val/test батареи берём из key split в metrics_rul.json — это ровно
         тот сплит, по которому обучалась доставленная model_rul.keras
         (сверено с src/model.py::_split_by_battery, random_state=42);
      2) SimpleImputer(median) и StandardScaler fit ТОЛЬКО на строках train;
      3) тот же MLP (_fit_mlp) обучается ТОЛЬКО на строках train; val-строки
         идут ТОЛЬКО как сигнал EarlyStopping(patience=10) — ровно как при
         обучении доставленной model_rul.keras (проверено: при seed=42 эта
         модель воспроизводится побитово, max|p_fit − p_model| = 0);
      4) permutation importance (scoring=R², n_repeats=5) считается на
         test-строках — батареях, которых модель никогда не видела.
         Тест не участвует ни в fit, ни в ранней остановке.
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        key_split = json.load(f)["split"]
    train_ids = {str(b) for b in key_split["train"]}
    val_ids = {str(b) for b in key_split["val"]}
    test_ids = {str(b) for b in key_split["test"]}
    assert not (train_ids & test_ids), "train и test батареи пересекаются"
    assert not ((train_ids | val_ids) & test_ids), "test батареи попали в fit"

    bids = rul_df["battery_id"].astype(str).to_numpy()
    is_tr = np.isin(bids, list(train_ids))
    is_va = np.isin(bids, list(val_ids))
    is_te = np.isin(bids, list(test_ids))
    assert is_tr.any() and is_te.any(), "пустая выборка train/test"

    X = rul_df[RUL_FEATURES].astype(float).to_numpy()
    y = rul_df[RUL_TARGET].astype(float).to_numpy()

    # (1) препроцессинг обучается ТОЛЬКО на train-строках
    imputer = SimpleImputer(strategy="median").fit(X[is_tr])
    scaler = StandardScaler().fit(imputer.transform(X[is_tr]))

    def _prep(mask: np.ndarray) -> np.ndarray:
        return scaler.transform(imputer.transform(X[mask])).astype(np.float32)

    Xtr, ytr = _prep(is_tr), y[is_tr]
    Xva, yva = (_prep(is_va), y[is_va]) if is_va.any() else (None, None)
    Xte, yte = _prep(is_te), y[is_te]

    # (2) MLP обучается ТОЛЬКО на train-строках (val — только для EarlyStopping)
    model = _fit_mlp(Xtr, ytr, seed=42, validation_data=(Xva, yva) if Xva is not None else None)

    # (3) importance считается на test-строках
    res = permutation_importance(_ModelWrap(model), Xte, yte, scoring="r2",
                                 n_repeats=5, random_state=42)
    imp = pd.Series(res.importances_mean, index=RUL_FEATURES).sort_values()
    idx = np.argsort(res.importances_mean)
    r2_base = float(r2_score(yte, _ModelWrap(model).predict(Xte)))

    n_tr_b, n_te_b = len(train_ids), len(test_ids)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(imp.index, imp.values, color=C_RUL, height=0.62,
            xerr=res.importances_std[idx], capsize=2.5)
    for bar, v in zip(ax.patches, imp.values):
        ax.annotate(f"{v:.4f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=14)
    ax.axvline(0, color="black", ls="--", lw=2)
    ax.set_xlabel("снижение R² при перетасовке признака (mean ± std)")
    ax.set_title(f"Feature importance (permutation) — модель {RUL_TARGET}")
    ax.text(0.5, -0.13,
            f"без утечки: fit imputer/scaler/MLP — только train ({n_tr_b} бат., {len(ytr)} строк), "
            f"importance — test ({n_te_b} бат., {len(yte)} строк, R² базовой = {r2_base:.2f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=13, color="#333333")
    fig.tight_layout()
    return _save(fig, "08_feature_importance.png")


# ---- 9. RUL_frac: доли и пересчёт в циклы -----------------------------
def plot_rul_vs_rul_frac(test_frames: dict, rul_df: pd.DataFrame):
    """Одна и та же модель RUL_frac в двух единицах: доли (0–1) и циклы.

    Слева — предсказанная доля против истинной, справа — та же величина,
    пересчитанная в циклы (RUL_frac · длина серии батареи) против истинного
    `rul_cycles`. Второе панельное значение получается из сохранённых метрик
    и parquet-датасета, обучение не запускается.
    """
    tf = test_frames["rul"]
    mae_frac = float(np.mean(np.abs(tf["y_pred"] - tf["y_true"])))

    # длина серии каждой батареи (для пересчёта долей в циклы)
    total = (rul_df.groupby("battery_id")["total_cycles"].first()
             .astype(float).to_dict())
    keys = list(zip(tf["battery_id"].astype(str), tf["discharge_num"].astype(int)))
    len_series = np.array([total.get(b, np.nan) for b, _ in keys], dtype=float)
    true_cycles = tf["y_true"].to_numpy(dtype=float) * len_series
    pred_cycles = tf["y_pred"].to_numpy(dtype=float) * len_series
    mae_cycles = float(np.mean(np.abs(pred_cycles - true_cycles)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.scatter(tf["y_true"], tf["y_pred"], s=60, alpha=0.7, color=C_RUL)
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Истинное RUL_frac (доли, 0–1)")
    ax.set_ylabel("Предсказанное RUL_frac (доли, 0–1)")
    ax.set_title(f"RUL_frac в долях: MAE = {mae_frac:.3f} доли")

    ax = axes[1]
    ax.scatter(true_cycles, pred_cycles, s=60, alpha=0.7, color=C_RUL)
    lo = float(min(true_cycles.min(), pred_cycles.min()))
    hi = float(max(true_cycles.max(), pred_cycles.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=2)
    ax.set_xlabel("Истинный остаток (циклы)")
    ax.set_ylabel("Предсказанный остаток (циклы)")
    ax.set_title(f"Тот же прогноз в циклах: MAE = {mae_cycles:.1f} цикл.")

    fig.suptitle(f"RUL_frac: доли и пересчёт в циклы (тест, {len(tf)} строк)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "09_rul_vs_rul_frac.png")


# ---- 10. Итоговая сводка -------------------------------------------
def plot_summary_dashboard(results: dict, test_frames: dict, histories: dict,
                           feat: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # (0,0) метрики трёх задач (группированные столбцы)
    ax = axes[0, 0]
    tasks = ["rul", "voltage", "capacity"]
    metrics = ["mae", "rmse", "r2"]
    x = np.arange(len(tasks))
    width = 0.25
    for i, m in enumerate(metrics):
        vals = [results[t]["test"][m] for t in tasks]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m.upper(),
                      color=["#d62728", "#ff9896", "#b07aa1"][i])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("значение (MAE/RMSE в своих ед., R² безразмерно)")
    ax.set_title("Метрики трёх задач (test)")
    ax.legend(fontsize=14, framealpha=0.9, ncol=3, loc="upper right")

    # (0,1) RUL_frac predictions
    ax = axes[0, 1]
    tf = test_frames["rul"]
    ax.scatter(tf["y_true"], tf["y_pred"], s=60, alpha=0.6, color=C_RUL)
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Истинный RUL_frac (доли, 0–1)")
    ax.set_ylabel("Предсказанный RUL_frac (доли, 0–1)")
    ax.set_title("RUL_frac: предсказания (test)")

    # (1,0) кривые деградации
    ax = axes[1, 0]
    bats = sorted(feat["battery_id"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, len(bats)))
    for b, c in zip(bats, cmap):
        g = feat[feat["battery_id"] == b].sort_values("discharge_num")
        ax.plot(g["discharge_num"], g["capacity"], lw=1.2, color=c, alpha=0.9)
    ax.set_xlabel("номер цикла (цикл)")
    ax.set_ylabel("Ёмкость (Ahr)")
    ax.set_title(f"Деградация ёмкости ({len(bats)} батарей)")

    # (1,1) кривые обучения
    ax = axes[1, 1]
    for name in ["rul", "voltage", "capacity"]:
        vm = np.asarray(histories[name]["val_mae"], dtype=float)
        vn = vm / max(vm.max(), 1e-12)
        ax.plot(np.arange(1, len(vn) + 1), vn, color=TASK_COLORS[name],
                label=TASK_LABELS[name])
    ax.set_xlabel("эпоха")
    ax.set_ylabel("val MAE (норм., 0–1)")
    ax.set_title("Кривые обучения (val MAE)")
    ax.legend(fontsize=14, framealpha=0.9, loc="best")

    fig.suptitle("Итоговая сводка лабораторной работы №1", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "10_summary_dashboard.png")


def build_all(results: dict, datasets: dict | None = None, test_frames: dict | None = None,
              histories: dict | None = None, feat: pd.DataFrame | None = None,
              cleaned: pd.DataFrame | None = None, rul_df: pd.DataFrame | None = None) -> list:
    """Построить все 10 инфографик (вызывается из main.py после обучения)."""
    out = []
    out.append(plot_comparison_metrics(results))
    out.append(plot_per_battery_mae(test_frames))
    out.append(plot_training_curves_all(histories))
    out.append(plot_pipeline_schema())
    out.append(plot_model_architecture())
    out.append(plot_correlation_heatmap(rul_df))
    out.append(plot_anomalous_batteries(feat))
    out.append(plot_feature_importance(rul_df))
    out.append(plot_rul_vs_rul_frac(test_frames, rul_df))
    out.append(plot_summary_dashboard(results, test_frames, histories, feat))
    return out