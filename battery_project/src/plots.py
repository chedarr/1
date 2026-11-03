"""Графики (Лаб. №1).

Для каждой из трёх задач: training_history, predictions, residuals.
Плюс общие: capacity_degradation (все 34 батареи) и rul_frac_distribution (8 батарей).

Единый стиль (ТЗ «Улучшение графиков»): крупный шрифт, толстые линии,
крупные точки, подписи осей с единицами, dpi=150, bbox_inches="tight".
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, RUL_TARGET

# ---- единый стиль (Часть 1.1 ТЗ) ------------------------------------
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
})

TASK_LABELS = {"rul": "RUL_frac", "voltage": "Напряжение", "capacity": "Ёмкость"}
TASK_UNITS = {"rul": "доли", "voltage": "В", "capacity": "Ahr"}


def _label(name: str) -> str:
    return TASK_LABELS.get(name, name)


def _unit(name: str) -> str:
    return TASK_UNITS.get(name, "")


def _grid(ax):
    """Сетка: светлая, тонкая (Часть 1.5 ТЗ)."""
    ax.grid(alpha=0.4, linewidth=1)


def _save(fig, filename: str):
    path = FIGURES_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  график: {path.name}")
    return path


def plot_training_history(name: str, history: dict, stem: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, key, title in (
        (axes[0], "loss", "Loss (MSE)"),
        (axes[1], "mae", "MAE"),
    ):
        ax.plot(history[key], label="train")
        ax.plot(history.get("val_" + key, []), label="val")
        ax.set_title(f"{_label(name)}: {title}", fontsize=20)
        ax.set_xlabel("эпоха")
        _grid(ax)
        ax.legend(fontsize=16, framealpha=0.9, loc="best")
    fig.suptitle(f"Тренировка MLP — {_label(name)}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"{stem or name}_training_history.png")


def plot_predictions(name: str, test_df: pd.DataFrame, target_units: str | None = None,
                     xlim=None, ylim=None, stem: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    label = _label(name)
    unit = _unit(name)
    bats = sorted(test_df["battery_id"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(bats), 1)))
    for b, c in zip(bats, colors):
        g = test_df[test_df["battery_id"] == b]
        ax.scatter(g["y_true"], g["y_pred"], s=60, color=c,
                   label=b if len(bats) <= 12 else None, alpha=0.7)
    lo = min(test_df["y_true"].min(), test_df["y_pred"].min())
    hi = max(test_df["y_true"].max(), test_df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="идеал (y = x)")
    ax.set_xlabel(f"Истинное значение {label} ({unit})")
    ax.set_ylabel(f"Предсказанное значение {label} ({unit})")
    ax.set_title(f"{label}: предсказания vs истина (тест)")
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    _grid(ax)
    if len(bats) <= 12:
        ax.legend(fontsize=16, framealpha=0.9, loc="best")
    fig.tight_layout()
    return _save(fig, f"{stem or name}_predictions.png")


def plot_residuals(name: str, test_df: pd.DataFrame, target_units: str | None = None,
                   stem: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    label = _label(name)
    unit = _unit(name)
    resid = test_df["y_pred"] - test_df["y_true"]
    ax.scatter(test_df["y_true"], resid, s=60, alpha=0.6)
    ax.axhline(0, color="k", ls="--", lw=2)
    ax.set_xlabel(f"Истинное значение {label} ({unit})")
    ax.set_ylabel(f"Остаток (предсказание − истина), {unit}")
    ax.set_title(f"{label}: остатки (тест)")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, f"{stem or name}_residuals.png")


def plot_capacity_degradation(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    bats = sorted(df["battery_id"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(bats), 1)))
    for b, c in zip(bats, colors):
        g = df[df["battery_id"] == b].sort_values("discharge_num")
        ax.plot(g["discharge_num"], g["capacity"], color=c,
                label=b if len(bats) <= 12 else None)
    ax.set_xlabel("номер цикла (цикл)")
    ax.set_ylabel("Ёмкость (Ahr)")
    ax.set_title(f"Деградация ёмкости ({len(bats)} батарей)")
    _grid(ax)
    if len(bats) <= 12:
        ax.legend(fontsize=16, framealpha=0.9, loc="best")
    fig.tight_layout()
    return _save(fig, "capacity_degradation.png")


def plot_rul_distribution(rul_df: pd.DataFrame, stem: str = "rul_frac"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(rul_df[RUL_TARGET], bins=40, range=(0, 1), edgecolor="white")
    ax.set_xlabel("RUL_frac (доли остаточного ресурса, 0–1)")
    ax.set_ylabel("Число строк (циклов)")
    ax.set_title(f"Распределение RUL_frac ({rul_df['battery_id'].nunique()} батарей)")
    _grid(ax)
    fig.tight_layout()
    return _save(fig, f"{stem}_distribution.png")


def make_task_figures(name: str, history: dict, test_df: pd.DataFrame,
                      target_units: str | None = None, xlim=None, ylim=None,
                      stem: str | None = None):
    plot_training_history(name, history, stem=stem)
    plot_predictions(name, test_df, target_units=target_units, xlim=xlim, ylim=ylim, stem=stem)
    plot_residuals(name, test_df, target_units=target_units, stem=stem)