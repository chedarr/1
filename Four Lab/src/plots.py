# -*- coding: utf-8 -*-
"""Графики: прогноз, ошибка по горизонту, кривые обучения, сравнение моделей."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(alpha=0.3)


def forecast_plot(pred: np.ndarray, truth: np.ndarray, path) -> None:
    """Один окно test: факт и прогноз на 24 часа + scatter."""
    errors = np.abs(pred - truth).mean(axis=1)
    idx = int(np.argsort(errors)[len(errors) // 2])
    steps = np.arange(1, pred.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(steps, truth[idx], "-", color="C0", label="факт", linewidth=2)
    axes[0].plot(steps, pred[idx], "--", color="C1", label="прогноз", linewidth=2)
    _style(axes[0], "Прогноз на 24 часа (медианное окно)", "шаг горизонта, 10 мин", "напряжение, В")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    sample = np.random.default_rng(config.SEED).choice(pred.size, size=min(4000, pred.size), replace=False)
    axes[1].scatter(truth.ravel()[sample], pred.ravel()[sample], s=4, alpha=0.3, color="C2")
    low = float(min(truth.min(), pred.min()))
    high = float(max(truth.max(), pred.max()))
    axes[1].plot([low, high], [low, high], "--", color="black", linewidth=1, label="идеал")
    _style(axes[1], "Прогноз против факта (все шаги test)", "факт, В", "прогноз, В")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mae_by_step(pred: np.ndarray, truth: np.ndarray, path) -> None:
    """MAE по шагам горизонта: модель и persistence."""
    model_mae = np.abs(pred - truth).mean(axis=0)
    persistence_mae = np.abs(truth[:, :1] - truth).mean(axis=0)
    steps = np.arange(1, pred.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(steps, model_mae, "-", color="C0", label="LSTM", linewidth=2)
    ax.plot(steps, persistence_mae, "--", color="C3", label="persistence", linewidth=2)
    _style(ax, "MAE по шагам горизонта (test)", "шаг горизонта, 10 мин", "MAE, В")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(history: dict, path) -> None:
    """loss и MAE по эпохам для train/val."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(epochs, history["loss"], "-o", color="C0", label="train")
    axes[0].plot(epochs, history["val_loss"], "--s", color="C3", label="val")
    _style(axes[0], "loss по эпохам", "epoch", "MSE")
    axes[1].plot(epochs, history.get("mae", []), "-o", color="C0", label="train")
    axes[1].plot(epochs, history.get("val_mae", []), "--s", color="C3", label="val")
    _style(axes[1], "MAE по эпохам", "epoch", "MAE, В")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparison_plot(rows: list[tuple[str, float]], path) -> None:
    """Горизонтальные столбцы MAE: baseline и эксперименты."""
    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in rows]
    values = [value for _, value in rows]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rows) + 1.8), constrained_layout=True)
    colors = ["C7" if name in ("persistence", "constant", "linear") else "C0" for name, _ in rows]
    bars = ax.barh(labels, values, color=colors)
    ax.set_title("MAE на test: baseline и LSTM-конфигурации", pad=12)
    ax.set_xlabel("MAE, В", labelpad=10)
    ax.set_ylabel("модель", labelpad=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)
