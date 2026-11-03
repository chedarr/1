# -*- coding: utf-8 -*-
"""Графики этапа модели: confusion matrix, кривые обучения, per-class F1, сравнение с baseline.

Все рисунки строятся из сохранённых артефактов, без обучения.
Требования: constrained_layout, легенды вне данных, подписи классов повёрнуты,
cmap='Blues', ничего не перекрывается.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def _load(name: str) -> dict:
    with open(config.REPORTS / name, encoding="utf-8") as handle:
        return json.load(handle)


def plot_confusion(matrix, class_names, path: Path) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix (test)", pad=12)
    ax.set_xlabel("предсказанный класс", labelpad=10)
    ax.set_ylabel("истинный класс", labelpad=10)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            if value == 0:
                continue
            ax.text(
                col, row, str(value), ha="center", va="center", fontsize=9,
                color="white" if value > threshold else "black",
            )
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label("число снимков", labelpad=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(epochs, history["loss"], "-o", label="train")
    axes[0].plot(epochs, history["val_loss"], "--s", label="val")
    axes[0].set_title("loss по эпохам", pad=12)
    axes[0].set_xlabel("epoch", labelpad=8)
    axes[0].set_ylabel("loss", labelpad=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], "-o", label="train")
    axes[1].plot(epochs, history["val_accuracy"], "--s", label="val")
    axes[1].set_title("accuracy по эпохам", pad=12)
    axes[1].set_xlabel("epoch", labelpad=8)
    axes[1].set_ylabel("accuracy", labelpad=8)
    axes[1].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(cnn_per_class: dict, baseline_per_class: dict,
                      class_names: list[str], path: Path) -> None:
    x = np.arange(len(class_names))
    width = 0.4
    cnn_f1 = [cnn_per_class[name]["f1"] for name in class_names]
    base_f1 = [baseline_per_class[name]["f1"] for name in class_names]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(x - width / 2, cnn_f1, width, label="CNN", color="C0")
    ax.bar(x + width / 2, base_f1, width, label="HOG + logreg", color="C1")
    ax.set_title("per-class F1: CNN и HOG + logreg", pad=12)
    ax.set_xlabel("класс", labelpad=10)
    ax.set_ylabel("F1", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(summary: dict, path: Path, cnn_metrics: dict | None = None) -> None:
    if cnn_metrics is not None:
        summary = {**summary, "cnn": cnn_metrics}
    order = ["majority", "random", "hog_logreg", "cnn"]
    labels = ["majority", "random", "HOG + logreg", "CNN"]
    accuracy = [summary[name]["accuracy"] for name in order]
    macro_f1 = [summary[name]["macro_f1"] for name in order]

    x = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars_a = ax.bar(x - width / 2, accuracy, width, label="accuracy", color="C0")
    bars_f = ax.bar(x + width / 2, macro_f1, width, label="macro-F1", color="C1")
    ax.set_title("Сравнение с baseline (test)", pad=12)
    ax.set_xlabel("модель", labelpad=10)
    ax.set_ylabel("значение метрики", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    for bars in (bars_a, bars_f):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = _load("metrics_eurosat.json")
    baseline = _load("baseline_metrics.json")
    summary = _load("summary_baselines.json")
    names = metrics["class_names"] if "class_names" in metrics else config.CLASS_NAMES

    plot_confusion(metrics["test"]["confusion_matrix"], names,
                   config.FIGURES / "04_confusion_matrix.png")
    plot_training(metrics["history"], config.FIGURES / "05_training_curves.png")
    plot_per_class_f1(metrics["test"]["per_class"], baseline["hog_logreg"]["per_class"],
                      names, config.FIGURES / "06_per_class_f1.png")
    control = _load("control_patience_max/metrics_eurosat_patience_max.json")
    plot_baseline_comparison(
        summary,
        config.FIGURES / "07_baseline_comparison.png",
        cnn_metrics={"accuracy": control["test"]["accuracy"],
                     "macro_f1": control["test"]["macro_f1"]},
    )
    print("рисунки: 04_confusion_matrix.png, 05_training_curves.png, "
          "06_per_class_f1.png, 07_baseline_comparison.png")
    print("ИТОГ: графики модели готовы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
