# -*- coding: utf-8 -*-
"""Графики контрольного прогона (patience=max). Базовые рисунки 04-07 не трогает."""
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

CONTROL_DIR = config.REPORTS / "control_patience_max"
FIGURES_DIR = CONTROL_DIR / "figures"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_confusion(payload: dict, path: Path) -> None:
    matrix = np.asarray(payload["test"]["confusion_matrix"])
    names = [row["class_name"] for row in payload["test"]["per_class"]]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(
        f"Confusion matrix (test), patience=max, best_epoch={payload['best_epoch']}",
        pad=12,
    )
    ax.set_xlabel("предсказанный класс", labelpad=10)
    ax.set_ylabel("истинный класс", labelpad=10)
    ticks = np.arange(len(names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            if value == 0:
                continue
            ax.text(col, row, str(value), ha="center", va="center", fontsize=9,
                    color="white" if value > threshold else "black")
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label("число снимков", labelpad=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(payload: dict, path: Path) -> None:
    history = payload["history"]
    epochs = np.asarray(history["epoch"]) + 1
    best = payload["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, train_key, val_key, title in (
        (axes[0], "train_loss", "val_loss", "loss по эпохам"),
        (axes[1], "train_accuracy", "val_accuracy", "accuracy по эпохам"),
    ):
        ax.plot(epochs, history[train_key], "-", color="C0", linewidth=1.6, label="train")
        ax.plot(epochs, history[val_key], "--", color="C1", linewidth=1.6, label="val")
        ax.axvline(best, color="black", linestyle=":", linewidth=1.2,
                   label=f"best_epoch={best}")
        ax.set_title(title, pad=12)
        ax.set_xlabel("epoch", labelpad=8)
        ax.set_ylabel(title.split()[0], labelpad=8)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(payload: dict, baseline: dict, path: Path) -> None:
    rows = payload["test"]["per_class"]
    names = [row["class_name"] for row in rows]
    cnn_f1 = [row["f1"] for row in rows]
    hog_f1 = [baseline["hog_logreg"]["per_class"][name]["f1"] for name in names]

    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(x - width / 2, cnn_f1, width, label="CNN (patience=max)", color="C0")
    ax.bar(x + width / 2, hog_f1, width, label="HOG + logreg", color="C1")
    ax.set_title(f"per-class F1 (test), best_epoch={payload['best_epoch']}", pad=12)
    ax.set_xlabel("класс", labelpad=10)
    ax.set_ylabel("F1", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    payload = _load(CONTROL_DIR / "metrics_eurosat_patience_max.json")
    baseline = _load(config.REPORTS / "baseline_metrics.json")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_confusion(payload, FIGURES_DIR / "confusion_matrix_patience_max.png")
    plot_training(payload, FIGURES_DIR / "training_curves_patience_max.png")
    plot_per_class_f1(payload, baseline, FIGURES_DIR / "per_class_f1_patience_max.png")

    print("рисунки: confusion_matrix_patience_max.png, "
          "training_curves_patience_max.png, per_class_f1_patience_max.png")
    print("ИТОГ: графики контрольного прогона готовы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
