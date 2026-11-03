# -*- coding: utf-8 -*-
"""Шесть графиков aug v2 и сравнения aug v1 vs aug v2."""
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

AUG_DIR = config.REPORTS / "control_patience_max_aug"
FIG_DIR = AUG_DIR / "figures"
V1 = config.REPORTS / "control_patience_max" / "metrics_eurosat_patience_max.json"
V2 = AUG_DIR / "metrics_eurosat_patience_max_aug.json"
BASE = config.REPORTS / "baseline_metrics.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def names_of(payload: dict) -> list[str]:
    return [row["class_name"] for row in payload["test"]["per_class"]]


def f1_of(payload: dict) -> dict[str, float]:
    return {row["class_name"]: float(row["f1"]) for row in payload["test"]["per_class"]}


def plot_confusion(payload: dict, path: Path) -> None:
    matrix = np.asarray(payload["test"]["confusion_matrix"], dtype=int)
    names = names_of(payload)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(
        f"Confusion matrix (test), aug v2, best_epoch={payload['best_epoch']}",
        pad=12,
    )
    ax.set_xlabel("predicted class", labelpad=10)
    ax.set_ylabel("true class", labelpad=10)
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
            ax.text(
                col, row, str(value), ha="center", va="center", fontsize=9,
                color="white" if value > threshold else "black",
            )
    bar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label("images", labelpad=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(payload: dict, path: Path) -> None:
    history = payload["history"]
    epochs = np.asarray(history["epoch"]) + 1
    best = payload["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, train_key, val_key, title, ylabel in (
        (axes[0], "train_loss", "val_loss", "loss по эпохам", "loss"),
        (axes[1], "train_accuracy", "val_accuracy", "accuracy по эпохам", "accuracy"),
    ):
        ax.plot(epochs, history[train_key], "-", color="C0", lw=1.7, label="train")
        ax.plot(epochs, history[val_key], "--", color="C1", lw=1.7, label="val")
        ax.axvline(best, color="black", ls=":", lw=1.2, label=f"best_epoch={best}")
        ax.set_title(title, pad=10)
        ax.set_xlabel("epoch", labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_v2(payload: dict, path: Path) -> None:
    rows = payload["test"]["per_class"]
    names = [r["class_name"] for r in rows]
    values = [float(r["f1"]) for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    bars = ax.bar(x, values, color="C0", width=0.72, label="CNN (aug v2)")
    ax.set_title(f"per-class F1 (test), aug v2, best_epoch={payload['best_epoch']}", pad=10)
    ax.set_xlabel("class", labelpad=10)
    ax.set_ylabel("F1", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline(v2: dict, baseline: dict, path: Path) -> None:
    models = ["majority", "random", "hog_logreg"]
    labels = ["majority", "random", "HOG + logreg", "CNN (aug v2)"]
    acc = [float(baseline[m]["accuracy"]) for m in models] + [float(v2["test"]["accuracy"])]
    f1 = [float(baseline[m]["macro_f1"]) for m in models] + [float(v2["test"]["macro_f1"])]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    b1 = ax.bar(x - width / 2, acc, width, color="C0", label="accuracy")
    b2 = ax.bar(x + width / 2, f1, width, color="C1", label="macro-F1")
    ax.set_title("Baseline и CNN (aug v2), test", pad=10)
    ax.set_ylabel("score", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_v1_vs_v2_per_class(v1: dict, v2: dict, path: Path) -> None:
    names = names_of(v2)
    a = f1_of(v1)
    b = f1_of(v2)
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    ax.bar(x - width / 2, [a[n] for n in names], width, color="C0", label="aug v1")
    ax.bar(x + width / 2, [b[n] for n in names], width, color="C1", label="aug v2")
    ax.set_title("per-class F1: aug v1 vs aug v2 (test)", pad=10)
    ax.set_xlabel("class", labelpad=10)
    ax.set_ylabel("F1", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_v1_vs_v2_metrics(v1: dict, v2: dict, path: Path) -> None:
    metrics = ["accuracy", "macro-F1"]
    a = [float(v1["test"]["accuracy"]), float(v1["test"]["macro_f1"])]
    b = [float(v2["test"]["accuracy"]), float(v2["test"]["macro_f1"])]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    b1 = ax.bar(x - width / 2, a, width, color="C0", label="aug v1")
    b2 = ax.bar(x + width / 2, b, width, color="C1", label="aug v2")
    ax.set_title("aug v1 vs aug v2: accuracy и macro-F1 (test)", pad=10)
    ax.set_ylabel("score", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    v1 = load(V1)
    v2 = load(V2)
    baseline = load(BASE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(v2, FIG_DIR / "confusion_matrix_aug_v2.png")
    plot_training(v2, FIG_DIR / "training_curves_aug_v2.png")
    plot_per_class_v2(v2, FIG_DIR / "per_class_f1_aug_v2.png")
    plot_baseline(v2, baseline, FIG_DIR / "baseline_comparison_aug_v2.png")
    plot_v1_vs_v2_per_class(v1, v2, FIG_DIR / "aug_v1_vs_v2_per_class.png")
    plot_v1_vs_v2_metrics(v1, v2, FIG_DIR / "aug_v1_vs_v2_metrics.png")
    print("figures:", ", ".join(sorted(p.name for p in FIG_DIR.glob("*.png"))))
    print("v2:", v2["test"]["accuracy"], v2["test"]["macro_f1"])
    print("delta:", v2["test"]["accuracy"] - v1["test"]["accuracy"],
          v2["test"]["macro_f1"] - v1["test"]["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
