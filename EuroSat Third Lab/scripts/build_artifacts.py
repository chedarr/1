# -*- coding: utf-8 -*-
"""Сборка артефактов этапа обработки данных EuroSAT.

Пишет data_check.json, split_info.json, normalization_stats.json,
leakage_check.json и три графика в reports/figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.augment import build_augmentation  # noqa: E402
from src.loader import load_arrays  # noqa: E402
from src.normalize import compute_channel_stats, save_stats  # noqa: E402
from src.split import (build_split_info, check_no_overlap,  # noqa: E402
                       check_stratification, save_split_info,
                       stratified_split)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, name: str) -> Path:
    path = config.FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def plot_class_distribution(labels: np.ndarray) -> Path:
    counts = np.bincount(labels, minlength=config.N_CLASSES)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    bars = ax.bar(range(config.N_CLASSES), counts, color="#4C72B0")
    ax.set_xticks(range(config.N_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_xlabel("класс", labelpad=8)
    ax.set_ylabel("число снимков", labelpad=8)
    ax.set_title("Распределение снимков по классам", pad=12)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + counts.max() * 0.02,
                str(int(value)), ha="center", va="bottom", fontsize=9)
    return _save(fig, "01_class_distribution.png")


def plot_split_distribution(labels: np.ndarray, splits: dict[str, np.ndarray]) -> Path:
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    width = 0.26
    x = np.arange(config.N_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for offset, name in zip((-width, 0.0, width), ("train", "val", "test")):
        counts = np.bincount(labels[splits[name]], minlength=config.N_CLASSES)
        ax.bar(x + offset, counts, width=width, label=name, color=colors[name])
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_xlabel("класс", labelpad=8)
    ax.set_ylabel("число снимков", labelpad=8)
    ax.set_title("Распределение по split", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="split", loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True)
    return _save(fig, "03_split_distribution.png")


def plot_examples(images: np.ndarray, labels: np.ndarray,
                  per_class: int = 5) -> Path:
    fig, axes = plt.subplots(config.N_CLASSES, per_class, figsize=(10, 20),
                             constrained_layout=True)
    rng = np.random.default_rng(config.SEED)
    for row, class_id in enumerate(range(config.N_CLASSES)):
        idx = np.flatnonzero(labels == class_id)
        chosen = rng.choice(idx, size=per_class, replace=False)
        for col, image_id in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(images[image_id])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(config.CLASS_NAMES[class_id], fontsize=9, labelpad=6)
    fig.suptitle("Примеры снимков по классам", fontsize=13)
    return _save(fig, "02_examples.png")


def main() -> int:
    images, labels = load_arrays()
    print(f"загружено: {images.shape} {images.dtype}, метки {labels.shape}")

    class_counts = {name: int((labels == i).sum())
                    for i, name in enumerate(config.CLASS_NAMES)}
    data_check = {
        "n_images": int(images.shape[0]),
        "n_classes": config.N_CLASSES,
        "class_names": list(config.CLASS_NAMES),
        "image_shape": list(images.shape[1:]),
        "dtype": str(images.dtype),
        "class_distribution": class_counts,
        "license": "MIT (Zenodo 7711810); Sentinel-2 imagery - Copernicus Sentinel Data Terms",
        "alternative_rejected": [
            {"name": "eurosat/all",
             "reason": "13 каналов, архив около 2.07 ГБ; постановка RGB, каналы не нужны"},
            {"name": "Kaggle Satellite Image Classification Dataset",
             "reason": "тот же EuroSAT в другой упаковке, преимуществ нет"},
        ],
        "potential_overlap": {
            "possible": True,
            "reason": "патчи нарезаны из более крупных сцен Sentinel-2; идентификатор сцены в данных не сохранён, поэтому пересечение по сцене нельзя исключить",
        },
    }
    _write_json(config.REPORTS / "data_check.json", data_check)

    splits = stratified_split(labels)
    check_no_overlap(splits)
    stratification = check_stratification(labels, splits)
    info = build_split_info(labels, splits)
    save_split_info(info, config.REPORTS / "split_info.json")

    train_images = images[splits["train"]]
    stats = compute_channel_stats(train_images)
    save_stats(stats)

    _ = build_augmentation()
    leakage = {
        "overlap_train_val": [],
        "overlap_train_test": [],
        "overlap_val_test": [],
        "normalization_only_train": True,
        "augmentation_only_train": True,
        "normalization_source": "train",
        "augmentation_scope": "train",
    }
    _write_json(config.REPORTS / "leakage_check.json", leakage)

    figures = [
        plot_class_distribution(labels),
        plot_examples(images, labels),
        plot_split_distribution(labels, splits),
    ]

    print("split sizes:", {k: int(v.size) for k, v in splits.items()})
    print("share per class (first class):",
          {k: round(v[0], 4) for k, v in stratification.items()})
    print("mean_rgb:", [round(x, 4) for x in stats["mean_rgb"]])
    print("std_rgb:", [round(x, 4) for x in stats["std_rgb"]])
    print("figures:", [p.name for p in figures])
    print("ИТОГ: артефакты собраны")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
