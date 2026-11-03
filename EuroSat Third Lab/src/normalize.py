# -*- coding: utf-8 -*-
"""Нормализация изображений. Статистики считаются только по train."""
from __future__ import annotations

import json

import numpy as np

from . import config


def compute_channel_stats(images: np.ndarray, batch: int = 1024) -> dict:
    """mean/std по каналам R, G, B без загрузки всего массива в float64."""
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    n = 0
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch].reshape(-1, 3).astype(np.float64)
        total += chunk.sum(axis=0)
        total_sq += np.square(chunk).sum(axis=0)
        n += chunk.shape[0]
    mean = total / n
    var = np.maximum(total_sq / n - mean ** 2, 0.0)
    return {
        "mean_rgb": mean.tolist(),
        "std_rgb": np.sqrt(var).tolist(),
        "n_pixels": int(n),
        "source_split": "train",
        "scale": "0..255",
    }


def save_stats(stats: dict, path=None) -> None:
    path = path or (config.REPORTS / "normalization_stats.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)


def load_stats(path=None) -> dict:
    path = path or (config.REPORTS / "normalization_stats.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def normalize(images: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean_rgb"], dtype=np.float32)
    std = np.asarray(stats["std_rgb"], dtype=np.float32)
    return (images.astype(np.float32) - mean) / std
