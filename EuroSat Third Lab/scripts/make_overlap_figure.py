# -*- coding: utf-8 -*-
"""Рисунок near-duplicate пар между train и test (по данным overlap_probe.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.loader import load_arrays


def stretch(image: np.ndarray) -> np.ndarray:
    """Растягивает яркость по перцентилям 2–98, чтобы тёмные снимки были видны."""
    low, high = np.percentile(image, 2), np.percentile(image, 98)
    return np.clip((image.astype(np.float32) - low) / max(high - low, 1.0), 0, 1)


def pick_pairs(pairs: list[dict], limit: int = 3) -> list[dict]:
    """Берёт пары с минимальной MAD, но с разными train- и test-индексами."""
    chosen, seen_train, seen_test = [], set(), set()
    for pair in sorted(pairs, key=lambda item: item["pixel_mad"]):
        if pair["train_index"] in seen_train or pair["test_index"] in seen_test:
            continue
        chosen.append(pair)
        seen_train.add(pair["train_index"])
        seen_test.add(pair["test_index"])
        if len(chosen) == limit:
            break
    return chosen


def main() -> int:
    probe = json.loads((config.REPORTS / "overlap_probe.json").read_text(encoding="utf-8"))
    pairs = pick_pairs(probe["phash_near_duplicates"]["near_duplicate_pairs"])
    images, labels = load_arrays()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for column, pair in enumerate(pairs):
        for row, (kind, index) in enumerate(
            (("test", pair["test_index"]), ("train", pair["train_index"]))
        ):
            ax = axes[row, column]
            ax.imshow(stretch(images[index]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{kind} #{index}", pad=8, fontsize=10)
        axes[1, column].set_xlabel(
            f"{pair['class']}\nMAD = {pair['pixel_mad']:.2f}", labelpad=8, fontsize=10
        )

    fig.suptitle(
        "Near-duplicate пары train / test (SeaLake)\n"
        "яркость растянута по перцентилям 2–98",
        fontsize=13,
    )
    path = config.FIGURES / "08_near_duplicates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"рисунок: {path.name}")
    print("ИТОГ: near-duplicate рисунок готов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
