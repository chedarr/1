# -*- coding: utf-8 -*-
"""Стратифицированный split 70/15/15 с фиксацией индексов снимков."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import CLASS_NAMES, SEED, SPLIT_FRACTIONS

TOLERANCE = 0.005  # 0.5 п.п. на долю класса в каждом сплите


def stratified_split(
    labels: np.ndarray,
    seed: int = SEED,
    fractions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Делит индексы на train/val/test, сохраняя доли классов."""
    fractions = fractions or SPLIT_FRACTIONS
    labels = np.asarray(labels)
    indices = np.arange(labels.shape[0])

    rest_fraction = fractions["val"] + fractions["test"]
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=rest_fraction,
        stratify=labels,
        random_state=seed,
    )
    val_share_of_rest = fractions["val"] / rest_fraction
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=1.0 - val_share_of_rest,
        stratify=labels[rest_idx],
        random_state=seed,
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def check_no_overlap(splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = np.intersect1d(splits[first], splits[second])
            if shared.size:
                raise AssertionError(
                    f"пересечение {first}/{second}: {shared.size} индексов"
                )


def check_stratification(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    fractions: dict[str, float] | None = None,
    tolerance: float = TOLERANCE,
) -> dict[str, dict[int, float]]:
    """Проверяет, что доли классов в каждом split совпадают с глобальными."""
    fractions = fractions or SPLIT_FRACTIONS
    labels = np.asarray(labels)
    total = max(labels.size, 1)
    global_shares = np.bincount(labels, minlength=len(CLASS_NAMES)) / total
    shares: dict[str, dict[int, float]] = {}
    for name, idx in splits.items():
        size_share = idx.size / total
        if abs(size_share - fractions[name]) > tolerance:
            raise AssertionError(
                f"{name}: размер {size_share:.4f} вместо {fractions[name]:.4f}"
            )
        class_counts = np.bincount(labels[idx], minlength=len(CLASS_NAMES))
        class_shares = class_counts / max(idx.size, 1)
        shares[name] = {
            int(cls): float(share) for cls, share in enumerate(class_shares)
        }
        for cls, share in enumerate(class_shares):
            if abs(share - global_shares[cls]) > tolerance:
                raise AssertionError(
                    f"{name}: класс {CLASS_NAMES[cls]} занимает "
                    f"{share:.4f}, глобально {global_shares[cls]:.4f}"
                )
    return shares


def build_split_info(
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    seed: int = SEED,
) -> dict:
    """Компактное описание split: индекс, класс, имя сплита."""
    labels = np.asarray(labels)
    n_images = int(labels.shape[0])
    split_of = np.empty(n_images, dtype=object)
    for name, idx in splits.items():
        split_of[idx] = name
    if any(value is None for value in split_of):
        raise AssertionError("не все снимки распределены по сплитам")

    counts: dict[str, dict[str, int]] = {}
    for name, idx in splits.items():
        per_class = np.bincount(labels[idx], minlength=len(CLASS_NAMES))
        counts[name] = {
            CLASS_NAMES[cls]: int(count) for cls, count in enumerate(per_class)
        }

    return {
        "seed": seed,
        "n_images": n_images,
        "fractions": dict(SPLIT_FRACTIONS),
        "class_names": list(CLASS_NAMES),
        "index": list(range(n_images)),
        "class_id": [int(value) for value in labels],
        "split": [str(value) for value in split_of],
        "counts": counts,
        "split_sizes": {name: int(idx.size) for name, idx in splits.items()},
    }


def save_split_info(info: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
    )
