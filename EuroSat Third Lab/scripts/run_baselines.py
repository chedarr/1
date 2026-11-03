# -*- coding: utf-8 -*-
"""Baseline-модели: majority, random, HOG + logreg."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from skimage.feature import hog

from src import config
from src.loader import load_arrays
from src.metrics import evaluate


def load_split() -> dict:
    info = json.loads((config.REPORTS / "split_info.json").read_text(encoding="utf-8"))
    split = np.asarray(info["split"])
    return {
        "train": np.flatnonzero(split == "train"),
        "val": np.flatnonzero(split == "val"),
        "test": np.flatnonzero(split == "test"),
    }


def hog_features(images: np.ndarray) -> np.ndarray:
    rows = []
    for image in images:
        gray = rgb2gray(image)
        rows.append(
            hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def main() -> int:
    images, labels = load_arrays()
    split = load_split()
    train, val, test = split["train"], split["val"], split["test"]

    majority = int(np.bincount(labels[train], minlength=config.N_CLASSES).argmax())
    pred_majority = np.full(test.size, majority, dtype=np.int64)
    pred_random = np.random.default_rng(config.SEED).integers(
        0, config.N_CLASSES, size=test.size
    )

    print("HOG: считаю признаки...")
    x_train = hog_features(images[train])
    x_val = hog_features(images[val])
    x_test = hog_features(images[test])
    print(f"HOG: {x_train.shape[1]} признаков")

    best_c, best_f1, best_model = None, -1.0, None
    for c in (0.1, 1.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, n_jobs=-1),
        )
        model.fit(x_train, labels[train])
        f1 = evaluate(labels[val], model.predict(x_val), config.CLASS_NAMES)["macro_f1"]
        print(f"  C={c}: val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1, best_model = c, f1, model

    pred_hog = best_model.predict(x_test)

    results = {
        "majority": evaluate(labels[test], pred_majority, config.CLASS_NAMES),
        "random": evaluate(labels[test], pred_random, config.CLASS_NAMES),
        "hog_logreg": evaluate(labels[test], pred_hog, config.CLASS_NAMES),
        "hog_logreg_C": best_c,
        "hog_logreg_val_macro_f1": best_f1,
        "hog_features": int(x_train.shape[1]),
    }

    (config.REPORTS / "baseline_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    summary = {
        name: {
            "accuracy": results[name]["accuracy"],
            "macro_f1": results[name]["macro_f1"],
        }
        for name in ("majority", "random", "hog_logreg")
    }
    (config.REPORTS / "summary_baselines.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    for name in ("majority", "random", "hog_logreg"):
        m = results[name]
        print(f"{name}: accuracy={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    print(f"ИТОГ: baseline готов (C={best_c})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
