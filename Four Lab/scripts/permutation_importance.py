# -*- coding: utf-8 -*-
"""Permutation importance каналов на test для сохранённой модели A."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src import config
from src.pipeline import build_dataset


def main() -> int:
    config.configure_runtime()
    data = build_dataset()
    mask = data["masks"]["test"]
    x = data["x"][mask]
    y = data["y"][mask]
    model = tf.keras.models.load_model(config.REPORTS / "experiments" / "A.keras")

    t_mean, t_std = data["target_mean"], data["target_std"]
    base_pred = model.predict(x, verbose=0) * t_std + t_mean
    base_mae = float(np.abs(y - base_pred).mean())

    rng = np.random.default_rng(config.SEED)
    repeats = 5
    result = []
    for feature_idx, name in enumerate(config.FEATURE_COLUMNS):
        maes = []
        for _ in range(repeats):
            x_perm = x.copy()
            order = rng.permutation(len(x))
            x_perm[:, :, feature_idx] = x_perm[order, :, feature_idx]
            pred = model.predict(x_perm, verbose=0) * t_std + t_mean
            maes.append(float(np.abs(y - pred).mean()))
        maes_arr = np.asarray(maes)
        result.append({
            "feature": name,
            "channel": config.CHANNELS[feature_idx],
            "mae_mean": float(maes_arr.mean()),
            "mae_std": float(maes_arr.std()),
            "delta_mae": float(maes_arr.mean() - base_mae),
            "delta_std": float(maes_arr.std()),
        })
    result.sort(key=lambda r: r["delta_mae"], reverse=True)
    payload = {
        "model": "A",
        "split": "test",
        "n_samples": int(len(x)),
        "repeats": repeats,
        "base_mae": base_mae,
        "note": "Перемешивание канала по окнам выборки; delta_mae = MAE после перестановки минус базовая MAE.",
        "features": result,
    }
    out = config.REPORTS / "permutation_importance.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in result:
        print(f"{row['channel']:9s} delta_mae={row['delta_mae']:+.5f} +- {row['delta_std']:.5f}")
    print(f"базовая MAE: {base_mae:.5f}; ИТОГ: permutation importance готова")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
