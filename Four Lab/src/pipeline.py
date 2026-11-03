# -*- coding: utf-8 -*-
"""Сборка выборок: очистка, нормализация по train, окна, split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .features import make_windows
from .loader import clean, load_raw
from .split import assign_windows, chronological_ranges, split_info


def build_dataset(
    lookback: int = config.LOOKBACK,
    horizon: int = config.HORIZON,
    stride: int = config.STRIDE,
    univariate: bool = False,
) -> dict:
    """Возвращает окна, цель, маски сплитов и scaler (fit только по train)."""
    raw = load_raw()
    cleaned, info = clean(raw)
    n_rows = len(cleaned)
    ranges = chronological_ranges(n_rows)

    train_slice = cleaned.iloc[ranges["train"][0] : ranges["train"][1]]
    scaler = StandardScaler().fit(train_slice[config.FEATURE_COLUMNS].to_numpy())

    scaled_values = scaler.transform(cleaned[config.FEATURE_COLUMNS].to_numpy()).astype(np.float32)
    scaled = pd.DataFrame(scaled_values, index=cleaned.index, columns=config.FEATURE_COLUMNS)

    windows = make_windows(scaled, lookback=lookback, horizon=horizon, stride=stride)
    masks = assign_windows(windows["starts"], lookback, horizon, ranges)

    target_idx = windows["target_index"]
    y = windows["y"] * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    x = windows["X"]
    if univariate:
        x = x[..., [target_idx]]

    return {
        "x": x,
        "y": y.astype(np.float32),
        "masks": masks,
        "starts": windows["starts"],
        "origins": windows["origins"],
        "skipped_windows": windows["skipped_windows"],
        "scaler": scaler,
        "target_index": target_idx,
        "target_mean": float(scaler.mean_[target_idx]),
        "target_std": float(scaler.scale_[target_idx]),
        "ranges": ranges,
        "n_rows": n_rows,
        "info": info,
        "lookback": lookback,
        "horizon": horizon,
        "stride": stride,
        "univariate": univariate,
    }


def save_artifacts(data: dict, out_dir: Path = config.DATA_PROCESSED) -> dict:
    """Сохраняет X/y по сплитам, scaler и split_info.json."""
    import pickle

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name in ("train", "val", "test"):
        mask = data["masks"][name]
        x = data["x"][mask]
        y = data["y"][mask]
        np.save(out_dir / f"X_{name}.npy", x)
        np.save(out_dir / f"y_{name}.npy", y)
        metrics[name] = {"x_shape": list(x.shape), "y_shape": list(y.shape)}
    with open(out_dir / "scaler.pkl", "wb") as handle:
        pickle.dump(data["scaler"], handle)

    info = split_info(
        data["n_rows"], data["ranges"], data["starts"], data["masks"],
        data["lookback"], data["horizon"], data["stride"],
    )
    info["skipped_windows"] = int(data["skipped_windows"])
    info["univariate"] = bool(data["univariate"])
    info["target_feature"] = config.TARGET_FEATURE
    info["feature_columns"] = config.FEATURE_COLUMNS
    info["arrays"] = metrics
    import json

    (out_dir / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info
